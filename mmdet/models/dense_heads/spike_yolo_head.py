# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet.models.layers import multiclass_nms
from mmdet.models.layers.spike_yolo_layer import SpikeConv
from mmdet.models.losses import accuracy
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances, multi_apply, gt_instances_preprocess
from mmengine.dist import get_dist_info
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmdet.utils import (
    ConfigType,
    OptConfigType,
    OptInstanceList,
    OptMultiConfig,
    InstanceList,
    reduce_mean,
)
from mmcv.ops.nms import batched_nms

from ..utils import multi_apply
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class SpikeYOLOHead(BaseDenseHead):
    """SpikeYOLO head."""

    from_channels = {
        "n": [64, 128, 256],
        "s": [128, 256, 512],
        "m": [192, 384, 768],
        "l": [256, 512, 1024],
        "x": [320, 640, 1280],
    }

    def __init__(
        self,
        scale: str,
        num_classes: int = 80,
        reg_max: int = 16,
        prior_generator: ConfigType = dict(
            type="MlvlPointGenerator", offset=0.5, strides=[8, 16, 32]
        ),
        bbox_coder: ConfigType = dict(type="YOLOv8BBoxCoder"),
        loss_cls: ConfigType = dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="none",
            loss_weight=0.5,
        ),
        loss_bbox=dict(
            type="YOLOv8IoULoss",
            iou_mode="ciou",
            bbox_format="xyxy",
            reduction="sum",
            loss_weight=7.5,
            return_iou=False,
        ),
        loss_dfl=dict(type="DistributionFocalLoss", reduction="mean", loss_weight=1.5 / 4),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.scale = scale
        self.num_classes = num_classes
        # DFL channels for bbox regression
        self.reg_max = reg_max
        # number of outputs per anchor
        self.num_outputs = num_classes + self.reg_max * 4

        # get the number of input channels for each scale
        self.in_channels = self.from_channels[self.scale]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # loss settings
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dfl = MODELS.build(loss_dfl)

        self.prior_generator = TASK_UTILS.build(
            prior_generator
        )  # prior generator for bbox prediction
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
        self._init_layers()

    def _init_layers(self):
        bbox_med_ch = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_med_ch = max(self.in_channels[0], min(self.num_classes, 100))  # channels

        self.bbox_decoder = nn.ModuleList(
            nn.Sequential(
                SpikeConv(in_channels=ch, out_channels=bbox_med_ch, kernal_size=3),
                SpikeConv(in_channels=bbox_med_ch, out_channels=bbox_med_ch, kernal_size=3),
                SpikeConv(
                    in_channels=bbox_med_ch,
                    out_channels=self.reg_max * 4,
                    kernal_size=1,
                    bn_flag=False,
                ),
            )
            for ch in self.in_channels
        )

        self.cls_decoder = nn.ModuleList(
            nn.Sequential(
                SpikeConv(in_channels=ch, out_channels=cls_med_ch, kernal_size=3),
                SpikeConv(in_channels=cls_med_ch, out_channels=cls_med_ch, kernal_size=3),
                SpikeConv(
                    in_channels=cls_med_ch,
                    out_channels=self.num_classes,
                    kernal_size=1,
                    bn_flag=False,
                ),
            )
            for ch in self.in_channels
        )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer("proj", proj, persistent=False)

    def _forward_single(self, x: Tensor, bbox_decoder: nn.Module, cls_decoder: nn.Module):
        _, B, _, H, W = x.shape
        cls_preds = cls_decoder(x).mean(0)
        bbox_dist_preds = bbox_decoder(x).mean(0)  # B, reg_max*4, H, W

        if self.reg_max > 1:
            # B, H*W, 4, reg_max
            bbox_dist_preds = bbox_dist_preds.reshape([B, 4, self.reg_max, H * W]).permute(
                0, 3, 1, 2
            )

            # B, H*W, 4
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            # B, 4, H, W
            bbox_preds = bbox_preds.transpose(1, 2).reshape(B, -1, H, W)
        else:
            bbox_preds = bbox_dist_preds  # B, reg_max*4=4, H, W
        if self.training:
            return cls_preds, bbox_preds, bbox_dist_preds
        else:
            return cls_preds, bbox_preds

    def forward(self, inputs: Tuple[Tensor]):
        # scale-wise forward
        # return Tuple[List[cls_preds for _ in scales], List[bbox_preds for _ in scales]]
        return multi_apply(self._forward_single, inputs, self.bbox_decoder, self.cls_decoder)

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],  # B, num_classes, H, W
        bbox_preds: Sequence[Tensor],  # B, 4, H, W
        bbox_dist_preds: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        num_imgs = len(batch_img_metas)

        # Add common attributes to reduce calculation
        self.featmap_sizes_train = None
        self.num_level_priors = None
        self.flatten_priors_train = None
        self.stride_tensor = None

        # [[h,w] for _ in sacles]
        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            # multi-level prior grid info: coord_x, coord_y, stride_w, stride_h
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True,
            )

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xywh
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes, self.stride_tensor[..., 0]
        )

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        assigned_bboxes = assigned_result["assigned_bboxes"]
        assigned_scores = assigned_result["assigned_scores"]
        fg_mask_pre_prior = assigned_result["fg_mask_pre_prior"]

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(flatten_pred_bboxes, prior_bbox_mask).reshape(
                [-1, 4]
            )
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, prior_bbox_mask).reshape(
                [-1, 4]
            )
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(
                -1
            )
            loss_bbox = (
                self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight)
                / assigned_scores_sum
            )

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.reg_max - 1,
                eps=0.01,
            )
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
        )

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        cfg = self.test_cfg if cfg is None else cfg

        # Add common attributes to reduce calculation
        self.featmap_sizes_predict = None
        # self.num_level_priors = None
        self.flatten_priors_predict = None
        self.stride_tensor = None

        num_imgs = len(cls_scores[0])
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        if featmap_sizes != self.featmap_sizes_predict:
            self.featmap_sizes_predict = featmap_sizes
            mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True,
            )
            # self.num_level_priors = [len(n) for n in mlvl_priors]
            self.flatten_priors_predict = torch.cat(mlvl_priors, dim=0)
            # assume stride is the same for x,y dimension
            self.stride_tensor = self.flatten_priors_predict[..., [2]]

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        flatten_bboxes = self.bbox_coder.decode(
            self.flatten_priors_predict[..., :2], flatten_bbox_preds, self.stride_tensor[..., 0]
        )

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask],
                labels=labels[valid_mask],
            )
            result_list.append(self._bbox_post_process(results, cfg, rescale, with_nms, img_meta))
        return result_list

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = False,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        if rescale:
            results.bboxes /= results.bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(
                results.bboxes, results.scores, results.labels, cfg.nms
            )
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results
