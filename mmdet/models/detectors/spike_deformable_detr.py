# Copyright (c) OpenMMLab. All rights reserved.
from ast import Not
from typing import Dict, Optional, Tuple, List, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmengine.model import xavier_init

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList, DetDataSample
from mmdet.utils import OptConfigType

from .base_detr import DetectionTransformer
from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from ..layers import (
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformerEncoder,
    SinePositionalEncoding,
    SpikeQueryGenerator,
)
from .base import BaseDetector

ForwardResults = Union[Dict[str, Tensor], List[DetDataSample], Tuple[Tensor], Tensor]


@MODELS.register_module()
class SpikeDeformableDETR(BaseDetector):
    """
    # snn_todo: 添加docs
    """

    def __init__(
        self,
        phase: str = "single",
        img_backbone: OptConfigType = None,
        event_backbone: OptConfigType = None,
        img_neck: OptConfigType = None,
        event_neck: OptConfigType = None,
        encoder: OptConfigType = None,
        decoder: OptConfigType = None,
        bbox_head: OptConfigType = None,
        positional_encoding: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        spike_query_generator: OptConfigType = None,
        num_queries: int = 300,
        with_box_refine: bool = False,
        as_two_stage: bool = True,
        num_feature_levels: int = 4,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args

        self.phase = phase
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.spike_query_generator = spike_query_generator

        # assert (
        #     self.as_two_stage and self.with_box_refine
        # ), "`as_two_stage`&`with_box_refine`  must be True for this model"

        if bbox_head is not None:
            assert (
                "share_pred_layer" not in bbox_head
                and "num_pred_layer" not in bbox_head
                and "as_two_stage" not in bbox_head
            ), (
                "The two keyword args `share_pred_layer`, `num_pred_layer`, "
                "and `as_two_stage are set in `detector.__init__()`, users "
                "should not set them in `bbox_head` config."
            )
            # Following the setting of Deformable DETR, the last prediction layer
            # is used to generate proposal from event feature map by SNN-Det module
            # when `as_two_stage` is `True`. And all the prediction layers should
            # share parameters when `with_box_refine` is `True`.
            bbox_head["share_pred_layer"] = not with_box_refine
            bbox_head["num_pred_layer"] = (
                (decoder["num_layers"] + 1) if self.as_two_stage else decoder["num_layers"]
            )
            bbox_head["as_two_stage"] = as_two_stage

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        # init model layers
        self.img_backbone = MODELS.build(img_backbone)
        self.neck = MODELS.build(img_neck)

        if self.phase == "fusion":
            self.event_backbone = MODELS.build(event_backbone)
            self.event_neck = MODELS.build(event_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self._init_layers()

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(Tensor(self.num_feature_levels, self.embed_dims))

        if self.phase == "single":
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)
        else:
            # snn_todo 初始化spike query generator
            self.spike_query_generator = SpikeQueryGenerator(**self.spike_query_generator)

    def _init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        # snn_todo 实现init weight
        # super()._init_weights()
        # self._init_transformer_weights(self.modules())
        # for coder in self.encoder, self.decoder:
        #     for p in coder.parameters():
        #         if p.dim() > 1:
        #             nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MultiScaleDeformableAttention):
        #         m.init_weights()
        # if self.as_two_stage:
        #     nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        #     nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        # else:
        #     xavier_init(self.reference_points_fc, distribution="uniform", bias=0.0)
        # normal_(self.level_embed)
        pass

    def forward(
        self,
        inputs: Tensor,
        event: Tensor = None,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        if mode == "loss":
            return self.loss(inputs, event, data_samples)
        elif mode == "predict":
            # raise NotImplementedError("predict mode is not implemented yet")
            return self.predict(inputs, event, data_samples)
        elif mode == "tensor":
            raise NotImplementedError("tensor mode is not implemented yet")
            # return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )

    def loss(
        self,
        batch_imgs: Tensor,
        batch_events: Tensor = None,
        batch_data_samples: OptSampleList = None,
    ):
        img_feats, event_feats = self.extract_feat(batch_imgs, batch_events)
        head_inputs_dict = self.forward_transformer(img_feats, event_feats, batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(
        self,
        batch_imgs: Tensor,
        batch_events: Tensor = None,
        batch_data_samples: SampleList = None,
        rescale: bool = True,
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats, event_feats = self.extract_feat(batch_imgs, batch_events)
        head_inputs_dict = self.forward_transformer(img_feats, event_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self):
        pass

    def extract_feat(self, batch_imgs: Tensor, batch_events: Tensor = None) -> Tuple[Tensor]:
        """
        RGB inputs提取backbone多尺度特征
        """
        if self.phase == "single":
            img_feats = self.img_backbone(batch_imgs)
            if self.with_neck:
                img_feats = self.neck(img_feats)
            return img_feats, None
        elif self.phase == "fusion":
            img_feats = self.img_backbone(batch_imgs)
            event_feats = self.event_backbone(batch_events)
            if self.with_neck:
                img_feats = self.neck(img_feats)
                event_feats = self.event_neck(event_feats)
            return img_feats, event_feats
        else:
            raise ValueError("phase must be single or fusion")

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        event_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & event_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict
        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance`. Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, event_feats, batch_data_samples
        )

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
        self,
        mlvl_img_feats: Tuple[Tensor],
        mlvl_event_feats: Tuple[Tensor],
        batch_data_samples: SampleList,
    ) -> Tuple[Dict]:
        """
        处理图像和事件流特征, 返回encoder_inputs_dict和decoder_inputs_dict,包括多尺度展平后的特征, 位置编码(positional encoding和level encoding), mask, spatial_shapes, level_start_index, valid_ratios

        Args:
            mlvl_img_feats (tuple[Tensor]): 多尺度图像特征, 每个特征图形状为 (bs, dim, h_lvl, w_lvl)
            mlvl_event_feats (tuple[Tensor]): 多尺度事件流特征, 每个特征图形状为 (bs, spike_t, dim, h_lvl, w_lvl)
            batch_data_samples (list[:obj:`DetDataSample`], optional): batch data的元数据, 包括gt_instance, img_shape等.

        Returns:
            tuple[Dict]: encoder_inputs_dict和decoder_inputs_dict
        """
        if mlvl_event_feats is None:
            mlvl_event_feats = []
            for feat in mlvl_img_feats:
                mlvl_event_feats.append(
                    feat.new_zeros((1, feat.size(0), feat.size(1), feat.size(2), feat.size(3)))
                )

        batch_size = mlvl_img_feats[0].size(0)

        # 根据batch中各个sample的input_shape确定mask.
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list])
        if same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_img_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(self.positional_encoding(mask=None, input=feat))
        else:
            masks = mlvl_img_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.
            # 非零值代表忽略的位置, 而零值代表有效位置.
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_img_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
                )
                mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        # 展平多尺度特征和mask
        img_feat_flatten = []
        event_feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (img_feat, event_feat, mask, pos_embed) in enumerate(
            zip(mlvl_img_feats, mlvl_event_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            batch_size, c, _, _ = img_feat.shape
            spike_t, _, _, _, _ = event_feat.shape
            spatial_shape = torch._shape_as_tensor(img_feat)[2:].to(img_feat.device)
            # [B, C, H, W] -> [B, H*W, C]
            img_feat = img_feat.reshape(batch_size, c, -1).permute(0, 2, 1)
            # [T, B, C, H, W] -> [T, B, H*W, C]
            event_feat = event_feat.reshape(spike_t, batch_size, c, -1).permute(0, 1, 3, 2)

            pos_embed = pos_embed.reshape(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape(1, 1, -1)
            # [B, H, W] -> [B, H*W]
            if mask is not None:
                mask = mask.flatten(1)

            img_feat_flatten.append(img_feat)
            event_feat_flatten.append(event_feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (B, num_feat_points, dim), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        img_feat_flatten = torch.cat(img_feat_flatten, 1)
        # (T, B, num_feat_points, dim)
        event_feat_flatten = torch.cat(event_feat_flatten, 2)
        # (B, num_feat_points, dim)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # (B, num_feat_points)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).contiguous().reshape(-1, 2)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])  # (num_level)
        )
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (B, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1
            )
        else:
            valid_ratios = mlvl_img_feats[0].new_ones(batch_size, len(mlvl_img_feats), 2)

        encoder_inputs_dict = dict(
            img_feat=img_feat_flatten,
            event_feat=event_feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(
        self,
        img_feat: Tensor,
        feat_mask: Tensor,
        feat_pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        **kwargs,
    ) -> Dict:
        """
        前向传播encoder部分
        """
        memory = self.encoder(
            query=img_feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        encoder_outputs_dict = dict(
            event_feat=kwargs["event_feat"],
            event_feat_pos=feat_pos,
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return encoder_outputs_dict

    def pre_decoder(
        self,
        event_feat: Tensor,
        event_feat_pos: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        batch_data_samples: OptSampleList = None,
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        预处理decoder输入部分,使用spike_query_generator生成query_pos, query, reference_points输入到decoder中
        """
        if self.phase == "single":
            batch_size, _, c = memory.shape
            spike_outputs_class, spike_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()
        elif self.phase == "fusion":
            # snn_todo 实现spike_query_generator
            query_pos, query, reference_points, spike_outputs_class, spike_outputs_coord = (
                self.spike_query_generator(
                    event_feat=event_feat,
                    event_feat_pos=event_feat_pos,
                    key_padding_mask=memory_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    cls_branch=self.bbox_head.cls_branches[self.decoder.num_layers],
                    reg_branch=self.bbox_head.reg_branches[self.decoder.num_layers],
                    batch_data_samples=batch_data_samples,
                )
            )
        else:
            ValueError("phase must be single or fusion")

        decoder_inputs_dict = dict(
            query=query, query_pos=query_pos, memory=memory, reference_points=reference_points
        )

        # 事件融合之后使用脉冲输出作为Proposal,loss计算时计算spike_outputs_cls和cord
        head_inputs_dict = (
            dict(enc_outputs_class=spike_outputs_class, enc_outputs_coord=spike_outputs_coord)
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Dict:
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,  # for cross_attn
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None,
        )
        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(hidden_states=inter_states, references=references)
        return decoder_outputs_dict

    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
