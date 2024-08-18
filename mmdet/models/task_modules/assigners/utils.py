# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.structures.bbox import HorizontalBoxes


def select_candidates_in_gts(priors_points: Tensor, gt_bboxes: Tensor, eps: float = 1e-9) -> Tensor:
    """Select the positive priors' center in gt.

    Args:
        priors_points (Tensor): Model priors points,
            shape(num_priors, 2)
        gt_bboxes (Tensor): Ground true bboxes,
            shape(batch_size, num_gt, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): shape(batch_size, num_gt, num_priors)
    """
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])

    priors_number = priors_points.size(0)
    priors_points = priors_points.unsqueeze(0).repeat(batch_size * num_gt, 1, 1)

    # calculate the left, top, right, bottom distance between positive
    # prior center and gt side
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat([priors_points - gt_bboxes_lt, gt_bboxes_rb - priors_points], dim=-1)
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])

    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(
    pos_mask: Tensor, overlaps: Tensor, num_gt: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
        overlaps (Tensor): IoU between all bbox and ground truth,
            shape(batch_size, num_gt, num_priors)
        num_gt (int): Number of ground truth.
    Return:
        gt_idx_pre_prior (Tensor): Target ground truth index,
            shape(batch_size, num_priors)
        fg_mask_pre_prior (Tensor): Force matching ground truth,
            shape(batch_size, num_priors)
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
    """
    fg_mask_pre_prior = pos_mask.sum(axis=-2)

    # Make sure the positive sample matches the only one and is the largest IoU
    if fg_mask_pre_prior.max() > 1:
        mask_multi_gts = (fg_mask_pre_prior.unsqueeze(1) > 1).repeat([1, num_gt, 1])
        index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(index, num_gt)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)

    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


# TODO:'mmdet.BboxOverlaps2D' will cause gradient inconsistency,
# which will be found and solved in a later version.
def yolov6_iou_calculator(bbox1: Tensor, bbox2: Tensor, eps: float = 1e-9) -> Tensor:
    """Calculate iou for batch.

    Args:
        bbox1 (Tensor): shape(batch size, num_gt, 4)
        bbox2 (Tensor): shape(batch size, num_priors, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): IoU, shape(size, num_gt, num_priors)
    """
    bbox1 = bbox1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    bbox2 = bbox2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]

    # calculate xy info of predict and gt bbox
    bbox1_x1y1, bbox1_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    bbox2_x1y1, bbox2_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]

    # calculate overlap area
    overlap = (
        (torch.minimum(bbox1_x2y2, bbox2_x2y2) - torch.maximum(bbox1_x1y1, bbox2_x1y1))
        .clip(0)
        .prod(-1)
    )

    # calculate bbox area
    bbox1_area = (bbox1_x2y2 - bbox1_x1y1).clip(0).prod(-1)
    bbox2_area = (bbox2_x2y2 - bbox2_x1y1).clip(0).prod(-1)

    union = bbox1_area + bbox2_area - overlap + eps

    return overlap / union


def bbox_overlaps(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_mode: str = "ciou",
    bbox_format: str = "xywh",
    siou_theta: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.

    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are ('iou', 'ciou', 'giou', 'siou').
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        siou_theta (float): siou_theta for SIoU when calculate shape cost.
            Defaults to 4.0.
        eps (float): Eps to avoid log(0).

    Returns:
        Tensor: shape (n, ).
    """
    assert iou_mode in ("iou", "ciou", "giou", "siou")
    assert bbox_format in ("xyxy", "xywh")
    if bbox_format == "xywh":
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1)).clamp(0) * (
        torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1)
    ).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]  # cw
    enclose_h = enclose_wh[..., 1]  # ch

    if iou_mode == "ciou":
        # CIoU = IoU - ( (ρ^2(b_pred,b_gt) / c^2) + (alpha x v) )

        # calculate enclose area (c^2)
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # calculate ρ^2(b_pred,b_gt):
        # euclidean distance between b_pred(bbox2) and b_gt(bbox1)
        # center point, because bbox format is xyxy -> left-top xy and
        # right-bottom xy, so need to / 4 to get center point.
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2)) ** 2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) - (bbox1_y1 + bbox1_y2)) ** 2 / 4
        rho2 = rho2_left_item + rho2_right_item  # rho^2 (ρ^2)

        # Width and height ratio (v)
        wh_ratio = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == "giou":
        # GIoU = IoU - ( (A_c - union) / A_c )
        convex_area = enclose_w * enclose_h + eps  # convex area (A_c)
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == "siou":
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # SIoU = IoU - ( (Distance Cost + Shape Cost) / 2 )

        # calculate sigma (σ):
        # euclidean distance between bbox2(pred) and bbox1(gt) center point,
        # sigma_cw = b_cx_gt - b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        # sigma_ch = b_cy_gt - b_cy
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        # sigma = √( (sigma_cw ** 2) - (sigma_ch ** 2) )
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # choose minimize alpha, sin(alpha)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha, sin_beta)

        # Angle cost = 1 - 2 * ( sin^2 ( arcsin(x) - (pi / 4) ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost = Σ_(t=x,y) (1 - e ^ (- γ ρ_t))
        rho_x = (sigma_cw / enclose_w) ** 2  # ρ_x
        rho_y = (sigma_ch / enclose_h) ** 2  # ρ_y
        gamma = 2 - angle_cost  # γ
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (1 - torch.exp(-1 * gamma * rho_y))

        # Shape cost = Ω = Σ_(t=w,h) ( ( 1 - ( e ^ (-ω_t) ) ) ^ θ )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # ω_w
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # ω_h
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), siou_theta) + torch.pow(
            1 - torch.exp(-1 * omiga_h), siou_theta
        )

        ious = ious - ((distance_cost + shape_cost) * 0.5)

    return ious.clamp(min=-1.0, max=1.0)
