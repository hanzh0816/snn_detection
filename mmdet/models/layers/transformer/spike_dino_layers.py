# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
from typing import Tuple, Union, Literal, Optional

import torch
from torch import nn
from torch import Tensor

from mmengine.model import BaseModule, ModuleList
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from mmdet.utils.typing_utils import ConfigType, ConfigDict
from mmcv.cnn import build_norm_layer

from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .detr_layers import DetrTransformerDecoderLayer
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
from .spike_dino_ops import SpikeDrivenMSDAttention, SpikeDrivenSelfAttention, SpikeFFN

from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)


class SpikeDinoTransformerEncoder(BaseModule):
    """Decoder of SpikeDino.
    # todo 完成doc
    Args:
    """

    def __init__(
        self,
        num_layers: int,
        layer_cfg: ConfigType,
        num_cp: int = -1,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self):
        self.layers = ModuleList(
            [SpikeDinoTransformerEncoderLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
        if self.num_cp > 0:
            # todo 实现checkpointing
            raise NotImplementedError("checkpointing is not implemented yet")

        self.embed_dims = self.layers[0].embed_dims

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        spike_t: int = 4,
        **kwargs
    ):
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device
        )
        # snn_tag 复制t个时间步
        reference_points = reference_points.unsqueeze(1).repeat(spike_t, 1, 1, 1, 1).flatten(0, 1)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs
            )
        return query

    @staticmethod
    def get_encoder_reference_points(
        spatial_shapes: Tensor, valid_ratios: Tensor, device: Union[torch.device, str]
    ) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class SpikeDinoTransformerEncoderLayer(BaseModule):
    """
    #todo 完成doc
    """

    def __init__(
        self,
        self_attn_cfg: OptConfigType = dict(embed_dims=256, num_heads=8, dropout=0.0),
        ffn_cfg: OptConfigType = dict(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        norm_cfg: OptConfigType = dict(type="LN"),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.self_attn_cfg = self_attn_cfg
        if "batch_first" not in self.self_attn_cfg:
            self.self_attn_cfg["batch_first"] = True
        else:
            assert (
                self.self_attn_cfg["batch_first"] is True
            ), "First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag."

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self):
        self.self_attn = SpikeDrivenMSDAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = SpikeFFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(2)]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        level_start_index: Tensor,
        reference_points: Tensor,
        **kwargs
    ) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query


class SpikeDinoTransformerDecoder(BaseModule):
    def __init__(
        self,
        num_layers: int,
        layer_cfg: ConfigType,
        return_intermediate: bool = True,
        init_cfg: Union[dict, ConfigDict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self._init_layers()

    def _init_layers(self):
        self.layers = ModuleList(
            [SpikeDinoTransformerDecoderLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
        self.embed_dims = self.layers[0].embed_dims

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: Optional[nn.Module] = None,
        spike_t: int = 4,
        **kwargs
    ) -> Tuple[Tensor]:
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            # snn_tag 重复t个时间步
            reference_points_input = (
                reference_points_input.unsqueeze(0).repeat(spike_t, 1, 1, 1, 1).flatten(0, 1)
            )
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs
            )
            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[..., :2] + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                # spike tensor to normal tensor
                mean_output = output.mean(0)
                intermediate.append(mean_output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        # spike tensor to normal tensor
        mean_output = output.mean(0)
        return mean_output, reference_points


class SpikeDinoTransformerDecoderLayer(DetrTransformerDecoderLayer):
    def _init_layers(self):
        self.self_attn = SpikeDrivenSelfAttention(**self.self_attn_cfg)
        self.cross_attn = SpikeDrivenMSDAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = SpikeFFN(**self.ffn_cfg)
        norms_list = [build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)]
        self.norms = ModuleList(norms_list)
