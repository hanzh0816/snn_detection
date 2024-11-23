# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union, Literal

import torch
from torch import Tensor, nn

from mmengine.model import BaseModule, ModuleList
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.scale import LayerScale
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.utils import ConfigType, OptConfigType
from mmdet.structures import SampleList

from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import (
    MultiStepIFNode,
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)


class LIFNeuron(BaseModule):
    """
    wrapper for unified LIF nueron node interface
    """

    def __init__(
        self,
        spike_mode: Literal["lif", "plif", "if"] = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = False,
        backend: Literal["torch", "cupy"] = "torch",
        **kwargs,
    ):
        super().__init__()
        if spike_mode == "lif":
            self.lif_neuron = MultiStepLIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
            )
        elif spike_mode == "plif":
            self.lif_neuron = MultiStepParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
            )
        elif spike_mode == "if":
            self.lif_neuron = MultiStepIFNode(
                v_threshold=v_threshold,
                v_reset=v_reset,
                detach_reset=detach_reset,
                backend=backend,
            )
        else:
            raise NotImplementedError("Only support LIF/P-LIF spiking neuron")

    def forward(self, x: Tensor) -> Tensor:
        return self.lif_neuron(x)


class SpikeFFN(BaseModule):

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels: int = 1024,
        num_fcs=2,
        add_identity=True,
        spike_cfg: OptConfigType = dict(
            spike_mode="lif",
            spike_backend="torch",
            spike_T=4,
        ),
        init_cfg=None,
        layer_scale_init_value=0.0,
    ):
        super().__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    LIFNeuron(**spike_cfg),
                    layer.SeqToANNContainer(
                        nn.Linear(in_channels, feedforward_channels),
                    ),
                )
            )
            in_channels = feedforward_channels

        layers.append(
            nn.Sequential(
                LIFNeuron(**spike_cfg),
                layer.SeqToANNContainer(nn.Linear(feedforward_channels, embed_dims)),
            )
        )
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            # snn_todo  LayerScale可能需要修改以适应脉冲数据
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.gamma2(out)
        if identity is None:
            identity = x
        return identity + out


class SpikeQueryGeneratorLayer(BaseModule):
    """
    Generate position query & content query from event data using SNN for Spike Deformable DETR Decoder.
    """

    def __init__(
        self,
        self_attn_cfg: OptConfigType = dict(embed_dims=256, num_heads=8, dropout=0.0),
        ffn_cfg: OptConfigType = dict(
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
        ),
        norm_cfg: OptConfigType = dict(type="BN"),
        init_cfg: OptConfigType = None,
        spike_cfg: OptConfigType = dict(
            spike_mode="lif",
            spike_backend="torch",
            spike_T=4,
        ),
        **kwargs,
    ) -> None:
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
        self.norm_cfg = norm_cfg  # 弃用参数
        self.spike_cfg = spike_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        self.lif = LIFNeuron(**self.spike_cfg)
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = SpikeFFN(**self.ffn_cfg)

        # snn_tag 此处使用BN，不确定BN和LN孰优孰劣
        norms_list = [layer.SeqToANNContainer(nn.BatchNorm1d(self.embed_dims)) for _ in range(2)]
        self.norms = ModuleList(norms_list)

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reference_points: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        SpikeQueryGeneratorLayer前向推理过程

        Args:
            query (Tensor): 输入的query, shape为[T, B, N, embed_dims]
            query_pos (Tensor): 输入的query位置信息, shape为[T, B, N, embed_dims]
            key_padding_mask (Tensor): 输入的key padding mask, shape为[T*B, N]

        Returns:
            Tensor: 输出的query, shape为[T, B, N, embed_dims]
        """
        # [T,B,N,C]
        T, B, N, C = query.shape

        # lif1
        query = self.lif(query)

        # self-attention & norm1
        query = query.flatten(0, 1)
        query_pos = query_pos.flatten(0, 1)  # [T*B,N,C]
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            **kwargs,
        )
        query = query.reshape(T, B, N, C).contiguous()
        query = query.permute(0, 1, 3, 2)  # [T,B,C,N]
        query = self.norms[0](query)
        query = query.permute(0, 1, 3, 2).contiguous()  # [T,B,N,C]

        # ffn
        query = self.ffn(query)

        # norm2
        query = query.permute(0, 1, 3, 2)  # [T,B,C,N]
        query = self.norms[1](query)
        query = query.permute(0, 1, 3, 2).contiguous()  # [T,B,N,C]

        return query


class SpikeQueryGenerator(BaseModule):
    def __init__(
        self,
        num_layers: int,
        num_queries: int,
        layer_cfg: ConfigType,
        init_cfg: OptConfigType = None,
    ) -> None:
        super(SpikeQueryGenerator, self).__init__(init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.layers = nn.ModuleList(
            [SpikeQueryGeneratorLayer(**self.layer_cfg) for _ in range(self.num_layers)]
        )
        self.embed_dims = self.layers[0].embed_dims

        self.memory_trans = nn.Sequential(
            LIFNeuron(**self.layer_cfg["spike_cfg"]),
            layer.SeqToANNContainer(nn.Linear(self.embed_dims, self.embed_dims)),
        )
        # snn_tag 此处用了BN
        self.memory_norm = layer.SeqToANNContainer(nn.BatchNorm1d(self.embed_dims))
        self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)

    def forward(
        self,
        event_feat: Tensor,
        event_feat_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        cls_branch: nn.Linear,
        reg_branch: nn.Sequential,
        **kwargs,
    ) -> Tensor:
        """
        SpikeQueryGenerator前向推理过程

        Args:
            event_feat (Tensor): 输入的事件特征, shape为[T, B, N, embed_dims]
            event_feat_pos (Tensor): 位置编码, shape为[B, N, embed_dims]
            key_padding_mask (Tensor): mask掉无效的key, shape为[B, N]
            spatial_shapes (Tensor): 每个level的shape, shape为[num_levels, 2]
            level_start_index (Tensor): 每个level的起始索引, shape为[num_levels]
            valid_ratios (Tensor): 每个level的有效比例, shape为[B, num_levels, 2]
        Returns:
            query_pos (Tensor): 输出query的position编码, shape为[B, num_queries, embed_dims]
            query (Tensor): 输出query的content编码, shape为[B, num_queries, embed_dims]
            enc_out_reference_points (Tensor): 输出query的reference points, shape为[B, num_queries, 4]
            spike_outputs_class (Tensor): 输出对事件流产生proposal的分类结果, shape为[B, N, 2]
            spike_outputs_coord (Tensor): 输出对事件流产生proposal的坐标结果, shape为[B, N, 4]
        """

        T, _, _, _ = event_feat.shape
        # event_feat_pos和key_padding_mask重复T次,与event_feat对齐
        event_feat_pos = event_feat_pos.unsqueeze(0).repeat(T, 1, 1, 1)
        event_key_padding_mask = key_padding_mask
        if event_key_padding_mask is not None:
            # 有mask时,重复T次,与event_feat对齐
            event_key_padding_mask = (
                event_key_padding_mask.unsqueeze(0).repeat(T, 1, 1, 1).flatten(0, 1)
            )  # [T*B, N]

        # encoder输入的reference_points
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=event_feat.device
        )

        # reference_points的shape为[T, B, N, 4, 2]
        # snn_todo 为什么reference_points的shape是这个
        reference_points = reference_points.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        reference_points = reference_points.flatten(0, 1)  # [T*B, N, 4, 2]

        for layer in self.layers:
            event_feat = layer(
                query=event_feat,
                query_pos=event_feat_pos,
                key_padding_mask=event_key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
            )

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            event_feat, key_padding_mask, spatial_shapes
        )

        # 这里的output_memory已经对时间维度取平均,可以直接计算cls和reg
        # snn_todo 重新构造head,这里只用到了前景-背景预测,不应该使用和其他cls相同的head
        enc_outputs_class = cls_branch(output_memory)
        enc_outputs_coord_unact = reg_branch(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        topk_proposals = torch.topk(enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords_unact = topk_coords_unact.detach()
        enc_out_reference_points = topk_coords_unact.sigmoid()
        pos_trans_out = self.pos_trans_fc(self.get_proposal_pos_embed(topk_coords_unact))
        pos_trans_out = self.pos_trans_norm(pos_trans_out)
        query_pos, query = torch.split(pos_trans_out, self.embed_dims, dim=2)

        return query_pos, query, enc_out_reference_points, enc_outputs_class, enc_outputs_coord

    def gen_encoder_output_proposals(
        self, memory: Tensor, memory_mask: Tensor, spatial_shapes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        生成预定义的proposal

        Args:
            memory (Tensor): 输入的memory, shape为[T, B, N, embed_dims]
            memory_mask (Tensor): 用于生成Proposal的mask, shape为[B, N]
            spatial_shapes (Tensor): 每个level的shape, shape为[B, num_levels, 2]

        Returns:
            output_memory (Tensor): 输出的memory, shape为[B, N, embed_dims]
            output_proposals (Tensor): 输出的proposal, shape为[B, N, 4]
        """

        T, B, _, _ = memory.shape
        memory = memory.mean(0)  # [B, N, embed_dims]
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur : (_cur + H * W)].view(B, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(B, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(B, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(B, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).sum(
            -1, keepdim=True
        ) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(memory_mask.unsqueeze(-1), float(0))

        output_proposals_valid = output_proposals_valid.unsqueeze(0).repeat(T, 1, 1, 1)
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.memory_trans(output_memory)  # [T,B,N,C]
        output_memory = output_memory.permute(0, 1, 3, 2)  # [T,B,C,N]
        output_memory = self.memory_norm(output_memory).permute(0, 1, 3, 2).mean(0)  # [B,N,C]

        # [B, N, embed_dims], [B, N, 2]
        return output_memory, output_proposals

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

    @staticmethod
    def get_proposal_pos_embed(
        proposals: Tensor, num_pos_feats: int = 128, temperature: int = 10000
    ) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
