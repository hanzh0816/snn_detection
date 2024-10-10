from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, ConfigType, OptMultiConfig
from mmdet.structures import OptSampleList
from mmengine.model import xavier_init


from .deformable_detr import DeformableDETR
from ..layers import (
    CdnQueryGenerator,
    SinePositionalEncoding,
    SpikeDinoTransformerEncoder,
    SpikeDinoTransformerDecoder,
)
from ..layers.transformer.spike_dino_ops import SpikeDrivenMSDAttention


@MODELS.register_module()
class SpikeDINO(DeformableDETR):
    """SpikeDINO detector.
    #snn_todo 完成doc
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = SpikeDinoTransformerEncoder(**self.encoder)
        self.decoder = SpikeDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            # snn_todo 实现two_stage
            pass
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def _init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, SpikeDrivenMSDAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(self.reference_points_fc, distribution="uniform", bias=0.0)
        normal_(self.level_embed)

    def pre_transformer(
        self,
        mlvl_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
        spike_t: int = 4,
    ) -> Tuple[Dict]:

        # mlvl_feats: [num_levels, T, bs, embed_dim, h, w]
        batch_size = mlvl_feats[0].size(1)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list])

        # all input shape is same
        if same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
                )
                mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            # spike feats: [T, bs, embed_dim, h, w]
            T, batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[3:].to(feat.device)
            # [T, bs, c, h_lvl, w_lvl] -> [T, bs, h_lvl*w_lvl, c]
            feat = feat.view(T, batch_size, c, -1).permute(0, 1, 3, 2)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (T, bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
            # snn_tag 复制T个时间步的mask
            mask_flatten = mask_flatten.unsqueeze(0).repeat(spike_t, 1, 1)
            mask_flatten = mask_flatten.flatten(0, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])  # (num_level)
        )
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1
            )
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
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

    def pre_decoder(
        self, memory: Tensor, memory_mask: Tensor, spatial_shapes: Tensor
    ) -> Tuple[Dict, Dict]:

        T, batch_size, _, c = memory.shape
        if self.as_two_stage:
            # snn_todo 实现two_stage
            pass
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).repeat(T, 1, 1, 1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query, query_pos=query_pos, memory=memory, reference_points=reference_points
        )
        head_inputs_dict = (
            dict(enc_outputs_class=enc_outputs_class, enc_outputs_coord=enc_outputs_coord)
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict
