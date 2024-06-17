from typing import Literal
from numpy import block
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..utils import make_divisible
from ..layers import (
    MS_GetT,
    MS_CancelT,
    MS_ConvBlock,
    MS_Block,
    MS_DownSampling,
    SpikeSPPF,
    SpikeConv,
)


@MODELS.register_module()
class SpikeYOLOBackbone(BaseModule):
    """SpikeYOLO backbone.

    Args:

    """

    arch_settings = {
        "layers": [1, 1, 2, 1, 2, 1, 8, 1, 1, 1],
        "args": [
            [3, 3],  # MS_GetT
            [3, 128, 7, 4, 2],  # MS_DownSampling
            [128, 128],  # MS_ConvBlock
            [128, 256, 3, 2, 1],  # MS_DownSampling
            [256, 256, 3, 2, 1],  # MS_ConvBlock (P3)
            [256, 512, 3, 2, 1],  # MS_DownSampling
            [512, 512],  # MS_Block (P4)
            [512, 1024, 3, 2, 1],  # MS_DownSampling
            [1024, 1024],  # MS_Block
            [1024, 1024],  # SpikeSPPF
        ],
        "layer_names": [
            "MS_GetT",
            "MS_DownSampling",
            "MS_ConvBlock",
            "MS_DownSampling",
            "MS_ConvBlock",
            "MS_DownSampling",
            "MS_Block",
            "MS_DownSampling",
            "MS_Block",
            "SpikeSPPF",
        ],
    }

    scales = {
        "n": [0.33, 0.25, 1024],
        "s": [0.33, 0.50, 1024],
        "m": [0.67, 0.75, 768],
        "l": [1.00, 1.00, 512],
        "x": [1.00, 1.25, 512],
    }

    def __init__(
        self,
        scale: str = "l",
        out_indices: tuple = None,
        T: int = 4,
        mlp_ratio: int = 4,
        num_heads: int = 8,
        full: bool = False,
    ):
        self.blocks = []
        self.layer_names = self.arch_settings["layer_names"]
        self.args = self.arch_settings["args"]
        self.out_indices = out_indices
        self.scale = scale
        self.first_layer = True

        # update layer arguments by model scale
        self.update_layer_args()

        layer_args = {
            "T": T,
            "mlp_ratio": mlp_ratio,
            "num_heads": num_heads,
            "full": full,
        }
        for i, layer_name in enumerate(self.layer_names):
            self.add_module(layer_name, self.make_layer(i, layer_args, layer_name))
            self.blocks.append(layer_name)

    def make_layer(
        self,
        layer_id: int,
        layer_args: dict,
        layer_name: Literal[
            "MS_GetT", "MS_DownSampling", "MS_ConvBlock", "MS_Block", "SpikeSPPF"
        ],
    ) -> BaseModule:
        args = {
            "in_channels": self.args[layer_id][0],
            "out_channels": self.args[layer_id][1],
        }
        if len(self.args[layer_id]) > 2:
            args["kernel_size"] = self.args[layer_id][2]
            args["stride"] = self.args[layer_id][3]
            args["padding"] = self.args[layer_id][4]

        layer_args.update(args)
        if layer_name == "MS_DownSampling":
            layer_args["first_layer"] = self.first_layer
            if self.first_layer == True:
                self.first_layer = False

        module = globals()[layer_name]
        return module(**layer_args)

    def update_layer_args(self):
        self.depth, self.width, self.max_channels = self.scales[self.scale]

        for i, layer_name in enumerate(self.layer_names):
            if i > 0:
                self.args[i][0] = self.args[i - 1][1]
            in_channels, out_channels = self.args[i]

            if layer_name == "SpikeSPPF":
                out_channels = make_divisible(
                    min(out_channels, self.max_channels) * self.width, 8
                )
            elif layer_name == "MS_DownSampling":
                out_channels = int(out_channels * self.width)

            self.args[i][1] = out_channels

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.blocks):
            block = getattr(self, layer_name)
            x = block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
