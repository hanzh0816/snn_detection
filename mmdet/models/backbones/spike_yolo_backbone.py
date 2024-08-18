from typing import List, Literal, Optional
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from ..utils import make_divisible
from ..layers import (
    MS_GetT,
    MS_CancelT,
    MS_ConvBlock,
    MS_AllConvBlock,
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
        "layer_repeat": [1, 1, 3, 1, 6, 1, 9, 1, 1, 1],
        "args": [
            [3, 3],  # MS_GetT
            [3, 128, 7, 4, 2],  # MS_DownSampling
            [128, 128, 7, 4],  # MS_AllConvBlock
            [128, 256, 3, 2, 1],  # MS_DownSampling
            [256, 256, 7, 4],  # MS_AllConvBlock (P3)
            [256, 512, 3, 2, 1],  # MS_DownSampling
            [512, 512, 7, 3],  # MS_ConvBlock (P4)
            [512, 1024, 3, 2, 1],  # MS_DownSampling
            [1024, 1024, 7, 2],  # MS_ConvBlock
            [1024, 1024],  # SpikeSPPF
        ],
        "layer_names": [
            "MS_GetT",
            "MS_DownSampling",
            "MS_AllConvBlock",
            "MS_DownSampling",
            "MS_AllConvBlock",
            "MS_DownSampling",
            "MS_ConvBlock",
            "MS_DownSampling",
            "MS_ConvBlock",
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
        out_indices: List[int] = [4, 6, 9],
        T: int = 4,
        full: bool = False,
        init_cfg: dict = None,
    ):

        super(SpikeYOLOBackbone, self).__init__(init_cfg)
        self.blocks = []
        self.layer_names = self.arch_settings["layer_names"]
        self.args = self.arch_settings["args"]
        self.layer_repeat = self.arch_settings["layer_repeat"]
        self.out_indices = out_indices
        self.scale = scale
        self.first_layer = True

        # update layer arguments by model scale
        self.update_layer_args()

        layer_args = {
            "T": T,
            "full": full,
        }
        for i, layer_name in enumerate(self.layer_names):
            module_name = layer_name + f"{i}"
            self.add_module(
                module_name,
                self.make_layer(i, layer_args, self.layer_repeat[i], layer_name),
            )
            self.blocks.append(module_name)

    def make_layer(
        self,
        layer_id: int,
        layer_args: dict,
        layer_repeat: int,
        layer_name: Literal[
            "MS_GetT", "MS_DownSampling", "MS_ConvBlock", "MS_AllConvBlock", "SpikeSPPF"
        ],
    ) -> BaseModule:
        args = {
            "in_channels": self.args[layer_id][0],
            "out_channels": self.args[layer_id][1],
        }
        # downsampling parameters
        if len(self.args[layer_id]) == 5:
            args["kernel_size"] = self.args[layer_id][2]
            args["stride"] = self.args[layer_id][3]
            args["padding"] = self.args[layer_id][4]

        # ConvBlock parameters
        if len(self.args[layer_id]) == 4:
            args["sep_kernel_size"] = self.args[layer_id][2]
            args["mlp_ratio"] = self.args[layer_id][3]

        layer_args.update(args)
        if layer_name == "MS_DownSampling":
            layer_args["first_layer"] = self.first_layer
            if self.first_layer == True:
                self.first_layer = False

        module = globals()[layer_name]

        return (
            (nn.Sequential(*(module(**layer_args) for _ in range(layer_repeat))))
            if layer_repeat > 1
            else module(**layer_args)
        )

    def update_layer_args(self):
        self.depth, self.width, self.max_channels = self.scales[self.scale]

        for i, layer_name in enumerate(self.layer_names):
            if i > 0:
                self.args[i][0] = self.args[i - 1][1]
            out_channels = self.args[i][1]

            # calculate output channels
            if layer_name == "SpikeSPPF":
                out_channels = make_divisible(
                    min(out_channels, self.max_channels) * self.width, 8
                )
            elif layer_name == "MS_DownSampling":
                out_channels = int(out_channels * self.width)

            # update output channels
            if layer_name == "MS_ConvBlock" or layer_name == "MS_AllConvBlock":
                self.args[i][1] = self.args[i][0]
            else:
                self.args[i][1] = out_channels

            # update repeat times
            n = self.layer_repeat[i]
            self.layer_repeat[i] = max(round(n * self.depth), 1) if n > 1 else n

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.blocks):
            block = getattr(self, layer_name)
            x = block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
