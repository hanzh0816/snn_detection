from typing import List, Literal, Tuple
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
    MS_DownSampling,
    SpikeSPPF,
    SpikeConv,
    Concat,
)


@MODELS.register_module()
class SpikeYOLONeck(BaseModule):
    backbone_out_channels = [256, 512, 1024]  # P3,P4,P5
    scales = {
        "n": [0.33, 0.25, 1024],
        "s": [0.33, 0.50, 1024],
        "m": [0.67, 0.75, 768],
        "l": [1.00, 1.00, 512],
        "x": [1.00, 1.25, 512],
    }
    arch_settings = {
        "layer_names": [
            "SpikeConv",
            "Upsample",
            "MS_ConvBlock",
            "Concat",
            "SpikeConv",
            "Upsample",
            "MS_AllConvBlock",  # change
            "Concat",
            "SpikeConv",
            "MS_AllConvBlock",  # change
            "SpikeConv",
            "Concat",
            "MS_ConvBlock",
            "SpikeConv",
            "Concat",
            "MS_ConvBlock",
        ],
        "args": [
            [1024, 512, 1, 1, None],  # SpikeConv
            [(1, 2, 2), "nearest"],  # Upsample
            [512, 512, 7, 3],  # MS_ConvBlock
            [-2, 2],  # Concat P4, 512+512=1024
            [1024, 256, 1, 1, None],  # SpikeConv
            [(1, 2, 2), "nearest"],  # Upsample
            [256, 256, 7, 4],  # MS_AllConvBlock
            [-3, 2],  # Concat P3, 256+256=512
            [512, 256, 1, 1, None],  # SpikeConv
            [256, 256, 7, 4],  # MS_AllConvBlock ->输出层P3
            [256, 256, 3, 2, None],  # SpikeConv
            [4, 2],  # Concat, 256+256=512
            [512, 512, 7, 3],  # MS_ConvBlock ->输出层P4
            [512, 512, 3, 2, None],  # SpikeConv
            [0, 2],  # Concat, 512+512=1024
            [1024, 1024, 7, 1],  # MS_ConvBlock ->输出层P5
        ],
    }

    def __init__(
        self,
        scale: str,
        out_indices: List[int] = [9, 11, 15],
        full: bool = False,
        init_cfg: dict = None,
        spike_mode: str = "lif",
        lif_backend: str = "torch",
    ):
        super(SpikeYOLONeck, self).__init__(init_cfg)
        self.blocks = []
        self.scale = scale
        self.args = self.arch_settings["args"]
        self.layer_names = self.arch_settings["layer_names"]

        self.out_indices = out_indices
        self.from_indices = []

        self.update_layer_args()
        layer_args = {
            "spike_mode": spike_mode,
            "lif_backend": lif_backend,
            "full": full,
        }
        for i, layer_name in enumerate(self.layer_names):
            module_name = layer_name + f"{i}"
            self.add_module(
                module_name,
                self.make_layer(i, layer_args, layer_name),
            )
            self.blocks.append(module_name)

    def make_layer(self, layer_id, layer_args, layer_name):
        if layer_name == "Concat":
            return Concat(dimension=self.args[layer_id][1])

        if layer_name == "Upsample":
            return nn.Upsample(scale_factor=self.args[layer_id][0], mode=self.args[layer_id][1])

        args = {
            "in_channels": self.args[layer_id][0],
            "out_channels": self.args[layer_id][1],
        }
        # SpikeConv parameters
        if len(self.args[layer_id]) == 5:
            args["kernel_size"] = self.args[layer_id][2]
            args["stride"] = self.args[layer_id][3]
            args["padding"] = self.args[layer_id][4]

        # ConvBlock parameters
        if len(self.args[layer_id]) == 4:
            args["sep_kernel_size"] = self.args[layer_id][2]
            args["mlp_ratio"] = self.args[layer_id][3]
        layer_args.update(args)
        module = globals()[layer_name]
        return module(**layer_args)

    def update_layer_args(self):
        self.depth, self.width, self.max_channels = self.scales[self.scale]

        # update neck outputs channels by scale factor
        # changes output channels follow the update code: int(in_channels * width)
        self.backbone_out_channels[0] = int(self.backbone_out_channels[0] * self.width)
        self.backbone_out_channels[1] = int(self.backbone_out_channels[1] * self.width)
        # SpikeSPPF channels update
        self.backbone_out_channels[2] = make_divisible(
            min(self.backbone_out_channels[2], self.max_channels) * self.width, 8
        )

        # record previous layer out channels
        out_channels = 0
        self.in_channels = []
        for i, layer_name in enumerate(self.layer_names):
            if layer_name == "Upsample":
                self.in_channels.append(out_channels)
                continue

            if layer_name == "Concat":
                self.from_indices.append(self.args[i][0])

            if i == 0:
                self.args[i][0] = self.backbone_out_channels[-1]
            elif layer_name != "Concat":
                self.args[i][0] = out_channels

            if layer_name == "SpikeConv":
                out_channels = min(
                    int(self.args[i][1] * self.width),
                    int(self.max_channels * self.width),
                )
            elif layer_name == "MS_ConvBlock" or layer_name == "MS_AllConvBlock":
                out_channels = self.args[i][0]
            elif layer_name == "Concat":
                concat_layer_id = self.args[i][0]
                # 分类讨论与Backbone连接还是与neck连接
                if concat_layer_id < 0:
                    assert self.backbone_out_channels[concat_layer_id] == out_channels
                    out_channels = self.backbone_out_channels[concat_layer_id] + out_channels
                else:
                    assert self.in_channels[concat_layer_id] == out_channels
                    out_channels = self.in_channels[concat_layer_id] + out_channels
            else:
                raise NotImplementedError(f"{layer_name} is not supported.")

            # update layer args
            self.in_channels.append(out_channels)
            if layer_name != "Concat":
                self.args[i][1] = out_channels

    def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        x = inputs[-1]
        outs = []
        froms = []
        for i, layer_name in enumerate(self.blocks):
            block = getattr(self, layer_name)

            if "Concat" in layer_name:
                concat_layer_id = self.args[i][0]
                if concat_layer_id < 0:
                    x = block([x, inputs[concat_layer_id]])
                else:
                    x = block([x, froms[concat_layer_id]])
            else:
                x = block(x)

            if i in self.out_indices:
                outs.append(x)
            if i in self.from_indices:
                froms.append(x)
            else:
                froms.append(None)
        return tuple(outs)
