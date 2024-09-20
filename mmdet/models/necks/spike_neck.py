# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig

from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)


class LIFNeuron(BaseModule):
    """
    wrapper for unified LIF nueron node interface
    """

    def __init__(
        self,
        spike_mode: Literal["lif", "plif"] = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        backend: Literal["torch", "cupy"] = "torch",
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
        else:
            raise NotImplementedError("Only support LIF/P-LIF spiking neuron")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif_neuron(x)


@MODELS.register_module()
class SpikeChannelMapper(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        num_outs: int = None,
        norm_cfg=dict(type="BN"),
        init_cfg: OptMultiConfig = dict(type="Xavier", layer="Conv2d", distribution="uniform"),
        spike_mode: str = "lif",
        spike_backend: str = "torch",
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)

        self.convs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                    norm_cfg=norm_cfg,
                ),
            )
            self.lifs.append(LIFNeuron(spike_mode=spike_mode, backend=spike_backend))

        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            self.extra_lifs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channels=in_channel,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                        bias=False,
                        norm_cfg=norm_cfg,
                    ),
                )
                self.extra_lifs.append(LIFNeuron(spike_mode=spike_mode, backend=spike_backend))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)

        outs = []
        for i in range(len(inputs)):
            T, B, _, H, W = inputs[i].shape
            out = self.convs[i](self.lifs[i](inputs[i]).flatten(0, 1)).reshape(T, B, -1, H, W)
            outs.append(out)

        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    T, B, _, H, W = inputs[-1].shape
                    outs.append(
                        self.extra_convs[0](self.extra_lifs[0](inputs[-1]).flatten(0, 1)).reshape(
                            T, B, -1, H, W
                        )
                    )
                else:
                    T, B, _, H, W = outs[-1].shape
                    outs.append(
                        self.extra_convs[i](self.extra_lifs[i](outs[-1]).flatten(0, 1)).reshape(
                            T, B, -1, H, W
                        )
                    )
        return tuple(outs)
