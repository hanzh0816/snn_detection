import warnings
from typing import List,Tuple, Union, Literal, Optional

import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule, ModuleList
from mmdet.utils import OptConfigType, OptMultiConfig

from mmdet.registry import MODELS

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


@MODELS.register_module()
class SpikeChannelMapper(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        num_outs: int = None,
        spike_cfg=dict(
            spike_mode="lif",
            spike_backend="torch",
            spike_T=4,
        ),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                nn.Sequential(
                    LIFNeuron(**spike_cfg),
                    layer.SeqToANNContainer(
                        nn.Conv2d(
                            in_channel,
                            out_channels,
                            kernel_size=kernel_size,
                            padding=(kernel_size - 1) // 2,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                    ),
                )
            )
        if num_outs > len(in_channels):
            self.extra_convs = ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    nn.Sequential(
                        LIFNeuron(**spike_cfg),
                        layer.SeqToANNContainer(
                            nn.Conv2d(
                                in_channel,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels),
                        ),
                    )
                )

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)


@MODELS.register_module()
class SpikeNeckWrapper(BaseModule):
    """
    neck wrapper for image-event bi-modality detector.
    """

    def __init__(self, img_neck_cfg, event_neck_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.img_neck = MODELS.build(img_neck_cfg)
        self.event_neck = MODELS.build(event_neck_cfg)

    def forward(self, img, event):
        return self.img_neck(img), self.event_neck(event)
