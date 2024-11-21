import warnings
from typing import Tuple, Union, Literal, Optional

import torch.nn as nn
from torch import Tensor
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

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


class Bottleneck(BaseModule):
    expansion = 4  # 输出通道数的倍乘

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        spike_cfg=dict(
            spike_mode="lif",
            spike_backend="torch",
            spike_T=4,
        ),
        **kwargs,
    ):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        # conv_layer 1
        self.lif1 = LIFNeuron(**spike_cfg)
        self.layer1 = layer.SeqToANNContainer(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        # conv_layer 2
        self.lif2 = LIFNeuron(**spike_cfg)
        self.layer2 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        # conv_layer 3
        self.lif3 = LIFNeuron(**spike_cfg)
        self.layer3 = layer.SeqToANNContainer(
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion),
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.layer1(self.lif1(x))
        out = self.layer2(self.lif2(out))
        out = self.layer3(self.lif3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


@MODELS.register_module()
class SpikeResNet(BaseModule):

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        frozen_stages=-1,
        spike_cfg=dict(
            spike_mode="lif",
            spike_backend="torch",
            spike_T=4,
        ),
        init_cfg=None,
        train_cls=False,
        **kwargs,
    ):
        super(SpikeResNet, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        self.train_cls = train_cls
        self.depth = depth
        self.num_stages = num_stages

        self.out_indices = out_indices
        assert max(out_indices) < num_stages

        self.frozen_stages = frozen_stages
        self.base_channels = 64
        self.inplanes = self.base_channels

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        # stage 0
        self.lif0 = LIFNeuron(**spike_cfg)
        self.stage0 = layer.SeqToANNContainer(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # stage 1-4
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                spike_cfg=spike_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = (
            self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)
        )

        if self.train_cls:
            self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
            self.fc = nn.Linear(self.feat_dim, 1000)

        self._freeze_stages()

    def make_res_layer(
        self, block, inplanes, planes, num_blocks, stride=1, spike_cfg=None, **kwargs
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LIFNeuron(**spike_cfg),
                layer.SeqToANNContainer(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                ),
            )

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                spike_cfg=spike_cfg,
                **kwargs,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, **kwargs))
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor):
        assert x.ndim == 5, "Input should be a 5D tensor"
        T, B, C, H, W = x.shape

        x = self.stage0(self.lif0(x))

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices and not self.train_cls:
                outs.append(x)

        if self.train_cls:
            x = self.avgpool(x)
            x = x.mean(0)
            x = x.flatten(1)
            x = self.fc(x)
            return x
        else:
            return tuple(outs)


@MODELS.register_module()
class SpikeBackboneWrapper(BaseModule):
    """Backbone for image-event bi-modality detector backbone."""

    def __init__(self, img_backbone_cfg, event_backbone_cfg, init_cfg=None):
        super().__init__(init_cfg)
        self.img_backbone = MODELS.build(img_backbone_cfg)
        self.event_backbone = MODELS.build(event_backbone_cfg)

    def forward(self, img, event):
        return self.img_backbone(img), self.event_backbone(event)
