from typing import Literal, Optional
import warnings
import numpy as np

from sympy import Li
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule

from mmdet.registry import MODELS

from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)


@torch.jit.script
def jit_mul(x, y):
    return x.mul(y)


@torch.jit.script
def jit_sum(x):
    return x.sum(dim=[-1, -2], keepdim=True)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class LIFNeuron(BaseModule):
    """
    wrapper for unified LIF nueron node interface
    """

    def __init__(
        self,
        spike_mode: Literal["lif", "plif"] = "lif",
        tau: float = 2.0,
        v_threshold: float = 0.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        backend: str = "torch",
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


class BNAndPadLayer(BaseModule):
    """
    BNAndPadLayer combines Batch Normalization (BN) with optional padding.

    Attributes:
        pad_pixels (int): Number of pixels to pad on each side of the input.
        bn (nn.BatchNorm2d): Batch normalization layer.
    """

    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            assert self.bn.running_mean and self.bn.running_var
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )

            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class MS_GetT(BaseModule):
    """
    generate temporal data from standard input
    """

    def __init__(self, T=4, **kwargs):
        super().__init__()
        self.T = T

    def forward(self, x):
        if len(x.shape) == 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:  # T,
            x = x.transpose(0, 1)  # B,T,C,H,W -> T,B,C,H,W
        return x


class MS_CancelT(BaseModule):
    """
    cancel tempral dimension from input
    """

    def __init__(self, T=4, **kwargs):
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert input has temporal dimension [T,B,C,H,W]
        assert len(x.shape) == 5, "input is not temporal data"
        x = x.mean(0)
        return x


class MS_DownSampling(BaseModule):
    """
    downsamping layer including spike neuron(using membrane shortcut)
    Layer: LIF(optional) ==> encode_conv ==> encode_bn
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        **kwargs,
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.first_layer = first_layer
        self.encode_bn = nn.BatchNorm2d(out_channels)
        if not first_layer:
            self.encode_lif = LIFNeuron(spike_mode="lif")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _, _, _ = x.shape

        if not self.first_layer:
            x = self.encode_lif(x)

        x = self.encode_conv(x.flatten(0, 1))
        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class SepConv(BaseModule):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        in_channels,
        expansion_ratio=2,
        bias=False,
        kernel_size=3,  # 7,3
        padding=1,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * in_channels)
        self.lif1 = LIFNeuron()
        self.pwconv1 = nn.Conv2d(
            in_channels, med_channels, kernel_size=1, stride=1, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = LIFNeuron()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv

        self.pwconv2 = nn.Conv2d(
            med_channels, in_channels, kernel_size=1, stride=1, bias=bias, groups=4
        )
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        T, B, _, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.bn2(self.pwconv2(self.dwconv(x.flatten(0, 1)))).reshape(T, B, -1, H, W)
        return x


class RepConv(BaseModule):
    """
    RepConv pipeline: 1x1Conv ==> bn ==> 3x3Conv

    Attributes:
        full: whether to use all dims to conv.
    """

    def __init__(
        self, in_channels, out_channel, kernel_size=3, full=True, bias=False, group=1
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        conv1x1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=bias, groups=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channels)
        if full:
            dw_conv = nn.Conv2d(
                in_channels, in_channels, kernel_size, 1, 0, groups=group, bias=bias
            )  # using all channel to conv
        else:
            dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                1,
                0,
                groups=in_channels,
                bias=bias,
            )  # using depth-wise conv
        conv3x3 = nn.Sequential(
            dw_conv,
            nn.Conv2d(in_channels, out_channel, 1, 1, 0, groups=group, bias=bias),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class SpikeConv(BaseModule):
    """
    ConvBlock Class with spike neuron
    pipeline: lif ==> Conv ==> BN
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernal_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
    ):
        super().__init__()
        self.s = stride
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            stride=stride,
            padding=autopad(kernal_size, padding, dilation),  # type:ignore
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.lif = LIFNeuron(spike_mode="lif")
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        T, B, _, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x


class MS_ConvBlock(BaseModule):
    """
    ConvBlock Class with MS shortcut
    pipeline: x ==> SepConv(shortcut) ==> [lif+SepConv+BN]x2(shortcut) ==> x

    Attributes:
        full (bool): FullRepConv or RepConv for the first convolution layer, full means full dims to conv.
    """

    def __init__(
        self, in_channels, mlp_ratio=4.0, sep_kernel_size=7, full=False, **kwargs
    ):
        super().__init__()

        self.full = full
        self.conv = SepConv(
            in_channels=in_channels, expansion_ratio=2, kernel_size=sep_kernel_size
        )
        self.mlp_ratio = mlp_ratio
        self.hidden_channel = int(in_channels * mlp_ratio)

        self.lif1 = LIFNeuron()
        self.lif2 = LIFNeuron()

        self.conv1 = RepConv(in_channels, self.hidden_channel, full=full, group=4)
        self.bn1 = nn.BatchNorm2d(self.hidden_channel)

        self.conv2 = RepConv(self.hidden_channel, in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv(x) + x  # sepconv:pw+dw+pw

        x_feat = x

        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(
            T, B, int(self.mlp_ratio * C), H, W
        )
        # repconv，对应conv_mixer，包含1*1,3*3,1*1三个卷积，等价于一个3*3卷积
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MetaSDSA(BaseModule):
    def __init__(
        self,
        in_channels,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode: Literal["lif", "plif"] = "lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            in_channels % num_heads == 0
        ), f"dim {in_channels} should be divided by num_heads {num_heads}."
        self.dim = in_channels
        self.dvs = dvs
        self.num_heads = num_heads
        # self.pool = Erode()
        self.qkv_conv = nn.Sequential(
            RepConv(
                in_channels=in_channels, out_channel=in_channels * 2, kernel_size=3
            ),
            nn.BatchNorm2d(in_channels * 2),
        )
        self.register_parameter(
            "scale", nn.Parameter(torch.tensor([0.0]))  # type:ignore
        )
        self.qk_lif = LIFNeuron(spike_mode=spike_mode)
        self.v_lif = LIFNeuron(spike_mode=spike_mode)
        self.proj_lif = LIFNeuron(spike_mode=spike_mode)
        self.talking_heads_lif = LIFNeuron(v_threshold=0.5, spike_mode=spike_mode)
        self.shortcut_lif = LIFNeuron(spike_mode=spike_mode)
        self.proj_conv = nn.Sequential(
            RepConv(in_channels, in_channels), nn.BatchNorm2d(in_channels)
        )

        self.mode = mode
        self.layer = layer

    def forward(
        self, x, hook=None
    ):  # 这里本质上只是用了transformer的算子，而没有切片等操作
        T, B, C, H, W = x.shape
        identity = x

        x = self.shortcut_lif(x)  # 把x变为脉冲 [T,B,C,H,W]
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        # [T,B,C,H,W]→[T,B,2C,H,W] .通过repconv使通道数加倍,一个是qk，一个是v
        qkv_conv_out = self.qkv_conv(x.flatten(0, 1)).reshape(T, B, 2 * C, H, W)
        qk, v = qkv_conv_out.chunk(2, dim=2)
        del qkv_conv_out

        qk = self.qk_lif(qk)  # 变为脉冲形式
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk_lif"] = qk.detach()

        qk = jit_sum(qk)  # [T,B,C,H,W]→[T,B,C,1,1]
        qk = self.talking_heads_lif(qk)  # 再变为脉冲形式。只要有数就是1，无数就是0
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_qk"] = qk.detach()

        v = self.v_lif(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v"] = v.detach()

        x = jit_mul(qk, v)  # 做哈达玛积，可以理解为通过qk对v进行mask。qk包含了全局信息
        del qk, v
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        x = self.proj_lif(x)
        x = x * self.scale

        # x = identity - x
        return x


class MS_MLP_SMT(BaseModule):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNeuron(spike_mode="lif")

        self.dw_lif = LIFNeuron(spike_mode="lif")
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNeuron(spike_mode="lif")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x_feat = self.dw_lif(x).reshape(T, B, self.c_hidden, H, W)
        x_feat = self.dwconv(x_feat.flatten(0, 1)).reshape(T, B, self.c_hidden, N)

        x = self.fc2_lif(x + x_feat)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Block(BaseModule):
    def __init__(
        self,
        in_channels,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        **kwargs,
    ):
        super().__init__()

        self.attn = MetaSDSA(
            in_channels,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.conv = SpikeConv(in_channels, in_channels)

        # self.attn = MS_Attention_RepConv(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        #     sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = MS_MLP_SMT(
            in_features=in_channels, hidden_features=mlp_hidden_dim, drop=drop
        )

    def forward(self, x):
        x = x + self.attn(x) + self.conv(x)  # 尝试添加并行的卷积
        x = x + self.mlp(x)

        return x


class SpikeSPPF(BaseModule):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(
        self, in_channels, out_channels, kernal_size=5, **kwargs
    ):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SpikeConv(in_channels, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernal_size, stride=1, padding=kernal_size // 2
        )

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(torch.cat((x, y1, y2, y3), 2))
