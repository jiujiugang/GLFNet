import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

def drop_path_f(x, drop_prob: float = 0., training: bool = False):#实现随机深度（Stochastic Depth），也称为Drop Path。这是一种正则化技术
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class LayerNorm11(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        # 通道注意力机制（SE模块）
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // 16, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力机制
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # 融合局部和全局特征的卷积
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)  # 局部特征投影
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)  # 全局特征投影

        # LayerNorm 和 1×1 卷积
        self.ln1 = LayerNorm11(ch_int, eps=1e-6, data_format="channels_first")
        self.ln2 = LayerNorm11(ch_int, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()
        self.conv1x1 = Conv(ch_int, ch_int, 1, bn=True, relu=False)

        # 最终融合的 LayerNorm 和 IRMLP
        self.final_ln = LayerNorm11(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)

        # DropPath
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g):
        # 局部特征处理：输入空间注意力机制
        max_result, _ = torch.max(l, dim=1, keepdim=True)  # 通道最大值
        avg_result = torch.mean(l, dim=1, keepdim=True)  # 通道平均值
        result = torch.cat([max_result, avg_result], 1)  # 拼接最大值和平均值
        l_attention = self.spatial(result)  # 空间注意力卷积
        l_attention = self.sigmoid(l_attention) * l  # 应用空间注意力权重

        # 全局特征处理：输入通道注意力机制
        max_result = self.maxpool(g)  # 全局最大池化
        avg_result = self.avgpool(g)  # 全局平均池化
        max_out = self.se(max_result)  # SE模块处理最大池化结果
        avg_out = self.se(avg_result)  # SE模块处理平均池化结果
        g_attention = self.sigmoid(max_out + avg_out) * g  # 应用通道注意力权重

        # 局部特征和全局特征分别通过 LayerNorm 和 1×1 卷积
        fused_projected = self.gelu(self.conv1x1(l + g))  # 相加并激活

        # 最终拼接三个特征
        fuse = torch.cat([l_attention, g_attention, fused_projected], dim=1)  # 拼接局部、全局和融合特征
        fuse = self.final_ln(fuse)  # 归一化
        fuse = self.residual(fuse)  # Inverted Residual MLP
        fuse = self.drop_path(fuse)  # DropPath

        return fuse

