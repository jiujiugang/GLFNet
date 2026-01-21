import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):#用于确保通道数可以被8整除,目的是优化计算效率。
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite

import torch

import torch
from torch import nn
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch


class ChannelAttention(nn.Module):  # 通道注意力机制
    def __init__(self, in_planes, scaling=16):  # scaling为缩放比例，
        # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):  # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):#提供了 fuse 函数，可以将卷积和BN融合为一个卷积层，用于推理优化。
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):#这是一个残差模块，用于引入跳跃连接 (skip connection)，帮助网络更好地传递梯度，避免梯度消失问题。这个模块中还包含了随机丢弃(dropout)的实现。
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):

        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):#结合了卷积和残差结构，用于特征提取。token_mixer 部分负责处理空间维度信息，channel_mixer 负责处理通道维度信息。
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                CBAM_Attention(inp, kernel_size=3),  # 替换这里为 eca_layer
                #SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()#从当前类的 _modules 属性中获取 BatchNorm (bn) 和 Linear (l) 层的实例
        w = bn.weight / (bn.running_var + bn.eps)**0.5#计算新的权重 w，使用 BatchNorm 的权重和运行方差（running_var）进行归一化。bn.eps 是一个小常数，防止除以零。
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5#计算新的偏置 b，将 BatchNorm 的偏置调整为考虑了输入的均值和权重的影响。
        w = l.weight * w[None, :]#将 Linear 层的权重与归一化后的权重相乘，以得到融合后的权重 w。这里使用 w[None, :] 是为了增加一个维度，使两个权重可以进行逐元素相乘
        if l.bias is None:
            b = b @ self.l.weight.T#检查 Linear 层是否有偏置。如果没有，则将新的偏置 b 乘以 Linear 层权重的转置。
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias#如果 Linear 层有偏置，则首先将 b 转换为列向量，然后与 Linear 层的权重进行矩阵乘法，最后加上 Linear 层的偏置。view(-1) 是为了将结果转换为一维向量。
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)#创建一个新的 Linear 层 m，其输入大小为 w 的第二维（即新的特征数），输出大小为 w 的第一维（即新的输出数）。device=l.weight.device 确保新层与原层在同一设备上
        m.weight.data.copy_(w)#将计算得到的权重 w 复制到新的 Linear 层 m 的权重中
        m.bias.data.copy_(b)#将计算得到的偏置 b 复制到新的 Linear 层 m 的偏置中。
        return m#返回新的融合后的 Linear 层 m

class Classfier(nn.Module):#最终的分类器，包含了一个线性层，并且可以选择是否使用蒸馏技术(distillation)来提升模型的泛化能力
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=3, distillation=False):#cfgs 是网络结构配置，用于定义每个残差块的参数（包括卷积核大小、扩展因子等）。
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer  构建第一个层 patch_embed
        input_channel = self.cfgs[0][2]#从 cfgs 的第一个配置中获取输入通道数，用于定义第一个卷积层的通道数
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]#是 RepViT 中的第一个卷积模块，用于将输入图像进行空间下采样并提取特征。它由两个卷积层和一个激活函数 GELU 组成：
        # building inverted residual blocks,构建倒置残差块
        block = RepViTBlock#每个残差块使用 RepViTBlock，这是核心卷积模块。
        for k, t, c, use_se, use_hs, s in self.cfgs:#从 cfgs 中遍历配置，每个配置定义一个残差块的参数：
            # k: 卷积核大小,t: 扩展因子，决定中间层的特征通道数,c: 输出通道数,use_se: 是否使用 Squeeze-and-Excitation (SE) 模块。use_hs: 是否使用 Hard-Swish 激活函数。s: 卷积步幅。
            output_channel = _make_divisible(c, 8)#确保输出通道数是 8 的倍数，用于确保硬件的高效计算。
            exp_size = _make_divisible(input_channel * t, 8)#计算扩展通道数，input_channel * t 决定了扩展后的特征维度。
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))#将构建的残差块添加到 layers 中。
            input_channel = output_channel#更新输入通道数，以便为下一个块提供正确的输入。
        self.features = nn.ModuleList(layers)#将所有构建的层存储在 self.features 中，使用 nn.ModuleList 来确保其可以被网络的 forward 函数迭代。
        self.classifier = Classfier(output_channel, num_classes, distillation)#构建分类器 classifier
        
    def forward(self, x):
        # x = self.features(x)
        for f in self.features:#依次通过每个模块进行前向计算，提取特征。
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)#使用自适应平均池化将特征图的空间维度压缩为 1x1，即每个通道变为一个标量。.flatten(1)：将 1x1 特征图展平成向量，用于分类器输入。
        x = self.classifier(x)
        return x

from timm.models import register_model


@register_model
def repvit_m0_6(pretrained=False, num_classes = 3, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],## 第一个RepVITBlock，包含 SE 模块
        [3,   2,  40, 0, 0, 1],#第二个RepVITBlock，不包含 SE 模块，Stage 1 结束
        [3,   2,  80, 0, 0, 2],# RepViT block (卷积核大小3x3, 通道数80, 下采样stride=2)
        [3,   2,  80, 1, 0, 1],#Stage 2 是网络中第二个块，第一个 RepVITBlock，包含 SE 模块
        [3,   2,  80, 0, 0, 1],## 第二个 RepVITBlock
        [3,   2,  160, 0, 1, 2],# 使用了3x3卷积，通道数160，下采样，与之前的 Downsample 类似，第二个 Downsample 负责进一步减小特征图的尺寸，并增加通道数。
        [3,   2, 160, 1, 1, 1],#Stage 3 是模型的第三个块，RepVITBlock, 使用 SE 模块
        [3,   2, 160, 0, 1, 1],# RepVITBlock
        [3,   2, 160, 1, 1, 1],# RepVITBlock，使用 SE 模块
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],#通道数增加到320，进行下采样， 第三个 Downsample
        [3,   2, 320, 1, 1, 1],## RepVITBlock，最后一个 Stage，负责在高维度上进行深层次的特征提取。
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)
#[卷积核大小, 步幅, 输出通道数, 是否使用SE模块, 是否使用HS激活函数, 重复次数]
@register_model
def repvit_m0_9(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m1_0(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  56, 1, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  112, 0, 0, 2],
        [3,   2,  112, 1, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  224, 0, 1, 2],
        [3,   2,  224, 1, 1, 1],
        [3,   2,  224, 0, 1, 1],
        [3,   2,  224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 448, 0, 1, 2],
        [3,   2, 448, 1, 1, 1],
        [3,   2, 448, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m1_1(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m1_5(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)



@register_model
def repvit_m2_3(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 0, 2],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  320, 0, 1, 2],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 640, 0, 1, 2],
        [3,   2, 640, 1, 1, 1],
        [3,   2, 640, 0, 1, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]    
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)