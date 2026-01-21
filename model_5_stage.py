from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn import init
from timm.models.registry import register_model

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block1(nn.Module):#  Only  AS
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)  # 用于变换通道数
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),  # 3x3卷积减少通道数
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积恢复通道数
            nn.Sigmoid()  # 归一化
        )

    def forward(self, x):
        # 直接使用注意力机制，不再有原先的卷积操作
        input = x
        # 计算注意力权重
        attention_weights = self.attention(x)
        # 通过drop_path和残差连接
        x = self.drop_path(x)
        # 应用注意力权重
        x = x + x * attention_weights
        return x


class Block(nn.Module):# demo + AS
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),  # 3x3卷积减少通道数
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积恢复通道数
            nn.Sigmoid()  # 归一化
        )


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        # 添加注意力机制
        attention_weights = self.attention(x)  # 计算注意力权重
        x = input + self.drop_path(x)
        x =  x + x * attention_weights  # 将注意力权重应用到特征图上
        return x


class Block1(nn.Module):# AS +demo
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),  # 3x3卷积减少通道数
            nn.ReLU(),  # 激活函数
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),  # 3x3卷积恢复通道数
            nn.Sigmoid()  # 归一化
        )

    def forward(self, x):
        input = x
        attention_weights = self.attention(x)  # 计算注意力权重
        x = input + self.drop_path(x)
        x = x + x * attention_weights  # 将注意力权重应用到特征图上

        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=256, depths=[3, 4,4, 3,3], mlp_ratio=4, drop_path_rate=0.0, **kwargs):
        super().__init__()
        self.in_channel = base_dim // 4
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.adaptive_pool(x)
        #x = torch.flatten(x, start_dim=1)  # 变为 (batch, channels)
        return x


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
        self.depthwise_conv = nn.Conv2d(dim * mlp_mult, dim * mlp_mult, kernel_size=3, padding=1, groups=dim * mlp_mult)

    def forward(self, x):
        out = self.net[0](x)  # 第一个卷积层
        #out = self.depthwise_conv(out)  ## 深度可分离卷积层
        # GELU激活函数
        out = self.net[1](out)
        # dropout
        out = self.net[2](out)
        # 第二个卷积层
        out = self.net[3](out)
        # dropout
        out = self.net[4](out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class HTNet(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            heads,
            num_hierarchies,
            block_repeats,
            mlp_mult=4,
            channels=3,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2  #
        fmap_size = image_size // patch_size  #
        blocks = 2 ** (num_hierarchies - 1)  #

        seq_len = (fmap_size // blocks) ** 2  # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        # star_output = self.star(img)
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
        # return self.mlp_head(x)
        return x


import torch
import torch.nn as nn



class FusionModel111(nn.Module):  #mul 下采样
    def __init__(self, num_classes=5):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 4, 4, 3], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes
        )  # Ensure HTnet's output also matches the number of classes

        # 添加一个卷积层用于下采样 global_features
        self.downsample_conv = nn.Conv2d(1024, 1024, kernel_size=8, stride=8)  # 下采样至 [64, 1024, 1, 1]

        self.fc1 = nn.Linear(1024 * 1 * 1, 256)  # 修改全连接层的输入维度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)  # [64, 1024, 8, 8]
        parts_features = self.HTnet(parts_x)  # [64, 1024, 1, 1]

        # 使用卷积操作将 global_features 下采样到 [64, 1024, 1, 1]
        downsampled_global_features = self.downsample_conv(global_features)  # size: [64, 1024, 1, 1]

        # 确保尺寸一致后执行逐元素乘法
        if downsampled_global_features.size() != parts_features.size():
           parts_features = F.interpolate(parts_features, size=downsampled_global_features.shape[2:], mode='bilinear', align_corners=False)

        # 执行逐元素乘法，将 downsampled_global_features 和 parts_features 相乘，进行特征融合
        x = downsampled_global_features * parts_features

        # 展平特征图并传递到全连接层
        x = torch.flatten(x, start_dim=1)  # 将 (1024 * 1 * 1) 展平为一个向量，大小变为 [64, 1024]

        x = F.relu(self.fc1(x))  # 将维度从 (1024) 映射到 256
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # 将维度从 256 映射到 128
        x = self.dropout2(x)
        x = self.fc3(x)  # 最终的分类输出
        return x



class FusionModel(nn.Module):  # MUL 上采样
    def __init__(self, num_classes=3):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 4, 4, 3], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes)  # Ensure HTnet's output also matches the number of classes

        self.fc1 = nn.Linear(1024 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)
        parts_features = self.HTnet(parts_x)
        # 打印输出的尺寸（形状）
        #print("global_features size:", global_features.size())
        #print("parts_features size:", parts_features.size())

        if global_features.size() != parts_features.size():
            parts_features = F.interpolate(parts_features, size=global_features.shape[2:], mode='bilinear',
                                           align_corners=False)

        if global_features.size(1) != parts_features.size(1):
            parts_features = self.final_conv(parts_features)

        x = global_features * parts_features  # 执行逐元素乘法，将 global_features 和调整后的 parts_features 相乘。这样的操作可以进行特征融合，把两个特征图的信息合并起来。

        if x.size(2) == 1 or x.size(3) == 1:
            x = self.final_upsample(
                x)  # 如果融合后的特征图 x 在空间维度 (height 或 width) 上的尺寸为1，则通过 self.final_upsample 进行上采样，将 x 的空间尺寸调整为 (28, 28)。这可以避免因为尺寸太小导致的信息丢失。

        x = torch.flatten(x,
                          start_dim=1)  # 将特征图 x 展平（flatten），将空间维度 (height, width) 展平为一个长向量，使其可以传递到全连接层中。start_dim=1 表示从第一个维度开始展平，保留批次维度。
        x = F.relu(self.fc1(x))  # 先将展平后的特征传递到第一个全连接层，将其维度从 (1024 * 8 * 8) 映射到 256。
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # 将特征维度从 256 映射到 128。
        x = self.dropout2(x)
        x = self.fc3(x)  # 将特征维度从 128 映射到 num_classes，即最终的分类输出。
        return x

"""
class FusionModel(nn.Module):#cat
    def __init__(self, num_classes=5):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 1, 2, 2], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes)  # Ensure HTnet's output also matches the number of classes

        self.fc1 = nn.Linear(2048 * 8 * 8, 256)  # 拼接后的特征维度会增加到 1024 + 1024 = 2048
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)
        parts_features = self.HTnet(parts_x)

        # 如果全局特征和局部特征的大小不一致，调整局部特征大小
        if global_features.size() != parts_features.size():
            parts_features = F.interpolate(parts_features, size=global_features.shape[2:], mode='bilinear', align_corners=False)

        # 如果通道数量不一致，使用卷积调整局部特征通道数
        if global_features.size(1) != parts_features.size(1):
            parts_features = self.final_conv(parts_features)

        # 执行特征拼接，而不是逐元素相乘
        x = torch.cat((global_features, parts_features), dim=1)  # 在通道维度上拼接特征

        # 如果拼接后的特征图在空间维度上的尺寸为1，则进行上采样
        if x.size(2) == 1 or x.size(3) == 1:
            x = self.final_upsample(x)

        # 将特征图展平
        x = torch.flatten(x, start_dim=1)

        # 全连接层分类
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
"""

"""
class FusionModel(nn.Module):#max值
    def __init__(self, num_classes=5):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 4, 6, 3], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 8),
            num_classes=num_classes)  # Ensure HTnet's output also matches the number of classes

        self.fc1 = nn.Linear(1024 * 8 * 8, 256)  # 使用逐元素 max 操作，通道数量保持不变
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)
        parts_features = self.HTnet(parts_x)

        # 如果全局特征和局部特征的大小不一致，调整局部特征大小
        if global_features.size() != parts_features.size():
            parts_features = F.interpolate(parts_features, size=global_features.shape[2:], mode='bilinear', align_corners=False)

        # 如果通道数量不一致，使用卷积调整局部特征通道数
        if global_features.size(1) != parts_features.size(1):
            parts_features = self.final_conv(parts_features)

        # 执行逐元素取最大值操作
        x = torch.max(global_features, parts_features)  # 逐元素取最大值

        # 如果特征图在空间维度上的尺寸为1，则进行上采样
        if x.size(2) == 1 or x.size(3) == 1:
            x = self.final_upsample(x)

        # 将特征图展平
        x = torch.flatten(x, start_dim=1)

        # 全连接层分类
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
"""

"""
class FusionModel(nn.Module):#sum拼接
    def __init__(self, num_classes=5):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 3, 3, 3], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes)  # Ensure HTnet's output also matches the number of classes

        self.fc1 = nn.Linear(1024 * 8 * 8, 256)  # 使用逐元素相加操作，通道数量保持不变
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)
        parts_features = self.HTnet(parts_x)

        # 如果全局特征和局部特征的大小不一致，调整局部特征大小
        if global_features.size() != parts_features.size():
            parts_features = F.interpolate(parts_features, size=global_features.shape[2:], mode='bilinear', align_corners=False)

        # 如果通道数量不一致，使用卷积调整局部特征通道数
        if global_features.size(1) != parts_features.size(1):
            parts_features = self.final_conv(parts_features)

        # 执行逐元素相加操作
        x = global_features + parts_features  # 逐元素相加

        # 如果加法后的特征图在空间维度上的尺寸为1，则进行上采样
        if x.size(2) == 1 or x.size(3) == 1:
            x = self.final_upsample(x)

        # 将特征图展平
        x = torch.flatten(x, start_dim=1)

        # 全连接层分类
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
"""
"""
class FusionModel(nn.Module):#MAX
    def __init__(self, num_classes=5):
        super(FusionModel, self).__init__()
        self.StarNet = StarNet(base_dim=128, depths=[3, 4, 6, 3], mlp_ratio=4, drop_path_rate=0.0)
        self.HTnet = HTNet(
            image_size=224,
            patch_size=7,
            dim=256,
            heads=3,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=num_classes)  # Ensure HTnet's output also matches the number of classes

        self.fc1 = nn.Linear(1024 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Change output size to num_classes

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.final_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False)

    def forward(self, global_x, parts_x):
        global_features = self.StarNet(global_x)
        parts_features = self.HTnet(parts_x)

        # 如果全局特征和局部特征的大小不一致，进行上采样或调整大小
        if global_features.size() != parts_features.size():
            parts_features = F.interpolate(parts_features, size=global_features.shape[2:], mode='bilinear',
                                           align_corners=False)

        if global_features.size(1) != parts_features.size(1):
            parts_features = self.final_conv(parts_features)

        # 执行特征最大值融合，而不是乘积或拼接
        x = torch.max(global_features, parts_features)  # 在通道维度上取最大值特征

        # 如果求和后的特征图在空间维度上的尺寸为1，则进行上采样
        if x.size(2) == 1 or x.size(3) == 1:
            x = self.final_upsample(x)

        # 将特征图展平
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
"""





if __name__ == "__main__":
    num_classes = 3 # 假设有3个类别
    model = FusionModel(
        num_classes=3)
    input_tensor = torch.randn(4, 3, 224, 224)
    out = model(input_tensor)
    print(out.shape)
"""
#     model = HTNet(
#         image_size=224,
#         patch_size=7,
#         num_classes=3,
#         dim=256,
#         heads=3,
#         num_hierarchies=3,
#         block_repeats=(2, 2, 2),
#     )
#     input_tensor = torch.randn(4, 3, 224, 224)
#     output = model(input_tensor)
#     print("输出形状：", output.shape)
#
#
# # This function is to confuse three models
# class Fusionmodel(nn.Module):
#     def __init__(self):
#         #  extend from original
#         super(Fusionmodel, self).__init__()
#         self.fc1 = nn.Linear(15, 3)
#         self.bn1 = nn.BatchNorm1d(3)
#         self.d1 = nn.Dropout(p=0.CASMEII_5CLASS_train)
#         self.fc_2 = nn.Linear(6, 3)
#         self.relu = nn.ReLU()
#         # forward layers is to use these layers above
#
#     def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
#         fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
#         fuse_out = self.fc1(fuse_four_features)
#         fuse_out = self.relu(fuse_out)
#         fuse_out = self.d1(fuse_out)  # drop out
#         fuse_whole_four_parts = torch.cat(
#             (whole_feature, fuse_out), 0)
#         fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
#         fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
#         out = self.fc_2(fuse_whole_four_parts)
#         return out
"""
