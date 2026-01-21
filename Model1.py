from functools import partial
import torch
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder


class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs["out_channels"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, **kwargs):
        super(DWConv, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["in_channels"],
                      kernel_size=kwargs["kernel_size"],
                      padding=kwargs["kernel_size"] // 2,
                      groups=kwargs["in_channels"],
                      bias=False),
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["out_channels"],
                      kernel_size=1,
                      bias=False)
        )

    def forward(self, x):
        return self.block(x)


class GraphLearningModel(nn.Module):
    def __init__(self,
                 input_dim: int = 49,
                 forward_dim: int = 128,
                 num_heads: int = 8,
                 head_dim: int = 16,
                 num_layers: int = 6,
                 attn_drop_rate: float = 0.1,
                 proj_drop_rate: float = 0.5,
                 in_channels: int = 30,
                 stride: int = 1,
                 kernel_size: int = 3):
        super(GraphLearningModel, self).__init__()

        # Depth wise convolution for the input
        self.DWConv = nn.Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=in_channels,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=in_channels),
            nn.Flatten(start_dim=2)
        )

        self.eyebrow_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])
        self.mouth_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])

        self.eyebrow_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(490, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.mouth_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(980, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Before: Shape of x: (batch_size, 30, 7, 7)
        # After: Shape of x: (batch_size, 30, 49)
        x = self.DWConv(x)

        # Extract the specific part of vectors
        eyebrow_vector = x[:, :10]
        mouth_vector = x[:, 10:]

        # Shape of eyebrow_vector: (batch_size, 490)
        # Shape of mouth_vector: (batch_size, 980)
        eyebrow_vector = self.eyebrow_encoder(eyebrow_vector)
        mouth_vector = self.mouth_encoder(mouth_vector)

        # Shape of eyebrow_vector: (batch_size, 160)
        # Shape of mouth_vector: (batch_size, 160)
        eyebrow_vector = self.eyebrow_layer(eyebrow_vector)
        mouth_vector = self.mouth_layer(mouth_vector)

        return eyebrow_vector, mouth_vector


# 深度可分离卷积块
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


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
        out = self.depthwise_conv(out)  ## 深度可分离卷积层
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
        #nn.Conv2d(dim, dim_out, 3, padding=1),
        nn.Conv2d(dim, dim, 3, 1, 1),
        nn.Conv2d(dim, dim_out, 1),#深度可分离卷积
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
        #self.se = CBAMLayer(256)
        #self.Gr = GraphLearningModel()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        #x = self.se(x) * x  # shape(32.256.4.4)
        # x1 = self.Gr(x)

        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
        return self.mlp_head(x)  # 输出是（[4, 1024, 8, 8]）


# This function is to confuse three models
class Fusionmodel(nn.Module):
    def __init__(self):
        #  extend from original
        super(Fusionmodel, self).__init__()
        self.fc1 = nn.Linear(15, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()
        # forward layers is to use these layers above

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
        fuse_out = self.fc1(fuse_four_features)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)  # drop out
        fuse_whole_four_parts = torch.cat(
            (whole_feature, fuse_out), 0)
        fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
        fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
        out = self.fc_2(fuse_whole_four_parts)
        return out


if __name__ == "__main__":
    model = HTNet(image_size=224, patch_size=7, dim=256, heads=3, num_hierarchies=3, block_repeats=(2, 2, 10),
                  num_classes=3)
    input_tensor = torch.randn(4, 3, 224, 224)
    output_tensor = model(input_tensor)
    print(output_tensor)
