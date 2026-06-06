import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops

def init_weights(modules):
    pass
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, wn,  group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]

        elif scale == 3:
            modules += [wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        elif scale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)),nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class BasicConv2d(nn.Module):

    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True))

        self.LR = nn.ReLU(inplace=True)
        init_weights(self.modules)

    def forward(self, x):
        x = self.conv(x)
        x = self.LR(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class OptimizedLocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()

        self.norm1 = LayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(dim // mlp_ratio, dim, 1)

        self.norm2 = LayerNorm2d(dim)
        self.ffn1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.ffn_sg = SimpleGate()
        self.ffn2 = nn.Conv2d(dim, dim, 1)

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)) * 0.1, requires_grad=True)

    def forward(self, x):
        identity = x
        x = x + self.beta * self.pos_embed(x)
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.dwconv(out)
        out = self.sg(out)
        out = self.conv2(out)
        x = x + out
        out = self.norm2(x)
        out = self.ffn1(out)
        out = self.ffn_sg(out)
        out = self.ffn2(out)

        return x + self.gamma * out

class TinyLocalAgg(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.pw(self.dw(x))

def ddwconv(in_ch, out_ch, dilation=1, kernel_size=3):
    pad = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    if in_ch >= 32:
        groups = in_ch
    else:
        groups = min(in_ch, 8)
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size, 1, pad, dilation=dilation, groups=groups, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=True),
        nn.ReLU(inplace=True),
    )

class RepLKNet_DDWConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, kernel_size=3, sparse_ratio=0.25):
        super(RepLKNet_DDWConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.sparse_ratio = sparse_ratio

        if in_ch >= 32:
            self.groups = in_ch
        else:
            self.groups = min(in_ch, 8)

        if in_ch % self.groups != 0:
            raise ValueError(f"in_ch ({in_ch}) must be divisible by groups ({self.groups})!")

        self.pad = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2

        self.small_kernel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=((3 - 1) // 2) * dilation,
            dilation=dilation,
            groups=self.groups,
            bias=True
        )

        self.large_kernel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=self.pad,
            dilation=dilation,
            groups=self.groups,
            bias=True
        )
        self._init_large_kernel_sparse()

        self.point_conv = None
        if self.groups < in_ch:
            self.point_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=True
            )

        self.pointwise_conv = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)

        assert self.small_kernel_conv.in_channels == self.small_kernel_conv.out_channels, \
            "small_kernel_conv in/out channels mismatch!"
        assert self.large_kernel_conv.in_channels == self.large_kernel_conv.out_channels, \
            "large_kernel_conv in/out channels mismatch!"

    def _init_large_kernel_sparse(self):
        center_size = max(3, int(self.kernel_size * self.sparse_ratio))
        center_start = (self.kernel_size - center_size) // 2
        center_end = center_start + center_size

        for ch in range(self.in_ch):
            self.large_kernel_conv.weight.data[ch, 0, :, :][:center_start, :] = 0.0
            self.large_kernel_conv.weight.data[ch, 0, :, :][center_end:, :] = 0.0
            self.large_kernel_conv.weight.data[ch, 0, :, :][:, :center_start] = 0.0
            self.large_kernel_conv.weight.data[ch, 0, :, :][:, center_end:] = 0.0

    def _fuse_kernels(self):
        device = self.small_kernel_conv.weight.device
        dtype = self.small_kernel_conv.weight.dtype

        fused_large_kernel = self.large_kernel_conv.weight.clone()
        small_kernel_pad = (self.kernel_size - 3) // 2
        small_kernel_padded = F.pad(
            self.small_kernel_conv.weight,
            (small_kernel_pad, small_kernel_pad, small_kernel_pad, small_kernel_pad),
            mode='constant',
            value=0.0
        )
        fused_large_kernel += small_kernel_padded

        if self.point_conv is not None:
            point_kernel = self.point_conv.weight
            point_kernel_padded = torch.zeros_like(fused_large_kernel, device=device, dtype=dtype)
            center = self.kernel_size // 2
            point_kernel_padded[:, :, center:center+1, center:center+1] = point_kernel
            fused_large_kernel += point_kernel_padded

        fused_bias = self.large_kernel_conv.bias.clone()
        fused_bias += self.small_kernel_conv.bias
        if self.point_conv is not None:
            fused_bias += self.point_conv.bias

        return fused_large_kernel, fused_bias

    def train_forward(self, x):
        out_small = self.small_kernel_conv(x)
        out_large = self.large_kernel_conv(x)
        if self.point_conv is not None:
            out_point = self.point_conv(x)
            out_dw = out_small + out_large + out_point
        else:
            out_dw = out_small + out_large

        out_dw = self.act1(out_dw)
        out = self.pointwise_conv(out_dw)
        out = self.act2(out)
        return out

    def eval_forward(self, x):
        fused_kernel, fused_bias = self._fuse_kernels()
        out_dw = F.conv2d(
            input=x,
            weight=fused_kernel,
            bias=fused_bias,
            stride=1,
            padding=self.pad,
            dilation=self.dilation,
            groups=self.groups
        )

        out_dw = self.act1(out_dw)
        out = self.pointwise_conv(out_dw)
        out = self.act2(out)
        return out

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

class DEMSB(nn.Module):
    def __init__(self, wn, in_ch, out_ch, split_ratio=2):
        super().__init__()
        self.ms_channels = in_ch // split_ratio
        self.st_channels = in_ch - self.ms_channels
        mid = self.ms_channels
        self.conv_entry = nn.Sequential(
            nn.Conv2d(mid, mid, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        kernel_size = 5
        self.dconv1 = ops.RepLKNet_DDWConv(mid, mid // 2, dilation=1,kernel_size=kernel_size)
        self.dconv2 = ops.RepLKNet_DDWConv(mid // 2, mid // 2, dilation=3,kernel_size=kernel_size)
        self.dconv3 = ops.RepLKNet_DDWConv(mid, mid, dilation=5,kernel_size=kernel_size)
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(mid * 2, mid, 1),
            nn.ReLU(inplace=True)
        )
        self.stripe_path = DirectionalStripeAttention(dim=self.st_channels, k1=1,k2=17)
        self.fuse = wn(nn.Conv2d(in_ch, out_ch, 1))

    def forward(self, x):
        x_ms, x_st = torch.split(x, [self.ms_channels, self.st_channels], dim=1)
        x1 = self.conv_entry(x_ms)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat([dx1, dx2], 1))
        out_ms = self.ms_fuse(torch.cat([x1, dx3], 1))
        out_st = self.stripe_path(x_st)
        out = torch.cat([out_ms, out_st], dim=1)
        out = ops.channel_shuffle(out, groups=2)
        out = self.fuse(out)
        return out + x

class DirectionalStripeAttention(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.local_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.stripe_vertical = nn.Conv2d(
                dim, dim,
                kernel_size=(k1, k2),
                padding=(k1 // 2, k2 // 2),
                groups=dim
        )
        self.stripe_horizontal = nn.Conv2d(
                dim, dim,
                kernel_size=(k2, k1),
                padding=(k2 // 2, k1 // 2),
                groups=dim
        )
        self.channel_mix = nn.Conv2d(dim, dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
            attn = self.local_conv(x)
            v_attn = self.stripe_vertical(attn)
            h_attn = self.stripe_horizontal(attn)
            attn_mix = self.channel_mix(v_attn + h_attn)
            attn_mix = self.sigmoid(attn_mix)
            return x * attn_mix

class StripeAttentionBlock(nn.Module):
        def __init__(self, d_model, k1=1, k2=19):
            super().__init__()
            self.proj_in = nn.Conv2d(d_model, d_model, 1)
            self.activation = nn.GELU()
            self.stripe_attn = DirectionalStripeAttention(d_model, k1, k2)
            self.proj_out = nn.Conv2d(d_model, d_model, 1)
        def forward(self, x):
            shortcut = x
            x = self.proj_in(x)
            x = self.activation(x)
            x = self.stripe_attn(x)
            x = self.proj_out(x)
            return x + shortcut
class LightBSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.depth_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=out_channels, bias=False)

    def forward(self, x):
        return self.depth_conv(self.point_conv(x))

class OptimizedResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = LightBSConvU(channels, channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = LightBSConvU(channels, channels)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        res = res * self.ca(res)
        return x + res
