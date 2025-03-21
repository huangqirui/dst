import torch
import torch.nn as nn
from neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math


class SpikingTokenizer(nn.Module):
    def __init__(
            self, img_size=128, patch_size=4, in_channels=2, embed_dims=256, pool_state=[True, True, True, True],
            backend="cupy", lif_tau=1.5):
        super().__init__()
        self.image_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pool_state = pool_state
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.proj1_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj2_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        T, B, C, H, W = x.shape
        ratio = 1

        x = self.proj_conv(x.flatten(0, 1))
        if self.pool_state[0]:
            x = self.maxpool(x)
            ratio *= 2
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        if self.pool_state[1]:
            x = self.maxpool1(x)
            ratio *= 2
        x = self.proj1_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()


        x = self.proj2_conv(x)
        if self.pool_state[2]:
            x = self.maxpool2(x)
            ratio *= 2
        x = self.proj2_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        if self.pool_state[3]:
            x = self.maxpool3(x)
            ratio *= 2
        x = self.proj3_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.rpe_conv(x).reshape(T, B, -1, (H // ratio) * (H // ratio)).contiguous()

        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8,
            pa=0.0, pr=0.0, masking_type_attn='random', masking_type_attn_lif='random',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_attn)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_attn)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_attn)

        self.v_dw_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.v_dw_bn = nn.BatchNorm2d(dim)

        self.attn_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_attn_lif)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_attn)

    def forward(self, x):
        T, B, C, N = x.shape
        x = self.proj_bn(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()
        x_for_qkv = self.proj_lif(x).flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()

        H = int(math.sqrt(N))
        v_dw = v_conv_out.mean(0).reshape(B, C, H, H)
        v_dw = self.v_dw_conv(v_dw)
        v_dw = self.v_dw_bn(v_dw).reshape(B, C, N)

        x = self.attn_lif(x, v_dw)
        x = x.flatten(0, 1)

        x = self.proj_conv(x).reshape(T, B, C, N)

        return x


class SpikingMLP(nn.Module):
    def __init__(
            self,
            in_features, hidden_features=None, out_features=None,
            pa=0.0, pr=0.0, masking_type_mlp='random',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_mlp)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=lif_tau, detach_reset=True, backend=backend, pa=pa, pr=pr, masking_type=masking_type_mlp)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N = x.shape

        x = self.fc2_bn(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x)

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1)).reshape(T, B, C, N)

        return x


class SpikingTransformerBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4.,
            pa=0.0, pr=0.0, masking_type_attn='random', masking_type_mlp='random', masking_type_attn_lif='random',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(
            dim, num_heads=num_heads,
            pa=pa, pr=pr, masking_type_attn=masking_type_attn, masking_type_attn_lif=masking_type_attn_lif,
            backend=backend, lif_tau=lif_tau,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(
            in_features=dim, hidden_features=mlp_hidden_dim,
            pa=pa, pr=pr, masking_type_mlp=masking_type_mlp,
            backend=backend, lif_tau=lif_tau,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class DST(nn.Module):
    def __init__(
            self, img_size=128, patch_size=16, in_channels=2, T=4, mlp_ratios=4, pool_state=[True, True, True, True],
            embed_dims=256, encoder_depths=6, encoder_num_heads=4,
            num_classes=11, TET=False,
            pa=0.0, pr=0.0, masking_type_attn='random', masking_type_mlp='random', masking_type_attn_lif='random',
            backend="cupy", lif_tau=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_depths = encoder_depths
        self.T = T
        self.TET = TET

        # ============ encoder ============
        self.patch_embed = SpikingTokenizer(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims,
            pool_state=pool_state, backend=backend, lif_tau=lif_tau,
        )
        self.num_patches = self.patch_embed.num_patches

        self.snn_blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=embed_dims, num_heads=encoder_num_heads, mlp_ratio=mlp_ratios,
                pa=pa, pr=pr,
                masking_type_attn=masking_type_attn, masking_type_mlp=masking_type_mlp, masking_type_attn_lif=masking_type_attn_lif,
                backend=backend, lif_tau=lif_tau,
            )
            for j in range(encoder_depths)
        ])
        self.bn = nn.BatchNorm1d(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) < 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.permute(1, 0, 2, 3, 4)

        x = self.patch_embed(x)

        for blk in self.snn_blocks:
            x = blk(x)

        T, B, D, N = x.shape
        x = self.bn(x.flatten(0, 1)).reshape(T, B, D, N)
        if not self.TET:
            x = self.head(x.mean(-1).mean(0))
        else:
            x = self.head(x.mean(-1))
        return x


@register_model
def dst_dvs(pretrained=False, **kwargs):
    model = DST(
        img_size=128, patch_size=16, in_channels=2, mlp_ratios=4, T=16, pool_state=[True, True, True, True],
        embed_dims=256, encoder_depths=2, encoder_num_heads=16,
        num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        pr=kwargs['pr'], pa=kwargs['pa'],
        masking_type_attn=kwargs['masking_type_attn'], masking_type_mlp=kwargs['masking_type_mlp'], masking_type_attn_lif=kwargs['masking_type_attn_lif'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model


@register_model
def dst_cifar10(pretrained=False, **kwargs):
    model = DST(
        img_size=32, patch_size=4, in_channels=3, mlp_ratios=4, T=4, pool_state=[False, False, True, True],
        embed_dims=384, encoder_depths=4, encoder_num_heads=12,
        num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        pr=kwargs['pr'], pa=kwargs['pa'],
        masking_type_attn=kwargs['masking_type_attn'], masking_type_mlp=kwargs['masking_type_mlp'], masking_type_attn_lif=kwargs['masking_type_attn_lif'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model


@register_model
def dst_cifar100(pretrained=False, **kwargs):
    model = DST(
        img_size=32, patch_size=4, in_channels=3, mlp_ratios=4, T=4, pool_state=[False, False, True, True],
        embed_dims=384, encoder_depths=4, encoder_num_heads=8,
        num_classes=kwargs['num_classes'], TET=kwargs['TET'],
        pr=kwargs['pr'], pa=kwargs['pa'],
        masking_type_attn=kwargs['masking_type_attn'], masking_type_mlp=kwargs['masking_type_mlp'], masking_type_attn_lif=kwargs['masking_type_attn_lif'],
        backend=kwargs['backend'], lif_tau=kwargs['lif_tau'],
    )
    return model

