# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Canonical SongUNet and EDM preconditioning for the active experiments.

Derived from ``RectangleImages/training/networks.py`` at Baptista's
DiffusionModelDynamics commit ``2719b5d50601deb4f17fdf5306b6e26495ba19f4``, which in
turn derives from NVIDIA EDM. Persistence decorators and unrelated architectures are
omitted; the computational SongUNet/EDMPrecond path is unchanged.
"""

import numpy as np
import torch
from torch.nn.functional import silu


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal",
                 init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(
            weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=(1, 1), fused_resample=False, init_mode="kaiming_normal",
                 init_weight=1, init_bias=0):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel,
                           fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
            * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias
        ) if kernel and bias else None
        filt = torch.as_tensor(resample_filter, dtype=torch.float32)
        filt = filt.ger(filt).unsqueeze(0).unsqueeze(1) / filt.sum().square()
        self.register_buffer("resample_filter", filt if up or down else None)

    def forward(self, x):
        weight = self.weight.to(x.dtype) if self.weight is not None else None
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        filt = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        weight_pad = weight.shape[-1] // 2 if weight is not None else 0
        filter_pad = (filt.shape[-1] - 1) // 2 if filt is not None else 0

        if self.fused_resample and self.up and weight is not None:
            x = torch.nn.functional.conv_transpose2d(
                x, filt.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels, stride=2, padding=max(filter_pad - weight_pad, 0))
            x = torch.nn.functional.conv2d(x, weight,
                                           padding=max(weight_pad - filter_pad, 0))
        elif self.fused_resample and self.down and weight is not None:
            x = torch.nn.functional.conv2d(x, weight, padding=weight_pad + filter_pad)
            x = torch.nn.functional.conv2d(
                x, filt.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x, filt.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels, stride=2, padding=filter_pad)
            if self.down:
                x = torch.nn.functional.conv2d(
                    x, filt.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels, stride=2, padding=filter_pad)
            if weight is not None:
                x = torch.nn.functional.conv2d(x, weight, padding=weight_pad)
        if bias is not None:
            x = x.add_(bias.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return torch.nn.functional.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype), eps=self.eps)


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        weight = torch.einsum(
            "ncq,nck->nqk", q.to(torch.float32),
            (k / np.sqrt(k.shape[1])).to(torch.float32)
        ).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, weight)
        return weight

    @staticmethod
    def backward(ctx, grad_weight):
        q, k, weight = ctx.saved_tensors
        grad_logits = torch._softmax_backward_data(
            grad_output=grad_weight.to(torch.float32), output=weight.to(torch.float32),
            dim=2, input_dtype=torch.float32)
        grad_q = torch.einsum("nck,nqk->ncq", k.to(torch.float32), grad_logits).to(q.dtype)
        grad_q = grad_q / np.sqrt(k.shape[1])
        grad_k = torch.einsum("ncq,nqk->nck", q.to(torch.float32), grad_logits).to(k.dtype)
        grad_k = grad_k / np.sqrt(k.shape[1])
        return grad_q, grad_k


class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, up=False, down=False,
                 attention=False, num_heads=None, channels_per_head=64, dropout=0,
                 skip_scale=1, eps=1e-5, resample_filter=(1, 1), resample_proj=False,
                 adaptive_scale=True, init=None, init_zero=None, init_attn=None):
        super().__init__()
        init = {} if init is None else init
        init_zero = dict(init_weight=0) if init_zero is None else init_zero
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (0 if not attention else num_heads if num_heads is not None
                          else out_channels // channels_per_head)
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3,
                            up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels,
                             out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3,
                            **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel=1, **init_zero)

    def forward(self, x, emb):
        original = x
        x = self.conv0(silu(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout,
                                                   training=self.training))
        x = x.add_(self.skip(original) if self.skip is not None else original)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
            ).unbind(2)
            weight = AttentionOp.apply(q, k)
            attention = torch.einsum("nqk,nck->ncq", weight, v)
            x = self.proj(attention.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(0, self.num_channels // 2, dtype=torch.float32,
                             device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class SongUNet(torch.nn.Module):
    def __init__(self, img_resolution, in_channels, out_channels, label_dim=0,
                 augment_dim=0, model_channels=128, channel_mult=(1, 2, 2, 2),
                 channel_mult_emb=4, num_blocks=4, attn_resolutions=(16,), dropout=0.10,
                 label_dropout=0, embedding_type="positional", channel_mult_noise=1,
                 encoder_type="standard", decoder_type="standard",
                 resample_filter=(1, 1)):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout,
            skip_scale=np.sqrt(0.5), eps=1e-6, resample_filter=resample_filter,
            resample_proj=True, adaptive_scale=False, init=init,
            init_zero=init_zero, init_attn=init_attn)

        self.map_noise = (PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                          if embedding_type == "positional"
                          else FourierEmbedding(num_channels=noise_channels))
        self.map_label = (Linear(in_features=label_dim, out_features=noise_channels, **init)
                          if label_dim else None)
        self.map_augment = (Linear(in_features=augment_dim, out_features=noise_channels,
                                   bias=False, **init) if augment_dim else None)
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            resolution = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{resolution}x{resolution}_conv"] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{resolution}x{resolution}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == "skip":
                    self.enc[f"{resolution}x{resolution}_aux_down"] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True,
                        resample_filter=resample_filter)
                    self.enc[f"{resolution}x{resolution}_aux_skip"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == "residual":
                    self.enc[f"{resolution}x{resolution}_aux_residual"] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=3, down=True,
                        resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attention = resolution in attn_resolutions
                self.enc[f"{resolution}x{resolution}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attention, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            resolution = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{resolution}x{resolution}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f"{resolution}x{resolution}_in1"] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{resolution}x{resolution}_up"] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attention = idx == num_blocks and resolution in attn_resolutions
                self.dec[f"{resolution}x{resolution}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attention, **block_kwargs)
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{resolution}x{resolution}_aux_up"] = Conv2d(
                        in_channels=out_channels, out_channels=out_channels, kernel=0,
                        up=True, resample_filter=resample_filter)
                self.dec[f"{resolution}x{resolution}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6)
                self.dec[f"{resolution}x{resolution}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            labels = class_labels
            if self.training and self.label_dropout:
                labels = labels * (torch.rand([x.shape[0], 1], device=x.device)
                                   >= self.label_dropout).to(labels.dtype)
            emb = emb + self.map_label(labels * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


class EDMPrecond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, use_fp16=False,
                 sigma_min=0, sigma_max=float("inf"), sigma_data=0.5,
                 model_type="SongUNet", **model_kwargs):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        if model_type != "SongUNet":
            raise ValueError(f"unsupported model_type {model_type!r}; canonical module contains SongUNet")
        self.model = SongUNet(img_resolution=img_resolution, in_channels=img_channels,
                              out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (None if self.label_dim == 0 else
                        torch.zeros([1, self.label_dim], device=x.device)
                        if class_labels is None else
                        class_labels.to(torch.float32).reshape(-1, self.label_dim))
        dtype = (torch.float16 if self.use_fp16 and not force_fp32
                 and x.device.type == "cuda" else torch.float32)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        model_out = self.model((c_in * x).to(dtype), c_noise.flatten(),
                               class_labels=class_labels, **model_kwargs)
        assert model_out.dtype == dtype
        return c_skip * x + c_out * model_out.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def songunet_config(img_resolution=128, model_channels=16):
    """Baptista-compatible factory options with the active 64->128 adaptation."""
    channel_mult = [2, 2, 2]
    return dict(
        img_resolution=img_resolution,
        img_channels=1,
        label_dim=0,
        use_fp16=False,
        model_type="SongUNet",
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        channel_mult_noise=1,
        resample_filter=[1, 1],
        model_channels=model_channels,
        channel_mult=channel_mult,
        dropout=0.0,
        attn_resolutions=[img_resolution >> (len(channel_mult) - 1)],
        sigma_data=0.5,
    )


def build_songunet(img_resolution=128, model_channels=16, device=None, **overrides):
    config = songunet_config(img_resolution=img_resolution, model_channels=model_channels)
    config.update(overrides)
    net = EDMPrecond(**config)
    return net if device is None else net.to(device)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
