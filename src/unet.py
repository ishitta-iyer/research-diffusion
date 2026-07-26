"""SmallUNet denoiser, consolidated from the notebooks/scripts.

SmallUNet(base_channels=16, emb_dim=64, num_levels=3) reproduces the original
notebook/script SmallUNet exactly: the layers are created in the same order with
the same shapes, so under a fixed torch.manual_seed the weights (and outputs)
are identical to the old inline definition. Module names differ (encs.0 vs
enc1), so old state_dicts need key remapping if they ever resurface.

num_levels is the depth knob: the number of resolution levels. num_levels=3 on
a 128x128 grid gives 128 -> 64 -> 32 with channels C -> 2C -> 4C. Each extra
level halves the resolution and doubles the channels once more.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        def ngroups(ch):
            for g in [8, 4, 2, 1]:
                if ch % g == 0:
                    return g
        self.norm1 = nn.GroupNorm(ngroups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(ngroups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    def __init__(self, base_channels=16, emb_dim=64, num_levels=3):
        super().__init__()
        assert num_levels >= 2, "need at least one down/up pair"
        C, E = base_channels, emb_dim
        self.num_levels = num_levels
        self.time_embed = nn.Sequential(SinusoidalEmbedding(E), nn.Linear(E, E * 2), nn.SiLU(), nn.Linear(E * 2, E))
        self.conv_in = nn.Conv2d(1, C, 3, padding=1)
        self.encs = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = C
        for _ in range(num_levels - 1):
            self.encs.append(ResBlock(ch, ch, E))
            self.downs.append(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1))
            ch *= 2
        self.mid = ResBlock(ch, ch, E)
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.ups.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
            self.decs.append(ResBlock(ch, ch // 2, E))
            ch //= 2
        self.conv_out = nn.Conv2d(C, 1, 3, padding=1)

    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.conv_in(x)
        skips = []
        for enc, down in zip(self.encs, self.downs):
            h = enc(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, emb)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1), emb)
        return self.conv_out(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


_LEGACY_KEY_MAP = {"enc1": "encs.0", "down1": "downs.0", "enc2": "encs.1", "down2": "downs.1",
                   "up2": "ups.0", "dec2": "decs.0", "up1": "ups.1", "dec1": "decs.1"}


def remap_legacy_state_dict(state_dict):
    """Map old inline-SmallUNet module names (enc1, down1, ...) to the consolidated
    3-level SmallUNet names (encs.0, downs.0, ...). Handles bare SmallUNet keys and
    EDMPrecond-prefixed keys (net.enc1....) alike."""
    out = {}
    for k, v in state_dict.items():
        parts = [_LEGACY_KEY_MAP.get(p, p) for p in k.split(".")]
        out[".".join(parts)] = v
    return out
