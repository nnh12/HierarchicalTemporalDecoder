import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_encoding(timesteps, dim, base=10000.0):
    """Sinusoidal positional encoding for scalar timesteps."""
    half = dim // 2
    freqs = torch.exp(-math.log(base) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: learns per-channel scale & shift."""

    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1)
        beta = beta.view(beta.size(0), -1, 1, 1)
        return x * (1 + gamma) + beta


class BandDecoder(nn.Module):
    """Single temporal-band FiLM-conditioned convolutional decoder.

    Produces an unbounded single-channel feature map (no final activation)
    so that multiple bands can be summed before a shared sigmoid.
    """

    def __init__(self, cond_dim, base_ch=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, base_ch * 8 * 14),
            nn.SiLU(),
        )

        ch = [base_ch, base_ch * 2, base_ch, base_ch // 2, base_ch // 4]

        self.up1 = nn.ConvTranspose2d(ch[0], ch[1], 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ch[1])
        self.film1 = FiLMBlock(cond_dim, ch[1])

        self.up2 = nn.ConvTranspose2d(ch[1], ch[2], 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch[2])
        self.film2 = FiLMBlock(cond_dim, ch[2])

        self.up3 = nn.ConvTranspose2d(ch[2], ch[3], 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.film3 = FiLMBlock(cond_dim, ch[3])

        self.up4 = nn.ConvTranspose2d(ch[3], ch[4], 4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ch[4])
        self.film4 = FiLMBlock(cond_dim, ch[4])

        self.head = nn.Conv2d(ch[4], 1, 3, padding=1)

    def forward(self, cond):
        x = self.fc(cond)
        x = x.view(x.size(0), -1, 8, 14)
        x = F.silu(self.film1(self.bn1(self.up1(x)), cond))
        x = F.silu(self.film2(self.bn2(self.up2(x)), cond))
        x = F.silu(self.film3(self.bn3(self.up3(x)), cond))
        x = F.silu(self.film4(self.bn4(self.up4(x)), cond))
        return self.head(x)


class HierarchicalTemporalDecoder(nn.Module):
    """Hierarchical Multi-Rate FiLM Decoder.

    Decomposes frame generation into three temporal frequency bands,
    each with its own FiLM-conditioned decoder:
      - Static  : label-only conditioning (time-invariant base field)
      - Trend   : label + low-frequency sinusoidal time encoding
      - Detail  : label + high-frequency sinusoidal time encoding

    The final frame is  sigmoid(static + trend + detail), resized to
    (output_h, output_w).
    """

    def __init__(self, label_dim=4, time_dim=128, cond_dim=256,
                 output_h=300, output_w=530, band_ch=32,
                 trend_base=100000.0, detail_base=100.0):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w
        self.time_dim = time_dim
        self.trend_base = trend_base
        self.detail_base = detail_base

        self.static_label_mlp = nn.Sequential(
            nn.Linear(label_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.static_decoder = BandDecoder(cond_dim, band_ch)

        self.trend_label_mlp = nn.Sequential(
            nn.Linear(label_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.trend_time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.trend_decoder = BandDecoder(cond_dim, band_ch)

        self.detail_label_mlp = nn.Sequential(
            nn.Linear(label_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.detail_time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.detail_decoder = BandDecoder(cond_dim, band_ch)

    def forward(self, label, timestep, return_bands=False):
        """
        label:        (B, label_dim)
        timestep:     (B,)  — scalar frame index
        return_bands: if True, also return individual band outputs
        returns:      (B, 1, output_h, output_w)
                      or (combined, static, trend, detail) when return_bands=True
        """
        static_cond = self.static_label_mlp(label)
        static_out = self.static_decoder(static_cond)

        t_trend = sinusoidal_encoding(timestep, self.time_dim, base=self.trend_base)
        trend_cond = self.trend_label_mlp(label) + self.trend_time_mlp(t_trend)
        trend_out = self.trend_decoder(trend_cond)

        t_detail = sinusoidal_encoding(timestep, self.time_dim, base=self.detail_base)
        detail_cond = self.detail_label_mlp(label) + self.detail_time_mlp(t_detail)
        detail_out = self.detail_decoder(detail_cond)

        combined = torch.sigmoid(static_out + trend_out + detail_out)
        combined = F.interpolate(combined, size=(self.output_h, self.output_w),
                                 mode="bilinear", align_corners=False)

        if return_bands:
            def _resize(t):
                return F.interpolate(t, size=(self.output_h, self.output_w),
                                     mode="bilinear", align_corners=False)
            return combined, _resize(static_out), _resize(trend_out), _resize(detail_out)
        return combined
