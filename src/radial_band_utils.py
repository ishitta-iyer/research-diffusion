import torch


def generate_matern_laplace(num_samples, grid_size, sigma_sq, length_scale, s, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    freq = torch.fft.fftfreq(grid_size) * grid_size
    freq_x, freq_y = torch.meshgrid(freq, freq, indexing="ij")
    laplacian = freq_x**2 + freq_y**2

    spectral_density = sigma_sq * (laplacian + (length_scale**2)) ** (-s)
    spectral_density[0, 0] = 0
    spectral_density = spectral_density.unsqueeze(0)

    noise_real = torch.randn(num_samples, grid_size, grid_size)
    noise_imag = torch.randn(num_samples, grid_size, grid_size)
    noise = noise_real + 1j * noise_imag

    spectral_sample = torch.sqrt(spectral_density) * noise
    sample = torch.fft.ifft2(spectral_sample, norm="forward").real
    return sample


def make_knrm_grid(grid_size: int, device=None):
    freq = torch.fft.fftfreq(grid_size, device=device) * grid_size
    kx, ky = torch.meshgrid(freq, freq, indexing="ij")
    return torch.sqrt(kx**2 + ky**2)


def make_radial_band_mask(grid_size: int, k_lo: float, k_hi: float, device=None, dtype=torch.float32):
    knrm = make_knrm_grid(grid_size, device=device)
    return ((knrm >= k_lo) & (knrm < k_hi)).to(dtype)


def radial_bandpass(x: torch.Tensor, mask: torch.Tensor, norm="forward"):
    X = torch.fft.fft2(x, dim=(-2, -1), norm=norm)
    Xf = X * mask
    return torch.fft.ifft2(Xf, dim=(-2, -1), norm=norm).real


@torch.no_grad()
def band_power_fraction(x: torch.Tensor, mask: torch.Tensor, norm=None, eps=1e-12):
    X = torch.fft.fft2(x, dim=(-2, -1), norm=norm)
    P = (X.real**2 + X.imag**2)

    P_in = (P * mask).sum(dim=(-2, -1))
    P_tot = P.sum(dim=(-2, -1)).clamp_min(eps)
    return P_in / P_tot


def generate_multiband_dataset_postmask(
    num_samples: int,
    grid_size: int,
    components: list[dict],
    weights: list[float] | None = None,
    seed: int | None = None,
    normalize: bool = True,
    device=None,
):
    device = device or "cpu"
    if weights is None:
        weights = [1.0] * len(components)
    if len(weights) != len(components):
        raise ValueError("weights and components must have same length")

    xs, ffts, used_bands = {}, {}, {}

    for j, comp in enumerate(components):
        name = comp.get("name", f"comp{j}")
        comp_seed = None if seed is None else seed + 10_000 * j

        xj = generate_matern_laplace(
            num_samples=num_samples,
            grid_size=grid_size,
            sigma_sq=comp.get("sigma_sq", 1.0),
            length_scale=comp.get("kappa", comp.get("length_scale", 1.0)),
            s=comp.get("s", 2.0),
            seed=comp_seed,
        ).to(device)

        band = comp.get("band", None)
        if band is not None:
            k_lo, k_hi = band
            mask = make_radial_band_mask(grid_size, k_lo, k_hi, device=device, dtype=xj.dtype)
            xj = radial_bandpass(xj, mask)
            used_bands[name] = band

        xs[name] = xj
        ffts[name] = torch.fft.fft2(xj, dim=(-2, -1))

    x = sum(w * xs[components[i].get("name", f"comp{i}")] for i, w in enumerate(weights))
    X = torch.fft.fft2(x, dim=(-2, -1))

    normalization = None
    if normalize:
        mean = x.mean()
        std = x.std().clamp_min(1e-8)
        x = (x - mean) / std
        normalization = {"mean": mean.item(), "std": std.item()}

    return {
        "combined": x,
        "components": xs,
        "component_ffts": ffts,
        "combined_fft": X,
        "bands": used_bands,
        "normalization": normalization,
        "grid_size": grid_size,
    }


def make_radial_k_grid(N, device=None):
    k = torch.fft.fftfreq(N, d=1.0, device=device) * N
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    return torch.sqrt(kx**2 + ky**2)


def add_fourier_bias_to_result(
    result: dict,
    kmin: float = None,
    kmax: float = None,
    k0: float = None,
    width: float = None,
    strength: float = 0.10,
    seed: int = 123,
    overwrite_combined: bool = True,
) -> dict:
    out = dict(result)

    x = out["combined"]
    _, N, _ = x.shape
    device = x.device

    k = torch.fft.fftfreq(N, d=1.0, device=device) * N
    kx, ky = torch.meshgrid(k, k, indexing="ij")
    kr = torch.sqrt(kx**2 + ky**2)

    if k0 is not None and width is not None:
        mask = ((kr >= k0 - width / 2) & (kr <= k0 + width / 2)).float()
    elif kmin is not None and kmax is not None:
        mask = ((kr >= kmin) & (kr <= kmax)).float()
    else:
        raise ValueError("Provide (kmin, kmax) or (k0, width)")

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    w = torch.randn(N, N, generator=g, device=device)
    W = torch.fft.fft2(w, norm="forward")
    W_f = W * mask
    p = torch.fft.ifft2(W_f, norm="forward").real
    p = p - p.mean()
    p = p / p.std().clamp_min(1e-8)

    x_biased = x + strength * p.unsqueeze(0)

    out["combined_clean"] = x
    out["combined_biased"] = x_biased
    out["combined_biased_fft"] = torch.fft.fft2(x_biased, dim=(-2, -1))
    out["bias_pattern"] = p
    out["bias_mask"] = mask
    out["bias_meta"] = {
        "kmin": kmin,
        "kmax": kmax,
        "k0": k0,
        "width": width,
        "strength": strength,
        "seed": seed,
    }

    if overwrite_combined:
        out["combined"] = x_biased

    return out
