"""Focused parity checks for the canonical active SongUNet source.

Run with ``python3 src/tests/test_songunet_parity.py``.
"""

import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from multiband_data_utils import (
    SONGUNET_MULTIBAND_COMPONENTS,
    SONGUNET_MULTIBAND_WEIGHTS,
    analytic_multiband_spectrum,
    anchor_regenerated_fields,
    baptista_rectangle_pair,
    first_pair_distance,
    generate_rescaled_songunet_matern_pool,
    inverse_spectrum_weights,
)
from songunet import build_songunet, count_parameters, songunet_config
from songunet_edm import (
    EDMLoss,
    edm_sampler,
    edm_sigma_schedule,
    edm_training_loss,
    ema_beta,
    make_batch1_dataloader,
    ramped_learning_rate,
    tikhonov_penalty_covariance,
    tikhonov_penalty_isotropic,
    update_ema_,
)


ACTIVE_NOTEBOOK = REPO_ROOT / "notebooks" / "multiscale" / "songunet_covariance_tikhonov.ipynb"
ACTIVE_ARTIFACT = (REPO_ROOT / "results" / "data" / "songunet_cov_tikhonov"
                   / "arm_gate_c0_result.pt")


def _inline_namespace(*cell_indices):
    notebook = json.loads(ACTIVE_NOTEBOOK.read_text())
    namespace = {"torch": torch, "np": __import__("numpy"),
                 "DEVICE": torch.device("cpu")}
    for index in cell_indices:
        source = notebook["cells"][index]["source"]
        exec("".join(source) if isinstance(source, list) else source, namespace)
    return namespace


def test_architecture_matches_frozen_inline_notebook():
    inline = _inline_namespace(4, 5)
    config = songunet_config(img_resolution=128, model_channels=4)
    torch.manual_seed(123)
    reference = inline["EDMPrecond"](**config).eval()
    torch.manual_seed(123)
    canonical = build_songunet(img_resolution=128, model_channels=4).eval()

    reference_parameters = list(reference.named_parameters())
    canonical_parameters = list(canonical.named_parameters())
    assert [name for name, _ in reference_parameters] == [name for name, _ in canonical_parameters]
    assert [tuple(value.shape) for _, value in reference_parameters] == [
        tuple(value.shape) for _, value in canonical_parameters]
    for (name, expected), (_, actual) in zip(reference_parameters, canonical_parameters):
        assert torch.equal(expected, actual), f"initialization differs at {name}"

    generator = torch.Generator().manual_seed(77)
    x = torch.randn(1, 1, 128, 128, generator=generator)
    sigma = torch.tensor([0.3])
    with torch.no_grad():
        expected = reference(x, sigma)
        actual = canonical(x, sigma)
    assert torch.equal(expected, actual)


def test_active_configuration_and_parameter_count():
    config = songunet_config(img_resolution=128, model_channels=16)
    assert config["channel_mult"] == [2, 2, 2]
    assert config["attn_resolutions"] == [32]
    assert config["embedding_type"] == "positional"
    assert config["encoder_type"] == config["decoder_type"] == "standard"
    assert config["channel_mult_noise"] == 1
    assert config["resample_filter"] == [1, 1]
    assert config["dropout"] == 0.0
    assert config["sigma_data"] == 0.5
    torch.manual_seed(0)
    assert count_parameters(build_songunet()) == 880_097


def test_preconditioning_coefficients():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = None
            self.noise = None

        def forward(self, x, noise, class_labels=None, **_):
            self.input = x.clone()
            self.noise = noise.clone()
            return torch.ones_like(x)

    net = build_songunet(img_resolution=8, model_channels=4)
    recorder = RecordingModel()
    net.model = recorder
    x = torch.randn(2, 1, 8, 8)
    sigma = torch.tensor([0.25, 2.0])
    output = net(x, sigma)
    shaped_sigma = sigma[:, None, None, None]
    c_skip = 0.5 ** 2 / (shaped_sigma ** 2 + 0.5 ** 2)
    c_out = shaped_sigma * 0.5 / (shaped_sigma ** 2 + 0.5 ** 2).sqrt()
    c_in = 1 / (shaped_sigma ** 2 + 0.5 ** 2).sqrt()
    assert torch.equal(output, c_skip * x + c_out)
    assert torch.equal(recorder.input, c_in * x)
    assert torch.equal(recorder.noise, sigma.log() / 4)


def test_edm_loss_fixed_rng_matches_inline():
    inline = _inline_namespace(9)

    class IdentityDenoiser(torch.nn.Module):
        def forward(self, x, sigma, labels=None, augment_labels=None):
            return x / (1 + sigma)

    images = torch.randn(3, 1, 8, 8)
    reference_loss = inline["EDMLoss"]()
    canonical_loss = EDMLoss()
    torch.manual_seed(44)
    expected = reference_loss(IdentityDenoiser(), images)
    torch.manual_seed(44)
    actual = canonical_loss(IdentityDenoiser(), images)
    assert torch.equal(expected, actual)


def test_sampler_schedule_and_deterministic_output_match_inline():
    inline = _inline_namespace(11)

    class ToyDenoiser(torch.nn.Module):
        sigma_min = 0.0
        sigma_max = float("inf")

        @staticmethod
        def round_sigma(sigma):
            return torch.as_tensor(sigma)

        def forward(self, x, sigma, class_labels=None):
            return x / (1 + sigma)

    net = ToyDenoiser()
    expected_schedule = edm_sigma_schedule(net, num_steps=40, sigma_min=0.002,
                                           sigma_max=80, rho=7, device="cpu")
    index = torch.arange(40, dtype=torch.float64)
    direct = (80 ** (1 / 7) + index / 39 * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
    direct = torch.cat([direct, torch.zeros_like(direct[:1])])
    assert torch.equal(expected_schedule, direct)

    latents = torch.randn(2, 1, 4, 4)
    zeros = lambda value: torch.zeros_like(value)
    expected = inline["edm_sampler"](net, latents, randn_like=zeros, num_steps=8)
    actual = edm_sampler(net, latents, randn_like=zeros, num_steps=8,
                         sampler_dtype=torch.float64)
    assert torch.equal(expected, actual)


def test_training_order_lr_ramp_and_ema():
    data = torch.arange(5, dtype=torch.float32)[:, None]
    generator_a = torch.Generator().manual_seed(9)
    generator_b = torch.Generator().manual_seed(9)
    actual = [batch[0].item() for (batch,) in make_batch1_dataloader(
        data, generator=generator_a)]
    direct_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=1, shuffle=True,
        generator=generator_b)
    expected = [batch[0].item() for (batch,) in direct_loader]
    assert actual == expected
    assert ramped_learning_rate(1e-3, 1, 10_000) == 1e-10
    assert ramped_learning_rate(1e-3, 10_000_000, 10_000) == 1e-3
    expected_beta = 0.5 ** (1 / (100 * 0.05))
    assert ema_beta(1, 100, 500, 0.05) == expected_beta

    net = torch.nn.Linear(2, 1, bias=False)
    ema = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(2)
        ema.weight.fill_(10)
    update_ema_(ema, net, 0.75)
    assert torch.equal(ema.weight, torch.full_like(ema.weight, 8.0))


def test_default_off_training_loss_is_bit_identical():
    class IdentityDenoiser(torch.nn.Module):
        def forward(self, x, sigma, labels=None, augment_labels=None):
            return x / (1 + sigma)

    images = torch.randn(3, 1, 8, 8)
    config = EDMLoss()
    torch.manual_seed(91)
    expected = config(IdentityDenoiser(), images).sum() / images.shape[0]
    torch.manual_seed(91)
    actual = edm_training_loss(IdentityDenoiser(), images, loss_config=config)
    assert torch.equal(expected, actual)


def test_rescaling_and_saved_field_anchor():
    rectangles = baptista_rectangle_pair()
    squared_distance = first_pair_distance(rectangles).square().item()
    assert squared_distance == 340.0
    assert abs(first_pair_distance(rectangles).item() - math.sqrt(340.0)) < 1e-6

    pool = generate_rescaled_songunet_matern_pool()
    rescaled = pool["combined_rescaled"]
    assert abs(first_pair_distance(rescaled).item() - math.sqrt(340.0)) < 1e-3
    assert abs(pool["scale"] - 0.09920109830812254) < 5e-7

    if ACTIVE_ARTIFACT.exists():
        saved = torch.load(ACTIVE_ARTIFACT, map_location="cpu", weights_only=False)
        delta = (rescaled[:2] - saved["data"]).abs().max().item()
        assert delta <= 1e-7, delta
        anchored = anchor_regenerated_fields(rescaled[:2], saved["data"])
        assert torch.equal(anchored, saved["data"])


def test_analytic_spectrum_and_inverse_weights():
    pool = generate_rescaled_songunet_matern_pool()
    spectrum = analytic_multiband_spectrum(
        128, SONGUNET_MULTIBAND_COMPONENTS, SONGUNET_MULTIBAND_WEIGHTS,
        normalization_std=pool["normalization"]["std"], scale=pool["scale"])
    reflected = torch.roll(torch.flip(spectrum, (-2, -1)), (1, 1), (-2, -1))
    assert torch.equal(spectrum, reflected)
    inverse, active = inverse_spectrum_weights(spectrum)
    assert torch.isfinite(inverse).all()
    assert torch.equal(inverse[~active], torch.zeros_like(inverse[~active]))
    assert abs(inverse.sum().item() - 128 ** 4) < 1e-5 * 128 ** 4

    if ACTIVE_ARTIFACT.exists():
        saved = torch.load(ACTIVE_ARTIFACT, map_location="cpu", weights_only=False)
        assert torch.equal(spectrum, saved["lam_pop"])


def test_tikhonov_penalty_reduction_and_stationary_points():
    torch.manual_seed(5)
    batch, side = 2, 4
    mode_count = side * side
    x0 = torch.randn(batch, 1, side, side)
    noisy = torch.randn_like(x0)
    sigma = torch.tensor([0.7, 1.3])
    c = 0.2

    trial_iso = ((sigma[:, None, None, None] ** 2 * x0 + c * noisy)
                 / (sigma[:, None, None, None] ** 2 + c)).requires_grad_()
    iso_objective = ((trial_iso - x0) ** 2).sum()
    iso_objective = iso_objective + tikhonov_penalty_isotropic(
        trial_iso, noisy, sigma, c).sum()
    iso_gradient = torch.autograd.grad(iso_objective, trial_iso)[0]
    assert iso_gradient.abs().max().item() < 2e-6

    real_fields = torch.randn(32, side, side)
    spectrum = torch.fft.fft2(real_fields, norm="forward").abs().square().mean(0) + 0.1
    inverse = 1 / spectrum
    x0_hat = torch.fft.fft2(x0.squeeze(1), norm="forward")
    noisy_hat = torch.fft.fft2(noisy.squeeze(1), norm="forward")
    sigma_squared = sigma[:, None, None] ** 2
    expected_hat = ((mode_count * sigma_squared * x0_hat + c * inverse * noisy_hat)
                    / (mode_count * sigma_squared + c * inverse))
    trial_cov = torch.fft.ifft2(expected_hat, norm="forward").real.unsqueeze(1).requires_grad_()
    cov_objective = ((trial_cov - x0) ** 2).sum()
    cov_objective = cov_objective + tikhonov_penalty_covariance(
        trial_cov, noisy, sigma, c, inverse).sum()
    cov_gradient = torch.autograd.grad(cov_objective, trial_cov)[0]
    assert cov_gradient.abs().max().item() < 2e-5

    denoised = torch.randn_like(x0)
    iso = tikhonov_penalty_isotropic(denoised, noisy, sigma, c)
    cov = tikhonov_penalty_covariance(
        denoised, noisy, sigma, c, torch.full((side, side), mode_count))
    assert torch.allclose(iso, cov, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} SongUNet checks passed.")
