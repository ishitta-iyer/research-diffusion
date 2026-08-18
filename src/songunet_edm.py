# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Upstream-compatible EDM mechanics for the active SongUNet experiments.

The loss and sampler derive from Baptista's DiffusionModelDynamics commit
``2719b5d50601deb4f17fdf5306b6e26495ba19f4`` and NVIDIA EDM. Training helpers
preserve the batch-1 shuffled DataLoader, LR ramp, and EMA equations in Baptista's
``RectangleImages/main.py``. The regularizers reproduce the active notebooks' forward-FFT,
sum-reduction convention and are default-off.
"""

import copy
import math

import numpy as np
import torch


class EDMLoss:
    """EDM loss from ``RectangleImages/training/loss.py``."""

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y) * sigma
        denoised = net(y + noise, sigma, labels, augment_labels=augment_labels)
        return weight * ((denoised - y) ** 2)


def edm_sigma_schedule(net, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                       device=None, dtype=torch.float64):
    """Exact upstream ``num_steps`` discretization followed by the terminal zero."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
             * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([net.round_sigma(steps), torch.zeros_like(steps[:1])])


def edm_sampler(net, latents, class_labels=None, randn_like=torch.randn_like,
                num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float("inf"), S_noise=1,
                sampler_dtype=None):
    """EDM Algorithm 2, with float32 used only when the device cannot use float64."""
    if sampler_dtype is None:
        sampler_dtype = torch.float32 if latents.device.type == "mps" else torch.float64
    time_steps = edm_sigma_schedule(net, num_steps=num_steps, sigma_min=sigma_min,
                                    sigma_max=sigma_max, rho=rho, device=latents.device,
                                    dtype=sampler_dtype)
    x_next = latents.to(sampler_dtype) * time_steps[0]
    for index, (time_cur, time_next) in enumerate(zip(time_steps[:-1], time_steps[1:])):
        x_cur = x_next
        gamma = (min(S_churn / num_steps, np.sqrt(2) - 1)
                 if S_min <= time_cur <= S_max else 0)
        time_hat = net.round_sigma(time_cur + gamma * time_cur)
        x_hat = (x_cur + (time_hat ** 2 - time_cur ** 2).sqrt()
                 * S_noise * randn_like(x_cur))
        denoised = net(x_hat, time_hat, class_labels).to(sampler_dtype)
        derivative_cur = (x_hat - denoised) / time_hat
        x_next = x_hat + (time_next - time_hat) * derivative_cur
        if index < num_steps - 1:
            denoised = net(x_next, time_next, class_labels).to(sampler_dtype)
            derivative_prime = (x_next - denoised) / time_next
            x_next = x_hat + (time_next - time_hat) * (
                0.5 * derivative_cur + 0.5 * derivative_prime)
    return x_next


def make_batch1_dataloader(data, shuffle=True, generator=None):
    """Baptista's batch-1 ``TensorDataset``/``DataLoader`` construction."""
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle,
                                       generator=generator)


def ramped_learning_rate(base_lr, cur_nimg, lr_rampup_kimg=10_000):
    return base_lr * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)


def ema_beta(batch_size, cur_nimg, ema_halflife_kimg=500, ema_rampup_ratio=0.05):
    halflife_nimg = ema_halflife_kimg * 1000
    if ema_rampup_ratio is not None:
        halflife_nimg = min(halflife_nimg, cur_nimg * ema_rampup_ratio)
    return 0.5 ** (batch_size / max(halflife_nimg, 1e-8))


@torch.no_grad()
def update_ema_(ema, net, beta):
    for ema_parameter, net_parameter in zip(ema.parameters(), net.parameters()):
        ema_parameter.copy_(net_parameter.detach().lerp(ema_parameter, beta))


def tikhonov_penalty_isotropic(denoised, noisy, sigma, c):
    """Active SongUNet isotropic penalty: pixel sum divided by sigma squared."""
    sigma_squared = sigma.reshape(-1, 1, 1, 1) ** 2
    difference_squared = (denoised - noisy) ** 2
    return c * difference_squared.sum(dim=(1, 2, 3)) / sigma_squared.flatten()


def tikhonov_penalty_covariance(denoised, noisy, sigma, c, inv_lambda):
    """Active SongUNet covariance penalty: forward-FFT mode sum, nulls already zero."""
    sigma_squared = sigma.reshape(-1, 1, 1, 1) ** 2
    difference_hat = torch.fft.fft2((denoised - noisy).squeeze(1), norm="forward")
    per_mode = difference_hat.abs() ** 2 * inv_lambda.unsqueeze(0).to(difference_hat.device)
    return (c / sigma_squared.flatten()) * per_mode.sum(dim=(-2, -1))


def edm_training_loss(net, images, loss_config=None, regularizer=None):
    """Expose the exact EDM draw while allowing a default-off active regularizer.

    ``regularizer``, when provided, receives ``(denoised, noisy, sigma)`` and returns one
    scalar per sample. Its result carries the same EDM weight and sample-sum reduction as
    the denoising term used by the active covariance notebooks.
    """
    loss_config = EDMLoss() if loss_config is None else loss_config
    sigma = (torch.randn([images.shape[0], 1, 1, 1], device=images.device)
             * loss_config.P_std + loss_config.P_mean).exp()
    weight = ((sigma ** 2 + loss_config.sigma_data ** 2)
              / (sigma * loss_config.sigma_data) ** 2)
    noisy = images + torch.randn_like(images) * sigma
    denoised = net(noisy, sigma)
    loss = (weight * ((denoised - images) ** 2)).sum() / images.shape[0]
    if regularizer is not None:
        penalty = regularizer(denoised, noisy, sigma)
        loss = loss + (weight.flatten() * penalty).sum() / images.shape[0]
    return loss


def train_songunet_epochs(net, data, epochs=50_000, device="cpu", lr=1e-3,
                          betas=(0.9, 0.999), eps=1e-8, lr_rampup_kimg=10_000,
                          ema_halflife_kimg=500, ema_rampup_ratio=0.05,
                          loss_config=None, regularizer=None, seed=None):
    """Minimal source-compatible version of Baptista's epoch-based training loop.

    This helper intentionally has no plotting, checkpoint naming, or experiment orchestration.
    """
    if seed is not None:
        torch.manual_seed(seed)
    net = net.train().requires_grad_(True).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, eps=eps)
    data_loader = make_batch1_dataloader(data, shuffle=True)
    cur_nimg = 1
    loss_history = []
    for _ in range(epochs):
        total_loss = 0.0
        count = 0
        net.train()
        for (images,) in data_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = edm_training_loss(net, images, loss_config=loss_config,
                                     regularizer=regularizer)
            loss.backward()
            current_lr = ramped_learning_rate(lr, cur_nimg, lr_rampup_kimg)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            for parameter in net.parameters():
                if parameter.grad is not None:
                    torch.nan_to_num(parameter.grad, nan=0, posinf=1e5,
                                     neginf=-1e5, out=parameter.grad)
            optimizer.step()
            beta = ema_beta(images.shape[0], cur_nimg, ema_halflife_kimg,
                            ema_rampup_ratio)
            update_ema_(ema, net, beta)
            total_loss += loss.item()
            count += images.shape[0]
            cur_nimg += images.shape[0]
        loss_history.append(total_loss / count)
    return dict(net=net, ema=ema, optimizer=optimizer, cur_nimg=cur_nimg,
                loss_history=loss_history)
