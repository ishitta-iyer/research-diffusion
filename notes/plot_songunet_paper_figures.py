"""Rebuild the compact SongUNet-facing figures from saved experiment artifacts.

This script only reads completed ``results/data`` artifacts and writes PNGs to
``results/figures``.  It does not train or re-evaluate any model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
SONG_DIR = ROOT / "results" / "data" / "songunet_cov_tikhonov"
GMM_PATH = (
    ROOT
    / "results"
    / "data"
    / "gmm_rescaled_cov_tikhonov"
    / "gmm_rescaled_cov_tikhonov_full.pt"
)
MAT_DIR = ROOT / "results" / "data" / "baptista_matern_n2"

C_VALUES = (0.003, 0.01, 0.03, 0.1)
BANDS = (
    ("coarse", "Coarse", r"$0.5 \leq k < 4$"),
    ("mid1", "Mid 1", r"$4 \leq k < 10$"),
    ("mid2", "Mid 2", r"$10 \leq k < 18$"),
    ("fine", "Fine", r"$18 \leq k < 32$"),
)
ISO_COLOR = "#D95F02"
COV_COLOR = "#5E3C99"
EMP_COLOR = "#0072B2"
GATE_COLOR = "#222222"


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def song_run(variant, c):
    return load(SONG_DIR / f"arm_{variant}_c{c}_result.pt")


def final(variant, c):
    return song_run(variant, c)["eval_log"][-1]


def finish(fig, filename, *, top=0.91, bottom=0.14):
    fig.subplots_adjust(top=top, bottom=bottom, wspace=0.24)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out.relative_to(ROOT))


def plot_training_samples():
    run = load(SONG_DIR / "arm_gate_c0_result.pt")
    data = run["data"]
    d_minus = run["D_minus"]
    vmax = float(data.abs().max())
    norms = data.flatten(1).norm(dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.15))
    for j, ax in enumerate(axes):
        image = ax.imshow(
            data[j, 0], vmin=-vmax, vmax=vmax, cmap="RdBu_r", interpolation="nearest"
        )
        ax.set_title(rf"$x_{j}$,  $\|x_{j}\|={norms[j]:.2f}$", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    fig.suptitle(rf"Training samples (rescaled, $D_- = {d_minus:.3f}$)", fontsize=13)
    finish(fig, "baptista_matern_n2_rescaled_training_data.png", top=0.84, bottom=0.03)


def plot_matern_geometry_curves():
    channels = (4, 8, 16, 32)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    for i, width in enumerate(channels):
        color = cmap(i / (len(channels) - 1))
        for geometry, linestyle in (("rescaled", "-"), ("unit", "--")):
            run = load(MAT_DIR / f"arm_{geometry}_c{width}_result.pt")
            log = run["eval_log"]
            label = f"{geometry.capitalize()}, {run['n_params']:,} parameters"
            ax.plot(
                [row["epoch"] for row in log],
                [row["nn_rel_median"] for row in log],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                label=label,
            )
    ax.axhline(1.0, color="0.55", linewidth=0.9, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Median relative distance")
    ax.set_title("SongUNet memorisation on Matérn fields")
    ax.grid(True, which="major", color="0.88", linewidth=0.7)
    ax.legend(fontsize=7.2, ncol=2, frameon=False)
    finish(fig, "baptista_rect_vs_matern_n2.png", top=0.91, bottom=0.14)


def plot_gmm_vs_songunet():
    gmm = load(GMM_PATH)["results"]
    iso_cs = (0.01, 0.03, 0.1)
    cov_cs = (0.003, 0.01, 0.03)
    models = (
        ("Closed-form GMM", lambda variant, c: gmm[(variant, c)]),
        ("Trained SongUNet", lambda variant, c: final(variant, c)),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
    for ax, (panel_title, getter) in zip(axes, models):
        x = np.arange(3)
        iso = np.array([getter("isotropic", c)["coarse_score"] for c in iso_cs])
        empirical = np.array([getter("cov_emp_n2", c)["coarse_score"] for c in cov_cs])
        analytic = np.array([getter("cov_population", c)["coarse_score"] for c in cov_cs])
        ax.plot(x, iso, color=ISO_COLOR, marker="s", linewidth=2.2, label="Isotropic")
        ax.plot(
            x,
            empirical,
            color=EMP_COLOR,
            marker="o",
            linewidth=2.2,
            label=r"Covariance, $\lambda$ from 2 fields",
        )
        ax.plot(
            x,
            analytic,
            color=COV_COLOR,
            marker="^",
            linewidth=2.2,
            label=r"Covariance, analytic $\lambda(k)$",
        )
        for j, ratio in enumerate(iso / empirical):
            ax.annotate(
                rf"${ratio:.1f}\times$",
                (j, np.sqrt(iso[j] * empirical[j])),
                xytext=(8, 0),
                textcoords="offset points",
                color=EMP_COLOR,
                fontsize=9,
                va="center",
            )
        labels = []
        for iso_c, cov_c in zip(iso_cs, cov_cs):
            fi = getter("isotropic", iso_c)["fine_score"]
            fc = getter("cov_emp_n2", cov_c)["fine_score"]
            labels.append(
                f"{fi:.3f} / {fc:.3f}\n" + rf"$c$: {iso_c:g} / {cov_c:g}"
            )
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 0.25)
        ax.set_title(panel_title)
        ax.set_xlabel("Fine-band score and penalty strength\n(isotropic / covariance)")
        ax.grid(True, which="major", axis="y", color="0.88", linewidth=0.7)
    axes[0].set_ylabel("Coarse-band ring score")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("GMM vs SongUNet memorisation", fontsize=15)
    finish(fig, "gmm_vs_songunet_cov_tikhonov_matched.png", top=0.83, bottom=0.22)


def plot_band_scores():
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.45), sharey=True)
    for ax, (key, label, interval) in zip(axes, BANDS):
        iso = [final("isotropic", c)[f"{key}_score"] for c in C_VALUES]
        cov = [final("cov_population", c)[f"{key}_score"] for c in C_VALUES]
        ax.plot(C_VALUES, iso, color=ISO_COLOR, marker="s", linewidth=1.9, label="Isotropic")
        ax.plot(
            C_VALUES,
            cov,
            color=COV_COLOR,
            marker="^",
            linewidth=1.9,
            label="Analytic covariance",
        )
        ax.set_xscale("log")
        ax.set_title(f"{label}\n{interval}", fontsize=10)
        ax.set_xlabel("Penalty strength $c$")
        ax.grid(True, which="major", color="0.88", linewidth=0.7)
    axes[0].set_ylabel("Band score")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(r"SongUNet band scores ($n_{\mathrm{train}}=2$)", fontsize=13)
    finish(fig, "songunet_cov_tikhonov_audit_band_scores_vs_c.png", top=0.76, bottom=0.19)


def plot_per_k():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.3), sharex=True, sharey=True)
    for ax, c in zip(axes.ravel(), C_VALUES):
        iso = final("isotropic", c)["mean_ratio"].numpy()
        cov = final("cov_population", c)["mean_ratio"].numpy()
        k = np.arange(len(iso))
        supported = (k >= 1) & (k < 32)
        ax.plot(k[supported], iso[supported], color=ISO_COLOR, linewidth=1.9, label="Isotropic")
        ax.plot(
            k[supported],
            cov[supported],
            color=COV_COLOR,
            linewidth=1.9,
            label="Analytic covariance",
        )
        ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
        ax.axvspan(0.5, 4, color="#4C78A8", alpha=0.12)
        ax.axvspan(18, 32, color="#E45756", alpha=0.10)
        ax.set_xlim(1, 32)
        ax.set_ylim(0, 1.08)
        ax.set_title(rf"$c={c:g}$", fontsize=10)
        ax.grid(True, which="major", color="0.9", linewidth=0.65)
    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Ring score")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(r"SongUNet ring scores ($n_{\mathrm{train}}=2$)", fontsize=13)
    finish(fig, "songunet_cov_tikhonov_audit_per_k.png", top=0.90, bottom=0.10)


def plot_lambda_pool_size():
    empirical_ns = (2, 8, 16, 32)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.15))
    for ax, (key, title, _) in zip(axes, (BANDS[0], BANDS[-1])):
        empirical = np.array(
            [
                [final(f"cov_emp_n{n}", c)[f"{key}_score"] for c in C_VALUES]
                for n in empirical_ns
            ]
        )
        analytic = np.array(
            [final("cov_population", c)[f"{key}_score"] for c in C_VALUES]
        )
        ax.fill_between(
            C_VALUES,
            empirical.min(axis=0),
            empirical.max(axis=0),
            color=EMP_COLOR,
            alpha=0.16,
            label=r"Empirical $\lambda$: range over 2, 8, 16, 32 fields",
        )
        ax.plot(
            C_VALUES,
            empirical.mean(axis=0),
            color=EMP_COLOR,
            marker="o",
            linewidth=2,
            label="Empirical mean",
        )
        ax.plot(
            C_VALUES,
            analytic,
            color=COV_COLOR,
            marker="^",
            linestyle="--",
            linewidth=2,
            label=r"Analytic $\lambda(k)$",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Penalty strength $c$")
        ax.set_ylabel("Band score")
        ax.set_title(f"{title} band")
        ax.grid(True, which="major", color="0.88", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Covariance-spectrum estimation by pool size", fontsize=14)
    finish(fig, "songunet_cov_tikhonov_headline_lambda_pool_size.png", top=0.82, bottom=0.16)


def plot_collapse_dynamics():
    gate = load(SONG_DIR / "arm_gate_c0_result.pt")["eval_log"]
    iso = song_run("isotropic", 0.1)["eval_log"]
    cov = song_run("cov_population", 0.1)["eval_log"]
    series = (
        ("Unregularized", gate, GATE_COLOR, "--"),
        (r"Isotropic, $c=0.1$", iso, ISO_COLOR, "-"),
        (r"Analytic covariance, $c=0.1$", cov, COV_COLOR, "-"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.9))
    for label, log, color, linestyle in series:
        epoch = [row["epoch"] for row in log]
        axes[0].plot(
            epoch,
            [row["fraction"] for row in log],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            epoch,
            [row["nn_rel_median"] for row in log],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
    # The two regularized collapse fractions are identically zero. Show the
    # analytic-covariance checkpoints as markers so both coincident series remain visible.
    cov_epoch = [row["epoch"] for row in cov]
    axes[0].plot(
        cov_epoch,
        [row["fraction"] for row in cov],
        linestyle="none",
        marker="o",
        markersize=3.2,
        markevery=5,
        color=COV_COLOR,
    )
    axes[0].set_ylabel("Collapse fraction")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Thresholded collapse")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median relative distance")
    axes[1].set_title("Distance to nearest training sample")
    for ax in axes:
        ax.set_xlabel("Training epoch")
        ax.grid(True, which="major", color="0.88", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(r"SongUNet collapse dynamics ($n_{\mathrm{train}}=2$)", fontsize=14)
    finish(fig, "songunet_cov_tikhonov_collapse_dynamics.png", top=0.81, bottom=0.16)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_samples()
    plot_matern_geometry_curves()
    plot_gmm_vs_songunet()
    plot_band_scores()
    plot_per_k()
    plot_lambda_pool_size()
    plot_collapse_dynamics()


if __name__ == "__main__":
    main()
