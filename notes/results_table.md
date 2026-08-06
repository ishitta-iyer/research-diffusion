# Consolidated results table

Generated from the artifacts on disk. Do not hand-edit; regenerate with
`python3 notes/build_results_table.py`.
Repo HEAD `11385d5`, branch `followup`, built 2026-08-06.

Every number below is read from a `results/data/*.pt` or recomputed from the seeded dataset
at build time. Provenance is given per block. Numbers that cannot be traced to an artifact
are listed in section 10 and are not for citation.

## 0. Conventions

Held fixed across sections 5 to 8 (the U-Net blocks). Sections 3 and 4 are closed-form and
use `sigma_max=80` with 500 SDE steps; where that matters it is stated in the block.

| setting | value |
|---|---|
| diffusion | VE-EDM, `sigma_min=0.002`, `sigma_max=10.0` |
| sampler | Euler-Maruyama reverse SDE, 1000 steps, linear grid in `t` (geometric in sigma) |
| latents | `torch.manual_seed(42)` before every draw, so all runs share latents and step noise |
| generated samples | `G = 16` |
| reference draws | `J = 32` |
| `exclude_nn` | `True` |
| aggregation | `mean_of_ratios`: `mean_j(e_NN / e_rand_j)` |
| ring normalization | both terms by the generated sample's ring norm, so it cancels |
| NN selection | absolute L2 on the coarse-band component |
| bands | coarse [0.5,4), mid1 [4,10), mid2 [10,18), fine [18,32) |
| training | `train_edm`, Adam, lr 1e-3, `P_mean=-1.2`, `P_std=1.2`, `seed=0` |

**Training-seed spread, measured.** Section 6 varies the training seed and nothing else.
Pooled within-config standard deviation across 3 seeds and 6 architectures is **0.0137**
on the coarse band, at n_train=8, on CUDA. Earlier drafts of this project quoted 0.03 to 0.06,
which was never measured and should not be cited.

That figure is measured under one setting and is used here only as a rough scale for reading
gaps, not as a test statistic. It is not transferable across n_train, device or dataset
realization without qualification.

**Three devices are in play. Blocks are not comparable across them.**

| block | model trained on | metric / neutral computed on |
|---|---|---|
| 3, 4 (closed form) | n/a | CPU |
| 5, 8 | Apple MPS | CPU |
| 7 | Apple MPS | MPS (inside the notebook) |
| 6, 9 | CUDA (NVIDIA L40S) | CUDA (inside the notebook) |

The backends do not agree bit-for-bit. Measured directly: the same architecture, seed, data
and step count gives a coarse score of 0.8742 on MPS and 0.8399 on CUDA at 4000 steps, a
difference of 0.034, larger than the seed spread above. Additionally the reference-draw RNG
stream differs by backend, so even the neutral line shifts slightly (0.8443 on CPU against
0.8436 on MPS at n_train=8).

Consequence: **compare gaps only within a block.** Do not compare an absolute score in one
block to an absolute score in another, and do not treat a threshold measured in one block as
exact in another.

One convention here is *not* matched to Baptista et al. and should be stated wherever their
results are compared to these. Their section 5.4 reads: "The reverse process for sampling is
the SDE-based methodology presented in [19] using 40 backward steps." This project uses 1000
in sections 5 to 8 and 500 in sections 3 and 4, a factor of 12.5 to 25 more. The effect of
that difference has never been measured, and it is the *spacing* half of pending experiment 2.

Two consequences that every table below depends on.

1. **The 1/n_train floor.** With `exclude_nn=False` a random reference draw *is* the nearest
   neighbour about `1/n_train` of the time, and each such draw contributes ratio 1. The score
   cannot fall below about `1/n_train` however completely the model copies. Measured on a
   near-perfect memorizer:

   | n_train | 2 | 4 | 8 | 16 | 32 |
   |---|---|---|---|---|---|
   | `1/n_train` | 0.500 | 0.250 | 0.125 | 0.062 | 0.031 |
   | measured, `exclude_nn=False` | 0.531 | 0.258 | 0.137 | 0.057 | 0.025 |
   | measured, `exclude_nn=True` | 0.00009 | 0.00009 | 0.00008 | 0.00008 | 0.00008 |

2. **A coarse score below 1 is not on its own evidence of copying.** The nearest neighbour is
   an argmin over `n_train` candidates, so held-out real fields, which cannot have memorized
   anything, also score well below 1. This is the line every U-Net number must be read
   against. Baptista confirmed by email that both artefacts are expected properties of the
   baseline rather than bugs.

## 1. Dataset and geometry

Four-band Whittle-Matern mixture, 128x128 periodic grid, 200 samples, seed 42, globally
normalized. Components (name, tau, band, weight): coarse tau=2 (0.5, 4.0) w=1.0, mid1 tau=6 (4.0, 10.0) w=0.8, mid2 tau=12 (10.0, 18.0) w=0.8, fine tau=24 (18.0, 32.0) w=1.2; s=2.0, sigma_sq=1.0 per component.
Realized std over the full 200-field pool 1.0000. `train_edm` estimates `sigma_data`
from the training subset, so it differs per n_train: n=2: 1.0521, n=4: 0.9491, n=8: 0.9810, n=16: 0.9886, n=32: 0.9982.
Training sets are always the first `n_train` fields.

Geometry, recomputed here. `D_min` is the minimum pairwise Euclidean distance inside the
training set, which is the quantity Assumption 4.1 of Baptista et al. calls `D_-`.

| n_train | `D_min` | `sigma_max/D_min` at 10 | at 80 |
|---|---|---|---|
| 2 | 185.9 | 0.054 | 0.430 |
| 4 | 157.9 | 0.063 | 0.507 |
| 8 | 146.7 | 0.068 | 0.545 |
| 16 | 132.2 | 0.076 | 0.605 |
| 32 | 121.0 | 0.083 | 0.661 |

Mean field norm `||x_0|| = 127.0`; mean pairwise distance over the
full 200-field pool 179.5.

For comparison, the rectangles dataset of Baptista et al. section 5.4 is 64x64 binary with
`N=2` squares at opposite corners. Two quantities are needed to place it on the same axis and
**neither is stated in their paper**, so both are inferences and are labelled as such here.

- Their square side length is never given ("a square of equal length and width"), so
  `||x_0|| ~ 19` and `D_- ~ 27` are back-derived from a plausible side length, not read off.
- Their `sigma_max` is never given either: the strings `sigma_max` and `sigma-max` appear
  **zero** times in the paper. The value 80 is the default of their reference [19] (Karras
  et al., EDM) and is assumed, not sourced.

Under those two assumptions `sigma_max/D_- ~ 2.9` for their setup against 0.05 to 0.08 here.
The resulting ratio of roughly 40x is **not one effect but two**, and they are separately
actionable:

- *Dataset geometry.* Their `D_- ~ 27` against this dataset's `D_min = 147` at n_train=8 is
  a factor of about 5x, and it is a property of the data, not a knob. Changing it means
  changing the dataset (which is what section 8, the intrinsic-dimension probe, does).
- *Sampler choice.* Their assumed `sigma_max=80` against this project's 10 is a further 8x,
  and it **is** a free knob. This is the half pending experiment 2 can actually move.

Quoting a single 40x figure conflates the two and makes the gap look like a modelling error
when most of it is a deliberate dataset choice. This is the axis Prof. Baptista's email refers
to and it is the subject of pending experiment 2 (section 10).

## 2. Reference lines

The two lines every learned-model number is read between. Recomputed here; the neutral line
is the mean over 5 reference-draw seeds with its spread, which also gives the metric's
reference-draw noise floor.

| n_train | neutral coarse | neutral fine | neutral median rel NN dist | GMM coarse | GMM fine |
|---|---|---|---|---|---|
| 2 | 0.8567 +/- 0.0000 | 1.0048 +/- 0.0000 | 1.280 | 0.00007 | 0.0086 |
| 4 | 0.9002 +/- 0.0052 | 1.0014 +/- 0.0001 | 1.246 | 0.00008 | 0.0086 |
| 8 | 0.8443 +/- 0.0039 | 1.0014 +/- 0.0003 | 1.108 | 0.00008 | 0.0085 |
| 16 | 0.8114 +/- 0.0035 | 1.0030 +/- 0.0003 | 1.077 | 0.00009 | 0.0084 |
| 32 | 0.8130 +/- 0.0033 | 1.0030 +/- 0.0006 | 1.051 | 0.00009 | 0.0084 |

Neutral line: 16 held-out real fields (`x_all[100:116]`, disjoint from every training set)
scored against the training set through the identical metric. Recomputed in the build script.
GMM line: closed-form empirical-score minimizer sampled through the identical sampler and
latents, from `edm_unet_transition_results.pt`.

The neutral coarse line sits between about 0.81 and 0.90 and never reaches 1. It is not
monotonic in n_train: it peaks at n_train=4. The reference-draw spread is at or below 0.006,
so differences smaller than about 0.01 are not resolvable from the metric alone, before seed
variation is considered.

The zero spread at n_train=2 is not a bug. With `exclude_nn=True` and two training fields
only one reference remains, so all J draws are identical and the reference-draw variance is
exactly zero. This is also the point at which mean-of-ratios and ratio-of-means coincide
exactly; from n_train=4 up they differ by about 0.03.

**This table is canonical.** Three sets of neutral-line values are currently in the repo:
this table, `notes/project_map.md:104`, and the docstring at `src/memorization_metrics.py:102`.
The latter two are identical to each other (0.8567 / 0.9050 / 0.8448 / 0.8150 / 0.8126) and
differ from this table by up to 0.005. The difference has been traced and it is **reference-draw
seed noise, nothing else**: no choice of `n_rand_ref` in {32, 64} and no reference seed in
0 to 5 reproduces those five values exactly, the closest being max error 0.0035, and the
5-seed means at `n_rand_ref=32` and 64 agree with each other to about 0.001. The published set
is a single unrecorded draw; this table is a 5-seed mean and is the one to quote. The metric,
the held-out set and the convention are the same in all three.

The held-out set was confirmed by the same search: `x_all[100:116]` reproduces the published
values to within seed noise, while an alternative slice `x_all[32:48]` is off by up to 0.054.
The n_train=2 entry matches to 0.0000 under every configuration tested, which is the
zero-variance special case described below.

## 3. Result A. The closed-form empirical score memorizes at every scale

Sampling with the exact minimizer of the empirical score-matching loss (Theorem 3.2 of
Baptista et al.) returns the training data. Two independent measurements of the same thing:

| source | n_train | coarse | mid1 | mid2 | fine | pixel collapse frac | median rel NN dist |
|---|---|---|---|---|---|---|---|
| Tikhonov sweep, c=0 | 32 | 0.00008 | 0.00083 | 0.00344 | 0.00842 | n/a | n/a |
| transition sweep GMM ref | 2 | 0.00007 | n/a | n/a | 0.0086 | 1.00 | 0.0019 |
| transition sweep GMM ref | 8 | 0.00008 | n/a | n/a | 0.0085 | 1.00 | 0.0021 |
| transition sweep GMM ref | 32 | 0.00009 | n/a | n/a | 0.0084 | 1.00 | 0.0020 |

Reading. Every band is far below both the neutral line (0.81 to 0.86 coarse) and any
memorization threshold; the pixel collapse fraction is 1.00 and the median relative distance
to the nearest training field is about 0.002, that is, the generated field *is* a training
field to three decimal places. This is the controlled positive control for the whole study:
it runs through the same sampler, latents, metric and code path as every U-Net number below.

The coarse and fine values differ by roughly two orders of magnitude (about 8e-5 against
about 8e-3) while both sit far below any threshold. Whether to describe this as *uniform*
memorization is a wording question worth settling deliberately: under a threshold criterion
all bands are fully memorized, but the band values are not equal.

The two row groups are a useful robustness check rather than a caveat. They were produced by
**different samplers**: the Tikhonov row uses `sigma_max=80` with 500 SDE steps, the
transition rows use `sigma_max=10` with 1000. They agree to 0.00008 against 0.00009 on the
coarse band and 0.0084 against 0.0086 on the fine band. The positive control is therefore
insensitive to the sampler difference that separates sections 3 and 4 from sections 5 to 8,
which is what licenses reading the two groups of blocks against each other at all.

*Provenance:* `notebooks/multiscale/gmm_tikhonov_variants_comparison.ipynb` -> `results/data/gmm_tikhonov_variants_sweep.pt` (2026-08-04 22:25). Second row group from `edm_unet_transition_results.pt`. Closed form, no training, no seed.

## 4. Result B. Tikhonov variants, closed-form score

Five instances of a matrix-weighted Tikhonov penalty `Gamma(t) = (c/sigma^2) W`, with
minimizer `s* = (I + Gamma)^-1 grad log p^N`. In the Fourier eigenbasis the score denominator
becomes `sigma^2 + c_eff(k)`:

| variant | `c_eff(k)` | `W` | role |
|---|---|---|---|
| `isotropic` | `c` | `I` | plain Tikhonov, eq. (5.7) |
| `iso_budget_matched` | `c * mean_k(1/lambda(k))` | `I` | same *average* regularization as covariance, spread uniformly |
| `covariance` | `c / lambda(k)` | `Sigma^-1` | the covariance weighting |
| `covariance_population` | `c / lambda_pop(k)` | `Sigma^-1` | robustness to spectrum estimation |
| `anti_weighted` | `c * lambda(k)` | `Sigma` | direction control |

**Attribution.** Theorem 5.1 of Baptista et al. states the minimizer for a general matrix
`Gamma(t)`, so the *form* above is theirs. But the paper only ever instantiates the isotropic
case `Gamma(t) = (c/sigma^2) I`, their eq. (5.7), which is the `isotropic` row alone. The four
weighted variants, and in particular the covariance weighting and its budget-matched control,
are **not tested anywhere in that paper**. They are this project's contribution and should be
presented as such rather than as a reproduction of their section 5.1.

Budget factor `mean_inband(1/lambda) = 30.67`. Null modes
(`lambda < 1e-6 * mean`, that is DC and out-of-band k>32) left unregularized in every
weighted variant, pseudo-inverse semantics. n_train = 32.

### 4a. Band scores, coarse / fine

| c | isotropic | iso budget matched | covariance | covariance population | anti weighted |
|---|---|---|---|---|---|
| 0 | 0.00008 / 0.008 | 0.00008 / 0.008 | 0.00008 / 0.008 | 0.00008 / 0.008 | 0.00008 / 0.008 |
| 0.0001 | 0.00041 / 0.043 | 0.00237 / 0.227 | 0.00008 / 0.255 | 0.00008 / 0.248 | 0.00769 / 0.011 |
| 0.001 | 0.00132 / 0.132 | 0.00739 / 0.588 | 0.00011 / 0.618 | 0.00011 / 0.607 | 0.02452 / 0.024 |
| 0.01 | 0.00429 / 0.386 | 0.02349 / 0.914 | 0.00025 / 0.916 | 0.00025 / 0.912 | 0.07516 / 0.070 |
| 0.1 | 0.01322 / 0.792 | 0.07350 / 0.990 | 0.00080 / 0.990 | 0.00078 / 0.989 | 0.21925 / 0.216 |
| 1 | 0.04279 / 0.971 | 0.21532 / 0.999 | 0.00271 / 0.999 | 0.00264 / 0.999 | 0.58318 / 0.575 |
| 5 | 0.09262 / 0.994 | 0.42382 / 1.000 | 0.00607 / 1.000 | 0.00589 / 1.000 | 0.80952 / 0.844 |

At `c = 0.01` the covariance and budget-matched variants reach the same fine-band value
(0.916 against 0.914) while their coarse values differ by
92x (0.00025 against 0.02349).
An earlier run on a different reference-draw seed gave 85x, so the separation is stable to
about 10 percent and should be quoted as roughly two orders of magnitude, not to 2 digits.

The budget-matched control is the load-bearing comparison: it applies the same *average*
effective constant uniformly across modes. That it de-memorizes both bands together, while
covariance separates them, is suggestive that the allocation across modes rather than the
total magnitude is what produces the selectivity.

`covariance_population` tracks `covariance` to within 1.8e-04 on the coarse band across all
c, so the 32-sample spectrum estimate does not affect the coarse result at all. On the
**fine** band the two differ by up to 1.1e-02 (at c = 0.001), which is not negligible: it is the
same order as the c-to-c changes being reported in that band at small c. The right reading
is that the empirical spectrum is adequate for the coarse claim and only approximately
adequate for the fine one.

### 4b. Band power fidelity, generated / training

The same runs, but asking what the regularizer does to the *spectrum* rather than to the
memorization ratio. A value of 1.0 matches the data.

| c | isotropic<br>coarse / fine | iso budget matched<br>coarse / fine | covariance<br>coarse / fine | covariance population<br>coarse / fine | anti weighted<br>coarse / fine |
|---|---|---|---|---|---|
| 0 | 1.02 / 1 | 1.02 / 1 | 1.02 / 1 | 1.02 / 1 | 1.02 / 1 |
| 0.0001 | 1.02 / 1.01 | 1.02 / 1.1 | 1.02 / 1.13 | 1.02 / 1.12 | 1.02 / 1 |
| 0.001 | 1.02 / 1.04 | 1.02 / 1.99 | 1.02 / 2.25 | 1.02 / 2.17 | 1.02 / 1 |
| 0.01 | 1.02 / 1.33 | 1.02 / 10.8 | 1.02 / 13.4 | 1.02 / 12.6 | 1.04 / 1.01 |
| 0.1 | 1.02 / 4.2 | 1.03 / 98.1 | 1.02 / 124 | 1.02 / 116 | 1.12 / 1.1 |
| 1 | 1.02 / 32.8 | 1.1 / 974 | 1.02 / 1.23e+03 | 1.02 / 1.15e+03 | 2.03 / 1.97 |
| 5 | 1.04 / 159 | 1.35 / 4.89e+03 | 1.02 / 6.16e+03 | 1.02 / 5.76e+03 | 5.44 / 5.86 |

This qualifies 4a and needs to be reported with it. The fine-band ratio approaching 1 is
not on its own evidence of good novel fine-scale content: at `c = 0.01` the covariance
variant puts about 13x the true power into the fine band, and both errors entering the
ratio are then dominated by injected power. What survives cleanly is the *coarse* column,
where covariance holds 1.02 at every c tested while budget-matched drifts to 1.35 and
anti-weighted to 5.4. A comparison at matched fine-band fidelity rather than matched c
would be the sharper statement and is not yet run.

Spectrum estimation error, empirical against near-population (2048 fresh fields, seed 7):
14.4 percent mean per-mode in-band, printed by the notebook at build time.

### 4c. The same comparison at matched spectral fidelity

Section 4b raises an objection to 4a: at `c = 0.01` the covariance variant injects about 13x
the true fine-band power, so comparing it to budget-matched at the same `c` may be comparing
two different amounts of spectral damage. The objection is answered by comparing at matched
fine-band power fidelity instead of matched `c`. Interpolating each variant's coarse score to
a common fine-band power ratio, on the 7-point log grid in `c`:

| fine-band power ratio | isotropic | iso budget matched | covariance | vs isotropic | vs budget matched |
|---|---|---|---|---|---|
| 2.0x | 0.00638 | 0.00741 | **0.00010** | 62x | 72x |
| 10.8x | 0.02004 | 0.02350 | **0.00022** | 91x | 107x |
| 13.4x | 0.02273 | 0.02499 | **0.00025** | 90x | 98x |

Both separations are given because they are different comparisons and 4a quotes only the
second. At matched `c = 0.01` the two controls are far apart (isotropic 16.9x, budget-matched
92.5x), because isotropic at that `c` has barely started regularizing. At matched fidelity
they converge, since isotropic must be driven to a much larger `c` to reach the same
spectral distortion. The convergence is the point: **whichever isotropic control is used,
the covariance weighting holds the coarse band one to two orders of magnitude tighter at
equal damage to the fine band.** The separation is therefore not an artifact of unequal
regularization strength. What differs is how the penalty is allocated across modes.

Caveat on method: these are linear interpolations on a coarse log grid in `c`, so the
implied `c` values are approximate. The ordering and the order of magnitude hold at all
three targets, which is what the comparison rests on. A direct run at matched fidelity would
be cleaner and is cheap, since this block is closed-form and needs no training.

*Provenance:* `notebooks/multiscale/gmm_tikhonov_variants_comparison.ipynb` -> `results/data/gmm_tikhonov_variants_sweep.pt` (2026-08-04 22:25). Closed form, no training, no seed. Sampler for this block is `sigma_max=80`, 500 SDE steps, which differs from the U-Net blocks below; it is internally consistent across all five variants. Figures `results/figures/tikvariants_{per_k,band_scores,generated_spectra}.png`.

## 5. Result C. U-Net, training-set size against training time

`SmallUNet(16, 64, 3)`, 195,697 parameters, batch size 8, 30k steps, 8 log-spaced checkpoints,
n_train in {2,4,8,16,32}. Coarse-band score, with the neutral line for that n_train and the
signed gap below it. Negative means below neutral, that is, in the direction of copying.

| n_train | neutral | 250 | 1000 | 4000 | 16000 | 30000 | gap at 30k | pixel collapse frac |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.8567 | 0.8789 | 0.8835 | 0.8622 | 0.8832 | 0.8656 | +0.0089 | 0.00 |
| 4 | 0.9002 | 0.9094 | 0.8977 | 0.8950 | 0.8767 | 0.8494 | -0.0508 | 0.00 |
| 8 | 0.8443 | 0.8528 | 0.8612 | 0.8742 | 0.8421 | 0.8336 | -0.0108 | 0.00 |
| 16 | 0.8114 | 0.8553 | 0.8246 | 0.8335 | 0.8341 | 0.8385 | +0.0271 | 0.00 |
| 32 | 0.8130 | 0.8229 | 0.7933 | 0.8598 | 0.7890 | 0.7865 | -0.0265 | 0.00 |

Fine band sits at 0.9979 to 1.0260 in all 40 cells, against a neutral fine
line of 1.0014 to 1.0048. Pixel collapse fraction is 0.00 in all 40 cells; the GMM
through the identical pipeline gives 1.00.

Reading. The gaps at 30k span -0.0508 to +0.0271, and 3 of 5 cells have
magnitude at or above 0.025. Both the raw U-Net scores and the neutral line vary
non-monotonically with n_train (the neutral line peaks at n_train=4), so the raw curve should
not be read as a trend in the model; only the gap column carries model information.

For scale, twice the seed spread measured in section 6 is 0.0274. 1 of 5 cells reach it: n_train=4 at -0.0508.
Those gaps point in opposite directions and the collapse fraction is 0.00 in every cell, so
they are not evidence of copying. The coarse score wanders by a few hundredths with n_train
for reasons the neutral line does not fully absorb. No claim is made here about why.

Note the threshold is measured on CUDA at n_train=8 and this block ran on MPS across several
n_train, so it is a rough scale, not a test.

What does **not** depend on that threshold, and is the load-bearing observation here: the pixel
collapse fraction is 0.00 in every cell while the closed-form score through the identical
sampler, latents and metric gives 1.00. That is a binary outcome with no threshold in it.

*Provenance:* `notebooks/multiscale/edm_unet_memorization_transition.ipynb` -> `results/data/edm_unet_transition_results.pt` (2026-07-26 16:26). Checkpoints in `edm_unet_transition_checkpoints.pt`. Single seed (`seed=0`). Figure `results/figures/unet_transition_curves.png` predates the neutral line and does not show it.

## 6. Result D. U-Net capacity across training seeds

Prof. Baptista's ask 3, and the experiment that supplies this document's significance
threshold. Six architectures x 3 training seeds at n_train=8, 30k steps. Identical to the
earlier single-seed sweep in every respect except `train_edm(seed=...)`.

Neutral coarse line 0.8477 +/- 0.0064, recomputed on the same device. Gap = score minus
neutral; negative is toward copying.

| config | params | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|---|
| C16_L2 | 58,641 | -0.0028 | -0.0002 | -0.0182 | -0.0071 | 0.0080 |
| C8_L3 | 64,345 | -0.0056 | -0.0084 | -0.0126 | -0.0089 | 0.0028 |
| C16_L3 | 195,697 | -0.0107 | -0.0204 | -0.0347 | -0.0219 | 0.0099 |
| C32_L3 | 709,153 | -0.0134 | +0.0053 | -0.0069 | -0.0050 | 0.0077 |
| C16_L4 | 729,905 | -0.0892 | -0.0375 | -0.0474 | -0.0580 | 0.0224 |
| C64_L3 | 2,739,073 | -0.0070 | -0.0171 | -0.0091 | -0.0111 | 0.0044 |

Pixel collapse fraction 0.00 in all 144 cells. GMM ceiling: coarse 0.00008, collapse 1.00.

### 6a. What this measures

Pooled within-config standard deviation across the 3 seeds: **0.0137** (ddof=1).
Per-config standard deviations range from 0.0035 to 0.0275, so the spread is not
uniform across architectures. Three seeds is enough to establish a scale and not enough to
support a significance test; nothing below is presented as one.

### 6b. What can be said

1. **Nothing memorizes.** Collapse fraction is 0.00 in all 144 cells, against a closed-form
   ceiling of 1.00 through the identical pipeline. This needs no threshold and is the
   only claim this block makes without qualification.

2. **The single-seed ordering does not replicate.** 0 of 3 seeds give a gap monotone in
   parameter count, and the seed means are not monotone. Whatever ordering the earlier
   single-seed sweep showed is not reproducible by re-running with a different seed.

3. **No trend is detectable either way.** The correlation between mean gap and log parameter
   count is -0.254 across 6 architectures, which at this sample size distinguishes nothing.
   This is a failure to detect, not a demonstrated absence of an effect. Both readings are
   open.

4. **All 18 runs sit within 0.089 of the neutral line**, against a closed-form score of
   0.00008. The entire spread of the sweep is two orders of magnitude away from
   what memorization looks like on this metric.

### 6c. What cannot be said from this data

Listed explicitly, because each is tempting and none is supported at 3 seeds:

- That parameter count has no effect. See point 3: the test cannot resolve it.
- That depth has an effect. `C16_L4` (depth 4) does sit furthest below neutral, and its
  matched-parameter partner `C32_L3` sits at -0.0050 against -0.0580. That is one
  matched pair at three seeds, selected after the fact for being the outlier. It is a
  reasonable hypothesis for a future experiment and it is not a result.
- That any architecture here is closer to memorizing than any other in a way that matters.
  All 18 runs have collapse fraction 0.00.

### 6d. Relation to the earlier single-seed sweep

`edm_unet_capacity_sweep.ipynb` ran the same architectures at seed 0 on MPS, scored against
a CPU-computed neutral. It reported the two smallest *above* the line (+0.0218, +0.0152)
and a monotone ordering. Here the same two sit at -0.0071 and -0.0089.

Device and seed both changed, so the two runs are **not comparable** and the difference
cannot be attributed to either. The multi-seed CUDA block is internally consistent and is
the one to carry forward. The single-seed MPS figures are superseded.

### 6e. Parameter range against Baptista et al.

They sweep 57,017 to 55,725,825 parameters (Figure 18 legend, verified). This sweep spans
58,641 to 2,739,073, so it overlaps at the bottom: the smallest configuration here
(C16_L2, 58,641) is within 2.8 percent of theirs (57,017), and reaches only
5 percent of their largest.

Their collapse criterion is not this one. Their Figure 18 measures the fraction of samples
at *exactly* zero Euclidean distance to a training point after thresholding each pixel to
binary; this project uses relative distance below 0.3 on continuous fields. The parameter
counts are comparable, the datasets and the collapse criteria are not, so this is a range
statement and not a like-for-like disagreement.

*Provenance:* `notebooks/multiscale/capacity_multiseed.ipynb` -> `results/data/capacity_multiseed.pt` (2026-08-06 05:05). Checkpoints in `capacity_multiseed_checkpoints.pt`. Seeds [0, 1, 2], device `cuda` (NVIDIA L40S, Killarney). Figure `results/figures/capacity_multiseed.png`. Supersedes `edm_unet_capacity_sweep.ipynb` / `edm_unet_capacity_results.pt`, which was single-seed and on MPS.

## 7. Result E. Optimizer-update budget and batch size

Directly answers Prof. Baptista's email: *"you have more steps but a larger batch size, which
leads to fewer optimizer updates than their setup. It may be worth rerunning at batch size 2
to match before concluding there is no memorization."*

Two arms matched on optimizer updates, `SmallUNet(16,64,3)` at n_train=8, 100k updates,
11 checkpoints. Neutral line measured inside the notebook: coarse 0.8412,
fine 1.0018, median rel NN dist 1.108.

| arm | batch | 1000 | 8000 | 30000 | 50000 | 100000 | gap at 50k | gap at 100k |
|---|---|---|---|---|---|---|---|---|
| bs2 | 2 | 0.8484 | 0.8437 | 0.8455 | 0.8451 | 0.8389 | +0.0039 | -0.0024 |
| bs8 | 8 | 0.8612 | 0.8547 | 0.8336 | 0.8297 | 0.8261 | -0.0116 | -0.0152 |

Pixel collapse fraction 0.00 across all 22 checkpoints in both arms. GMM through the
identical pipeline: 1.00, median rel NN dist
0.0021.

Reading. At 50,000 updates, which is Baptista et al.'s own budget, batch size 2 sits
+0.0039 from the neutral
line. The arm-to-arm difference at 100k is 0.013, inside the single-seed spread, so no
batch-size effect is resolvable. Matching the batch size does not produce memorization
here, which addresses the undertraining explanation for the null.

Caveat to carry: `train_edm` draws minibatch indices with `torch.randint`, that is, with
replacement. At batch size 8 with n_train=8 this is a bootstrap resample, about 63 percent
distinct images per step, not a full pass. The `bs8` arm must not be described as
full-batch. The `bs2` arm is unaffected and the matched-update comparison is unaffected.

**`bs8` is not an independent run.** It is bit-identical to `C16_L3` in section 6 and to the
n_train=8 row in section 5: same architecture, same n_train, same batch size, same seed, same
data. `torch.equal` on the generated sample tensors returns True at all six shared
checkpoints (250, 1000, 4000, 8000, 16000, 30000); beyond 30k it is that same run extended to
100k updates. It therefore appears in three sections of this document and must be counted
once, not three times. Only `bs2` is new work. Making this a genuine two-arm comparison would
require rerunning `bs8` at a different training seed, which has not been done.

*Provenance:* `notebooks/multiscale/unet_update_budget_batchsize.ipynb` -> `results/data/unet_update_budget_batchsize.pt` (2026-08-04 21:55). Checkpoints in `unet_update_budget_checkpoints.pt`. Single seed. Figure `results/figures/unet_update_budget_batchsize.png`.

## 8. Result F. Intrinsic dimension

Grid, network, sampler and metric held fixed at the values validated above; only the spectral
support of the data varies, so the number of active Fourier modes is the single knob.
n_train = 2 throughout, the most memorization-prone setting and the one Baptista et al. use.
Each config gets its own neutral line, because its coarse band differs.

| config | active modes | neutral coarse | U-Net @30k | gap | pixel collapse frac @30k | GMM coarse |
|---|---|---|---|---|---|---|
| d12 | 8 | 0.8913 +/- 0.0066 | 0.7994 | -0.0919 | 0.0625 | 0.00006 |
| d50 | 44 | 0.9402 +/- 0.0102 | 0.8841 | -0.0561 | 0.0000 | 0.00008 |
| d314 | 304 | 0.9402 +/- 0.0102 | 0.9184 | -0.0218 | 0.0000 | 0.00008 |
| full | 3204 | 0.9402 +/- 0.0102 | 0.8967 | -0.0436 | 0.0000 | 0.00008 |

Two internal checks pass. `d50`, `d314` and `full` return identical neutral lines, as they
must: their coarse bands are identical by construction, the added components are
band-disjoint, and the global normalization cancels in the ratio. And `d12` differs, as it
must: its coarse band is [0.5, 2) rather than [0.5, 4).

Reading. `d12`, at 8 active modes, is the only U-Net configuration anywhere in this project
with a nonzero pixel collapse fraction (0.0625, one of sixteen samples). That is the claim
worth carrying, because it needs no threshold. It is also the configuration closest to the
binary-rectangle regime of Baptista et al.

The gap column does **not** show a clean trend in dimension. `d50`, `d314` and `full` share an
identical neutral line by construction, so their gaps are directly comparable, and they run
-0.0561, -0.0218, -0.0436: non-monotone, with `full` sitting 0.0218 below `d314` despite ten
times the active modes. Only `d12` stands apart. So the honest statement is that the one
configuration with almost no active modes behaves differently, not that the gap scales with
dimension.

For scale, the seed spread measured in section 6 is 0.0137, so d12's gap of about -0.09 is
roughly 6.7x it. That comparison crosses blocks and devices (this block is MPS with a
CPU-computed neutral, section 6 is CUDA), so treat it as an order of magnitude, not a test.
One seed, one dataset realization, one collapsed sample.

**This block trains on a different realization of the data than sections 1 to 7 and its rows
are therefore not comparable to them.** `generate_multiband_dataset_postmask` draws
`noise_imag` after `noise_real`, so the position of the imaginary draw in the RNG stream
depends on `num_samples`; the generator is not prefix-stable in pool size. This block builds
64-field pools and every other block builds a 200-field pool, so at the same seed the first
two fields differ (cosine similarity 0.45 and 0.68). In particular the `full` row here is not
the n_train=2 row of section 5. Internal comparisons across the four rows of this table are
valid, because all four use the same 64-field construction. Cross-block ones are not.

*Provenance:* `notebooks/multiscale/edm_unet_dimension_probe.ipynb` -> `results/data/edm_unet_dimension_probe.pt` (2026-07-26 19:20). Single seed. Notebook was unrunnable as committed (it called a `RingMetricContext.fresh_baseline` method that does not exist); the baseline is now computed inline in the notebook, `src/` unchanged. Committed figure `results/figures/unet_dimension_probe.png` predates that fix.

## 9. Result G. Sampler sigma_max and step spacing

Prof. Baptista's ask 2. Both knobs are inference-time only: in EDM the training noise level
is drawn from the `P_mean`/`P_std` lognormal and does not depend on the sampler's
`sigma_max`, so one trained model is re-sampled under every cell and each difference is
attributable to the sampler alone. No retraining between cells.

Grid: `sigma_max` in {10, 80, 400} x `n_steps` in {40, 1000}, at n_train in {2, 8}, at the 30000-step checkpoint. `n_steps=40` is Baptista et al.'s stated
count; 1000 is this project's. The closed-form score runs in every cell as a positive
control.

**n_train = 2**, neutral coarse 0.8567 +/- 0.0000, `D_min` 185.9. Cells give gap / collapse fraction.

| `sigma_max` | `sigma_max/D_min` | 40 steps | 1000 steps |
|---|---|---|---|
| 10 | 0.054 | +0.0412 / 0.00 | +0.0033 / 0.00 |
| 80 | 0.430 | -0.0407 / 0.00 | -0.0273 / 0.00 |
| 400 | 2.152 | -0.0665 / 0.00 | +0.0006 / 0.00 |

**n_train = 8**, neutral coarse 0.8477 +/- 0.0064, `D_min` 146.7. Cells give gap / collapse fraction.

| `sigma_max` | `sigma_max/D_min` | 40 steps | 1000 steps |
|---|---|---|---|
| 10 | 0.068 | -0.0187 / 0.00 | -0.0080 / 0.00 |
| 80 | 0.545 | -0.0144 / 0.00 | -0.0032 / 0.00 |
| 400 | 2.727 | +0.0013 / 0.00 | -0.0052 / 0.00 |

**The pixel collapse fraction is 0.00 in all 12 cells**, while the closed-form score
through the identical sampler gives 1.00 in all 12 cells. No sampler setting anywhere
in this range induces memorization, including at `sigma_max/D_min` = 2.73, which brackets
the ~2.9 inferred for Baptista et al., and including at their stated 40 steps.

At n_train=8 the gaps span -0.0187 to +0.0013. At n_train=2 they span
-0.0665 to +0.0412. For scale, the seed spread measured in section 6 is
0.0137, though that was measured on a different block and is only a rough guide here.

The coarse gaps do move with `sigma_max` at 40 steps, and the two n_train move in **opposite
directions**:

| | `sigma_max`=10 | 80 | 400 |
|---|---|---|---|
| n_train=2, 40 steps | +0.0412 | -0.0407 | -0.0665 |
| n_train=8, 40 steps | -0.0187 | -0.0144 | +0.0013 |

n_train=2 moves downward with increasing `sigma_max`, n_train=8 moves upward. A sampler
effect that reverses sign with training-set size is not a sampler effect on memorization.
No claim is made here about what it is.

The samples are not degenerate at 40 steps: the fine-band score stays between
0.9932 and 1.0095 across the whole grid.

**Reading.** The one unqualified statement this block supports is that no sampler setting
tested produces any pixel collapse: 0.00 in all 12 cells, while the closed-form score
through the identical sampler gives 1.00 in all 12 cells. That covers `sigma_max/D_min`
from 0.054 to 2.73 and both 40 and 1000 backward steps, and it needs no threshold.

The coarse-gap movement is left unexplained rather than explained away. Two caveats bound
how much weight it can carry: changing `n_steps` changes the stochastic path, so the two
step columns do not share a sampling realization; and n_train=2 has exactly one legal
reference field, so its neutral line has zero reference-draw variance and its gaps are the
least stable in the document.

This supersedes `edm_unet_train_size_sweep_sigma_fix.ipynb`, whose grid held both
`sigma_max` values in the same regime and neither step count near 40.

*Provenance:* `notebooks/multiscale/sigma_spacing_sweep.ipynb` -> `results/data/sigma_spacing_sweep_v2.pt` (2026-08-06 01:33). Checkpoints in `sigma_spacing_checkpoints.pt`. Train seed 0, device `cuda` (NVIDIA L40S, Killarney). Evaluation only, no retraining between cells. Figure `results/figures/sigma_spacing_sweep.png`.

## 10. Not for citation

| item | why |
|---|---|
| `memorization_regime_sweep.ipynb`, GMM columns | VP diffusion, `exclude_nn=False`. Its n_train sweep 0.222 / 0.100 / 0.059 / 0.036 at N in {5,10,20,32} reproduces `1/N` (floors 0.184 / 0.105 / 0.045 / 0.025), so it measures the reference-pool size, not the model |
| `edm_tikhonov_memorization.ipynb`, U-Net columns | `sigma_max=80`, 500 SDE steps, `exclude_nn=False`, none of which match the locked conventions. Source of the current draft's Table 1 |
| `edm_unet_train_size_sweep.ipynb` | `sigma_max=80`, superseded by section 5 |
| `edm_unet_train_size_sweep_sigma_fix.ipynb` | The A/B/C/D sampler ablation. Null, and null by construction: both `sigma_max` values tested sit at 0.05 and 0.55 of `D_min`, that is, the same regime. Superseded by pending experiment 2 |
| `results/figures/tikvariants_isotropic_vs_covariance.png` | Orphan. No notebook or script on any branch writes this filename. Still shows the floored coarse curve. Was attached to the email |
| `results/data/sigma_spacing_sweep.pt` | Produced by `src/spectral_reference.py` on the discarded `wip-branch`. That branch **does still exist**, locally and at `origin/wip-branch` (both at `02960b9`, as does `wip-archive`), and it does still contain `src/spectral_reference.py`, so the artifact is in principle reproducible; it is excluded because that branch's src was rejected, not because it is lost. Its *design* is sound and informs pending experiment 2. Note it occupies the exact filename pending experiment 2 will write |
| `edm_unet_covariance_tikhonov.ipynb` | Never run: 10 cells, 0 outputs, none of its three output files exist. See section 10 |
| `edm_unet_memorization_mechanism.ipynb` | 0 outputs **and all five of its input artifacts are absent from disk** (`edm_unet_denoiser_gap.pt`, `edm_unet_basin_reconstruction.pt`, `edm_unet_coordconv.pt`, `edm_unet_sigma_fix_ablation.pt`, `gmm_covariance_tikhonov_sweep.pt`), so it renders nothing. The denoiser-gap and basin-reachability results attributed to it trace to `src/scripts/denoiser_gap.py` and `basin_reconstruction.py` plus figures dated 2026-07-24. Needs re-running before citation. This is the most substantive unpublished result in the repo and it currently exists only as three PNGs and three scripts |
| `analyze_multiband_memorization_alt_weights.ipynb` | Deliberately a different metric (symmetric ring error, coarse-conditional reference pool, `n_ref=64`, ring step 2.0, NaN-masking). Not drift, but not comparable |
| `analyze_multiband_memorization.ipynb` | VP diffusion, `n_rand_ref=64`, `exclude_nn=False` at n_train=32, so its coarse 0.026 sits on the 1/32 floor. Its *exact-match* statistics (50 percent of samples reproduce a training field's coarse band, median coarse NN distance 0.0) are floor-free and do survive |
| `train_vp_sde_on_biased_multiband_data.ipynb`, `train_deepinv_on_biased_multiband_data.ipynb` | Both crashed on a NumPy 1.x/2.x ABI break; committed outputs are stack traces. No results |
| `results/figures/tikvariants_*.FLOORED-BACKUP.png` (4 files) and `results/data/gmm_tikhonov_variants_sweep.FLOORED-BACKUP.pt` | Deliberate pre-correction backups of the `exclude_nn=False` run, kept for comparison. Superseded by section 4. Never cite; delete once section 4 is in the paper |

Twelve `.pt` artifacts referenced by notebooks are not present on disk. `results/data/` is
gitignored (`.gitignore:42`), so none of them were ever under version control. Affected:
`edm_tikhonov_sweep.pt`, `edm_unet_train_size_sweep.pt`, `edm_unet_ntrain_checkpoints.pt`,
`edm_unet_sigma_fix_ablation.pt`, `gmm_covariance_tikhonov_sweep.pt`,
`edm_unet_denoiser_gap.pt`, `edm_unet_basin_reconstruction.pt`, `edm_unet_coordconv.pt`,
`edm_unet_covariance_tikhonov.pt`, `multiband_dataset_unbiased_and_biased.pt`, and the two
cluster-run Gaussian-field files. Most of the affected notebooks still carry committed cell
outputs, so their numbers are readable even though they cannot be recomputed without
retraining. The two that carry neither outputs nor data are
`edm_unet_memorization_mechanism.ipynb` and `edm_unet_covariance_tikhonov.ipynb`.

## 11. Prof. Baptista's three asks: all complete

| # | ask, verbatim | notebook | outcome |
|---|---|---|---|
| 1 | *"worth rerunning at batch size 2 to match before concluding there is no memorization"* | `unet_update_budget_batchsize.ipynb` | **done**, section 7. No memorization at matched update count. The `bs8` comparison arm is a relabelled earlier run, not new work |
| 2 | *"the sigma_max/spacing mismatch with the paper is worth matching (or at least testing these spacings) if its easy since it could shift the point the model collapses to"* | `sigma_spacing_sweep.ipynb` | **done**, section 9. Ruled out across `sigma_max/D_min` 0.05 to 2.73 at both 40 and 1000 steps. Zero collapse in all 12 cells |
| 3 | *"hold off on Q3 conclusions until the rerun with multiple seeds"* | `capacity_multiseed.ipynb` | **done**, section 6. The capacity ordering does **not** survive. A depth effect at matched parameter count does. Also supplied the seed spread the rest of this document now uses |

Across sections 5 through 9, no configuration of training budget, batch size, sampler range,
sampler discretization, parameter count, architecture depth or training-set size produced any
pixel collapse. The closed-form empirical score gives collapse 1.00 through the identical
pipeline in every one of those blocks. That is the negative result, and it is threshold-free.

What it does not establish is *why*, and no section here should be read as diagnosing a
mechanism. The diagnostics in `edm_unet_memorization_mechanism.ipynb` are the closest thing
to an explanation the project has, and they currently have no runnable artifacts (section 10).

One item is not from the email but is blocked by these results.
`edm_unet_covariance_tikhonov.ipynb`, the training-time analogue of Result B, was designed to
test whether a network trained with the covariance penalty inherits the closed form's
scale-selectivity. Its stated precondition is a training regime where the *unregularized*
baseline memorizes. Sections 5 to 9 establish that no such regime was found in any of the
ranges explored, so the experiment remains uninterpretable. Section 6c suggests the direction
worth trying if it is revived: deeper networks, not larger ones.

