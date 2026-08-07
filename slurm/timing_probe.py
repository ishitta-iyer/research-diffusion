"""Measure the real cost of one arm on this GPU and extrapolate the full run.

Run this inside a short interactive allocation BEFORE submitting the real job, so the
--time request is based on measurement instead of a guess:

    python3 slurm/timing_probe.py            # all six arms
    python3 slurm/timing_probe.py 128        # just the expensive one

Training is batch-size 1, so it is dominated by kernel-launch latency rather than arithmetic;
the small arms usually all cost about the same. Sampling is the part that scales with model
size, and across a full run it costs roughly as much as training does.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "notebooks/multiscale/baptista_rectangles_n2_reproduction.ipynb"

# Load the notebook's definitions (everything before the config cell).
os.chdir(NB.parent)
cells = [
    "".join(c["source"])
    for c in json.loads(NB.read_text())["cells"]
    if c["cell_type"] == "code"
]
G = {"__name__": "__main__"}
_tmp = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "rect_probe"
_tmp.mkdir(parents=True, exist_ok=True)
for src in cells:
    if src.lstrip().startswith("SMOKE ="):
        break
    exec(compile(src, "<nb>", "exec"), G)
    if "resolve_device" in src:
        # this probe only measures speed -- keep its plots out of results/
        G["fig_dir"] = G["results_dir"] = str(_tmp)

build_net, EDMLoss, DEVICE, data = G["build_net"], G["EDMLoss"], G["DEVICE"], G["data"]

# The protocol the real run uses.
EPOCHS, UPDATES_PER_EPOCH = 50_000, 2
EVAL_EVERY, N_EVAL, N_STEPS, EVAL_BATCH = 1_000, 100, 40, 50
N_EVALS = EPOCHS // EVAL_EVERY
NFE_PER_SAMPLE = 2 * N_STEPS - 1            # Heun: 2 evals per step, 1 fewer on the last
BATCHES = -(-N_EVAL // EVAL_BATCH)          # ceil


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


arms = [int(a) for a in sys.argv[1:]] or [4, 8, 16, 32, 64, 128]
print(f"device: {DEVICE}"
      + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
print(f"protocol: {EPOCHS:,} epochs x {UPDATES_PER_EPOCH} updates, "
      f"{N_EVALS} evals x {N_EVAL} samples x {NFE_PER_SAMPLE} NFE\n")
print(f"{'C':>5} {'params':>13} {'train ms':>9} {'nfe ms':>8} "
      f"{'train h':>8} {'sample h':>9} {'TOTAL h':>8}")
print("-" * 68)

grand = 0.0
per_arm = {}
for C in arms:
    net = build_net(C, DEVICE)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    lf, x = EDMLoss(), data[:1].to(DEVICE)

    def one_update():
        """Everything main.py does per update, including the EMA -- which is a Python loop
        over every parameter tensor and is a real fraction of the cost at these batch sizes."""
        opt.zero_grad(set_to_none=True)
        (lf(net, x).sum() / 1).backward()
        opt.step()
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, 0.999))

    net.train()
    for _ in range(5):                                   # warm up / autotune cudnn
        one_update()
    sync()
    t0, n = time.time(), 25
    for _ in range(n):
        one_update()
    sync()
    train_ms = (time.time() - t0) / n * 1000

    net.eval()
    with torch.no_grad():
        xb = torch.randn(EVAL_BATCH, 1, 64, 64, device=DEVICE)
        sg = torch.full((EVAL_BATCH,), 1.0, device=DEVICE)
        for _ in range(3):
            net(xb, sg)
        sync()
        t0, m = time.time(), 10
        for _ in range(m):
            net(xb, sg)
        sync()
        nfe_ms = (time.time() - t0) / m * 1000

    train_h = train_ms * EPOCHS * UPDATES_PER_EPOCH / 1000 / 3600
    sample_h = nfe_ms * N_EVALS * BATCHES * NFE_PER_SAMPLE / 1000 / 3600
    total = train_h + sample_h
    grand += total
    per_arm[C] = total
    print(f"{C:>5} {sum(p.numel() for p in net.parameters()):>13,} {train_ms:>9.1f} "
          f"{nfe_ms:>8.1f} {train_h:>8.2f} {sample_h:>9.2f} {total:>8.2f}")
    del net, ema, opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

print("-" * 68)
print(f"serial (one job, all arms) : {grand:>5.1f} h wall  "
      f"-> --time={int(grand * 1.5) + 1}:00:00  (50% headroom)")
if len(arms) > 1:
    slowest = max(per_arm, key=per_arm.get)
    print(f"array (one arm per GPU)    : {per_arm[slowest]:>5.1f} h wall  "
          f"-> --time={int(per_arm[slowest] * 1.5) + 1}:00:00  "
          f"(slowest arm is C={slowest}; {grand:.1f} h of GPU time total)")
print("\nResume is per-arm, so an under-estimate costs a resubmit, not the run.")
