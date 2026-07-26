"""Device selection helper.

Notebooks call resolve_device() once in their config cell:

    DEVICE = resolve_device()        # picks cuda > mps > cpu automatically
    DEVICE = resolve_device("cpu")   # one-line override for the personal laptop

Keeping this in one place means every new notebook switches device the same way.
"""

import torch


def resolve_device(override=None):
    """Return a torch.device: the override if given, else cuda > mps > cpu.

    Args:
        override: optional device string ("cpu", "mps", "cuda") to force a
            device regardless of availability checks.
    """
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
