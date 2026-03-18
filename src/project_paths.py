from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for path in (candidate, *candidate.parents):
        if (path / "src").exists() and (path / "notebooks").exists():
            return path

    raise RuntimeError("Could not locate repository root containing src/ and notebooks/.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DATA_DIR = RESULTS_DIR / "data"
