from torchvision import datasets
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "results" / "data" / "mnist"
data_dir.mkdir(parents=True, exist_ok=True)

try:
    print("Attempting to download MNIST...")
    dataset = datasets.MNIST(root=str(data_dir), train=True, download=True)
    print("MNIST successfully downloaded!")
except Exception as e:
    print(f"Error downloading MNIST: {e}")
