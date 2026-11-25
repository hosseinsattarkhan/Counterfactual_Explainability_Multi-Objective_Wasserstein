from pathlib import Path


def test_repository_layout():
    root = Path(__file__).resolve().parents[2]
    assert (root / "Implementation" / "src" / "counterfactuals" / "prototype.py").exists()
