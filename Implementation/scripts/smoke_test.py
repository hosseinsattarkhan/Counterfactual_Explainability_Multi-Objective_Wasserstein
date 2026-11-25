from pathlib import Path

root = Path(__file__).resolve().parents[2]
required = [
    "Implementation/src/counterfactuals/prototype.py",
    "Implementation/src/algorithms/exact_radius_v9.py",
    "Implementation/tests/test_smoke.py",
]
for name in required:
    assert (root / name).is_file(), name
print("PASS: repository layout check (does not test optimization solvers)")
