"""Run all 13 model-comparison scripts for CH4 or N2O."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ALGORITHMS = [
    "linear.py",
    "ridge.py",
    "lasso.py",
    "svr.py",
    "knn.py",
    "decision_tree.py",
    "random_forest.py",
    "extra_trees.py",
    "gradient_boosting.py",
    "xgboost.py",
    "lightgbm.py",
    "catboost.py",
    "ann.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gas", required=True, choices=["CH4", "N2O"])
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "02_model_comparison" / args.gas
    for name in ALGORITHMS:
        script = base / name
        print(f"\nRunning {args.gas}: {name}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()

