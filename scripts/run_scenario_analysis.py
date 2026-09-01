"""Run deterministic statutory predictions and residual-based uncertainty."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    scripts = Path(__file__).resolve().parent
    sequence = [
        scripts / "07_statutory_scenarios" / "predict_CH4_scenarios.py",
        scripts / "07_statutory_scenarios" / "predict_N2O_scenarios.py",
        scripts / "08_uncertainty_analysis" / "export_test_residuals.py",
        scripts / "08_uncertainty_analysis" / "residual_monte_carlo.py",
    ]
    for script in sequence:
        print(f"\nRunning {script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()

