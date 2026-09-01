"""Validate the release structure without loading serialized models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "data/model_inputs/CH4_model_input.xlsx",
    "data/model_inputs/N2O_model_input.xlsx",
    "data/scenario_inputs/province_secondary_effluent.xlsx",
    "data/scenario_inputs/statutory_effluent_scenarios.xlsx",
    "models/CH4_XGBoost_model.joblib",
    "models/CH4_XGBoost_metadata.json",
    "models/N2O_ET_model.joblib",
    "models/N2O_ET_metadata.json",
]

EXPECTED_HASHES = {
    "models/CH4_XGBoost_model.joblib": "56933847984294939fb2037337b5c05617cf6d380f09accfa6e36e11084433fe",
    "models/N2O_ET_model.joblib": "64bfb6b57b7e3d5c04ba329690bda30076959c62d70c80daf09eb98b8cace941",
}

EXPECTED_SCENARIOS = ["Class I-A", "Class I-B", "Class II", "Class III"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    ch4 = pd.read_excel(ROOT / "data/model_inputs/CH4_model_input.xlsx")
    n2o = pd.read_excel(ROOT / "data/model_inputs/N2O_model_input.xlsx")
    provinces = pd.read_excel(ROOT / "data/scenario_inputs/province_secondary_effluent.xlsx")
    scenarios = pd.read_excel(ROOT / "data/scenario_inputs/statutory_effluent_scenarios.xlsx")

    assert ch4.shape[0] == 243 and "CH4" in ch4.columns
    assert n2o.shape[0] == 295 and "N2O" in n2o.columns
    assert provinces.shape[0] == 31 and "Province" in provinces.columns
    assert scenarios["Province"].tolist() == EXPECTED_SCENARIOS

    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        assert actual == expected, f"Model hash mismatch: {relative}"

    for gas, metadata_name, expected_target in [
        ("CH4", "models/CH4_XGBoost_metadata.json", "CH4"),
        ("N2O", "models/N2O_ET_metadata.json", "N2O"),
    ]:
        with (ROOT / metadata_name).open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["target"] == expected_target
        assert metadata["split"] == {"train": 0.7, "test": 0.3}
        assert metadata["cv"] == 10
        print(f"{gas}: {len(metadata['selected_features'])} selected features; test R2={metadata['metrics']['test_R2']:.3f}")

    print("Repository validation passed.")


if __name__ == "__main__":
    main()

