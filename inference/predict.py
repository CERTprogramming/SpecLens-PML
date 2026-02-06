#!/usr/bin/env python3

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import yaml
import joblib
import pandas as pd

from pml.parser import parse_file


def load_config():
    CONFIG_PATH = ROOT / "config.yaml"
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def risk_level(score: float, low: float, medium: float) -> str:
    if score < low:
        return "LOW"
    elif score < medium:
        return "MEDIUM"
    else:
        return "HIGH"


def main():
    if len(sys.argv) < 2:
        print("Usage: predict.py <source.py>")
        sys.exit(1)

    source_path = Path(sys.argv[1]).resolve()

    config = load_config()
    thr = config["risk_thresholds"]
    LOW = thr["low"]
    MEDIUM = thr["medium"]

    ACTIVE_PATH = ROOT / "models" / "active_model.txt"
    with open(ACTIVE_PATH) as f:
        model_path = ROOT / f.read().strip()

    model = joblib.load(model_path)

    units = parse_file(source_path)

    print(f"Analysis of {source_path.name}")
    print(f"(active model: {model_path.name})\n")

    FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]

    for u in units:
        invariants = u.get("invariants", u.get("invariant", []))

        features = {
            "n_params": len(u.get("params", [])),
            "n_requires": len(u.get("requires", [])),
            "n_ensures": len(u.get("ensures", [])),
            "n_invariants": len(invariants),
            "n_loc": u.get("loc", 0),
        }

        X = pd.DataFrame([features], columns=FEATURE_COLS)

        score = float(model.predict_proba(X)[0][1])
        level = risk_level(score, LOW, MEDIUM)

        print(f"- {u['name']} (line {u['lineno']})")
        print(f"  requires: {u.get('requires', [])}")
        print(f"  ensures:  {u.get('ensures', [])}")
        print(f"  invariant:{invariants}")
        print(f"  → risk score: {score:.3f} [{level}]\n")


if __name__ == "__main__":
    main()
