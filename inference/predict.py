#!/usr/bin/env python3

"""
Inference and risk scoring module for SpecLens-PML.

This script represents the serving stage of the pipeline:

- It loads the currently active trained model artifact.
- It parses a new Python source file annotated with PML contracts.
- It extracts the same feature representation used during training.
- It produces a probabilistic risk score for each function or method.
- It maps the score into operational decision levels (LOW / MEDIUM / HIGH)
  using thresholds defined in the centralized configuration file.

This module provides decision support rather than formal guarantees.
"""

from pathlib import Path
from pml.parser import parse_file

import joblib
import pandas as pd
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    """
    Load the centralized system configuration from config.yaml.

    This configuration defines operational thresholds and model settings
    used consistently across training and inference.

    :return: Parsed configuration dictionary.
    """
    # Define the location of the YAML configuration file
    CONFIG_PATH = ROOT / "config.yaml"

    # Load and parse the YAML file into a Python dictionary
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def risk_level(score: float, low: float, medium: float) -> str:
    """
    Convert a numeric risk score into an operational decision level.

    The mapping is based on two thresholds:

    - below low threshold    → LOW risk
    - between low and medium → MEDIUM risk
    - above medium threshold → HIGH risk

    :param score: Predicted probability of contract violation.
    :param low: Threshold for LOW risk.
    :param medium: Threshold for MEDIUM risk.
    :return: Risk level string (LOW, MEDIUM, HIGH).
    """
    # Apply threshold-based categorization of the risk score
    if score < low:
        return "LOW"
    elif score < medium:
        return "MEDIUM"
    else:
        return "HIGH"


def main():
    """
    Run inference on a Python source file annotated with PML.

    This function loads the active model, parses the source code,
    extracts feature vectors, and prints risk predictions for each unit.
    """
    # Validate command-line arguments
    if len(sys.argv) < 2:
        print("Usage: predict.py <source.py>")
        sys.exit(1)

    # Resolve the path of the Python source file to analyze
    source_path = Path(sys.argv[1]).resolve()

    # --- Step 1: Load operational thresholds from configuration -------------
    config = load_config()
    thr = config["risk_thresholds"]
    LOW = thr["low"]
    MEDIUM = thr["medium"]

    # --- Step 2: Load the currently active model artifact ------------------
    # The active model is defined through a pointer file for controlled serving
    ACTIVE_PATH = ROOT / "models" / "active_model.txt"
    with open(ACTIVE_PATH) as f:
        model_path = ROOT / f.read().strip()

    # Load the trained classifier from disk
    model = joblib.load(model_path)

    # --- Step 3: Parse the input source file -------------------------------
    # Extract functions and methods annotated with PML contracts
    units = parse_file(source_path)

    # Print header information for traceability
    print(f"Analysis of {source_path.name}")
    print(f"(active model: {model_path.name})\n")

    # Feature columns expected by the trained model
    FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]

    # --- Step 4: Predict risk for each parsed unit -------------------------
    for u in units:
        # Normalize invariant field naming across parsing outputs
        invariants = u.get("invariants", u.get("invariant", []))

        # Build the feature vector used during model training
        features = {
            "n_params": len(u.get("params", [])),
            "n_requires": len(u.get("requires", [])),
            "n_ensures": len(u.get("ensures", [])),
            "n_invariants": len(invariants),
            "n_loc": u.get("loc", 0),
        }

        # Convert features into a single-row DataFrame for sklearn compatibility
        X = pd.DataFrame([features], columns=FEATURE_COLS)

        # Predict probability of belonging to the RISKY class
        score = float(model.predict_proba(X)[0][1])

        # Map probability into an operational decision level
        level = risk_level(score, LOW, MEDIUM)

        # Print contract information and the associated risk prediction
        print(f"- {u['name']} (line {u['lineno']})")
        print(f"  requires: {u.get('requires', [])}")
        print(f"  ensures:  {u.get('ensures', [])}")
        print(f"  invariant:{invariants}")
        print(f"  → risk score: {score:.3f} [{level}]\n")


# Entry point for standalone execution
if __name__ == "__main__":
    main()
