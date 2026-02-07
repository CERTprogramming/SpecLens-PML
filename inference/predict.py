"""
SpecLens-PML Inference Script.

This module is part of the SpecLens demo project:
it performs inference on annotated Python functions.

The script loads the promoted production model:

    models/best_model.pkl

Then, for every function annotated with PML contracts in the input file,
it extracts features and predicts a risk score (SAFE vs RISKY).

Feature extraction is shared with training via pipeline/features.py,
ensuring schema consistency across the pipeline.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from pml.parser import parse_file
from pipeline.features import extract_features


# ---------------------------------------------------------------------------
# Load promoted best model
# ---------------------------------------------------------------------------

BEST_MODEL_PATH = Path("models/best_model.pkl")

if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(
        "No promoted model found.\n"
        "Run the pipeline first:\n"
        "   python demo.py"
    )

model = joblib.load(BEST_MODEL_PATH)


# ---------------------------------------------------------------------------
# Risk interpretation helper
# ---------------------------------------------------------------------------

def risk_level(score: float) -> str:
    """
    Convert a probability score into a human-readable risk level.
    """
    if score > 0.7:
        return "HIGH"
    if score > 0.3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def predict_file(path: Path):
    """
    Run inference on all annotated functions inside a Python source file.
    """
    print(f"Analysis of {path.name}")
    print(f"(active model: {BEST_MODEL_PATH.name})\n")

    # Parse annotated functions from the source file
    functions = parse_file(path)

    # Load dataset reference to ensure feature schema alignment
    dataset = pd.read_csv("data/datasets_v1.csv")

    feature_cols = [
        c for c in dataset.columns
        if c not in ("name", "class", "source_file", "label")
    ]

    # --------------------------------------------------------
    # Predict risk score for each function
    # --------------------------------------------------------
    for f in functions:

        # Skip dunder methods (e.g., __init__)
        if f["name"].startswith("__") and f["name"].endswith("__"):
            continue

        # Extract numeric features from the parsed contract data
        feats = extract_features(f)

        # Align feature vector with the training schema
        row = {col: 0 for col in feature_cols}
        row.update(feats)

        X = pd.DataFrame([row])

        # Predict probability of the RISKY class
        prob_risky = model.predict_proba(X)[0][1]
        level = risk_level(prob_risky)

        # Print structured report
        print(f"- {f['name']} (line {f['line']})")
        print(f"  requires: {f['requires']}")
        print(f"  ensures:  {f['ensures']}")
        print(f"  invariant:{f['invariant']}")
        print(f"  → risk score: {prob_risky:.3f} [{level}]\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <python_file>")
        sys.exit(1)

    predict_file(Path(sys.argv[1]))

