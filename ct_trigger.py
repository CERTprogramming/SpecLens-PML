"""
Continuous Training Trigger for SpecLens-PML.

Implements a Champion/Challenger strategy:

- Multiple candidate models are trained (logistic, forest).
- Each candidate is evaluated on the same held-out TEST dataset.
- The model with best Recall on the RISKY class is promoted.

The promoted model is always saved as:

    models/best_model.pkl

This ensures inference always uses the best available model.
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

from pipeline.train import evaluate_model


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODELS_DIR = Path("models")

CANDIDATES = {
    "logistic": MODELS_DIR / "logistic.pkl",
    "forest": MODELS_DIR / "forest.pkl",
}

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


# ------------------------------------------------------------
# Trigger logic
# ------------------------------------------------------------

def main(test_dataset_path: Path):
    print("=== Continuous Training Trigger ===")

    # --------------------------------------------------------
    # Load TEST dataset (held-out evaluation set)
    # --------------------------------------------------------
    if not test_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

    df = pd.read_csv(test_dataset_path)

    feature_cols = [
        c for c in df.columns
        if c not in ("name", "class", "source_file", "label")
    ]

    X_test = df[feature_cols]
    y_test = df["label"]

    best_name = None
    best_recall = -1.0
    best_model = None

    # --------------------------------------------------------
    # Evaluate all candidate models
    # --------------------------------------------------------
    for name, path in CANDIDATES.items():

        if not path.exists():
            print(f"Skipping {name}: model file not found.")
            continue

        print(f"\nEvaluating candidate: {name}")

        model = joblib.load(path)

        recall_risky = evaluate_model(model, X_test, y_test)

        print(f"Recall (RISKY) for {name}: {recall_risky:.3f}")

        if recall_risky > best_recall:
            best_recall = recall_risky
            best_name = name
            best_model = model

    # --------------------------------------------------------
    # Promote best model
    # --------------------------------------------------------
    if best_model is None:
        print("\nNo valid candidate models found.")
        return

    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n=== Promotion Result ===")
    print(f"Best model: {best_name}")
    print(f"Best Recall (RISKY): {best_recall:.3f}")
    print(f"Promoted as: {BEST_MODEL_PATH}")


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ct_trigger.py <test_dataset.csv>")
        sys.exit(1)

    test_path = Path(sys.argv[1])
    main(test_path)

