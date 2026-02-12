"""
Continuous Training Trigger for SpecLens-PML.

This module implements a simple Champion / Challenger governance strategy:

- Multiple candidate models are trained (logistic, forest)
- Each candidate is evaluated on the same held-out TEST dataset
- The model with the best recall on the RISKY class is promoted

Promotion occurs only if the RISKY-class recall is above a configurable minimum threshold.

The promoted champion model is always saved as:

    models/best_model.pkl
"""

from pathlib import Path
from pipeline.train import evaluate_model

import joblib
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------

# Directory containing all trained candidate and champion model artifacts
MODELS_DIR = Path("models")

# Candidate model registry (baseline + challenger)
CANDIDATES = {
    "logistic": MODELS_DIR / "logistic.pkl",
    "forest": MODELS_DIR / "forest.pkl",
}

# Path where the promoted champion model is stored
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Central configuration file defining MLOps governance rules
CONFIG_PATH = Path("config.yaml")


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """
    Load continuous training governance rules from ``config.yaml``.

    Returns
    -------
    dict
        Parsed YAML configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file is missing.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config.yaml")

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Continuous Training Trigger Logic
# ---------------------------------------------------------------------------

def main(test_dataset: Path) -> None:
    """
    Execute the Continuous Training Trigger.

    This function evaluates all available candidate models on the held-out
    TEST dataset and promotes the best-performing one according to the
    safety-oriented metric: recall on the RISKY class.

    Parameters
    ----------
    test_dataset : Path
        Path to the held-out TEST dataset CSV used for candidate evaluation.

    Raises
    ------
    FileNotFoundError
        If the TEST dataset file does not exist.
    """
    print("=== Continuous Training Trigger ===")

    cfg = load_config()

    governance = cfg.get("mlops", {})
    min_recall = governance.get("min_recall_risky", 0.0)

    print(f"Minimum recall on the RISKY class required for promotion: {min_recall:.3f}")

    # -----------------------------------------------------------------------
    # Load TEST dataset (held-out evaluation set)
    # -----------------------------------------------------------------------

    if not test_dataset.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset}")

    df = pd.read_csv(test_dataset)

    feature_cols = [
        c for c in df.columns
        if c not in ("name", "class", "source_file", "label")
    ]

    X_test = df[feature_cols]
    y_test = df["label"]

    best_name = None
    best_recall = -1.0
    best_model = None

    # -----------------------------------------------------------------------
    # Evaluate all candidate models
    # -----------------------------------------------------------------------

    for name, path in CANDIDATES.items():

        if not path.exists():
            print(f"Skipping {name}: model file not found.")
            continue

        print(f"\nEvaluating candidate: {name}")

        model = joblib.load(path)

        recall_risky = evaluate_model(model, X_test, y_test)

        print(f"Recall on the RISKY class for {name}: {recall_risky:.3f}")

        if recall_risky > best_recall:
            best_recall = recall_risky
            best_name = name
            best_model = model

    # -----------------------------------------------------------------------
    # Promotion Decision
    # -----------------------------------------------------------------------

    if best_model is None:
        print("\nNo valid candidate models found.")
        return

    print("\n=== Promotion Decision ===")
    print(f"Best candidate: {best_name}")
    print(f"Best recall on the RISKY class: {best_recall:.3f}")

    if best_recall < min_recall:
        print("\nPromotion blocked: minimum safety threshold not met.")
        print("No model was promoted.")
        return

    # Promote champion model
    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n=== Promotion Result ===")
    print(f"Champion model promoted: {BEST_MODEL_PATH}")
    print(f"Active champion: {best_name}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage: python ct_trigger.py <datasets_test.csv>")
        sys.exit(1)

    main(Path(sys.argv[1]))

