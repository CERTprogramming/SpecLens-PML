#!/usr/bin/env python3

"""
Continuous training trigger for SpecLens-PML.

This module implements a simplified Continuous Training (CT) loop:

- It retrains a new model using the current dataset.
- It evaluates both the active model and the newly trained model.
- It promotes the new model only if it improves a safety-oriented metric
  (recall on the RISKY class).

This reflects a realistic MLOps governance pattern where training and
deployment are decoupled, and model updates are controlled through
explicit promotion rules.
"""

from pathlib import Path
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import subprocess

# Define repository root and key artifact locations
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ACTIVE = MODELS / "active_model.txt"
DATASET = ROOT / "data" / "datasets_v1.csv"

# Feature columns used consistently across training and evaluation
FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]

def evaluate(model_path: Path) -> float:
    """
    Evaluate a trained model on the current dataset.

    The evaluation focuses on recall for the RISKY class, which is treated
    as the safety-critical metric in this project.

    :param model_path: Path to the model artifact (.pkl file).
    :return: Recall score on the test split.
    """
    # Load the latest dataset produced by the data pipeline
    df = pd.read_csv(DATASET)

    # Extract feature matrix and target labels
    X = df[FEATURE_COLS]
    y = df["label"]

    # Split data into training and test partitions
    # Stratification is applied when both classes are present
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    # Load the trained model artifact from disk
    model = joblib.load(model_path)

    # Generate predictions on the test set
    ypred = model.predict(Xte)

    # Compute recall, ensuring stable behavior even with edge cases
    return recall_score(yte, ypred, zero_division=0)

def main():
    """
    Execute the continuous training trigger.

    This function retrains a new model, compares it with the currently
    active one, and updates the active model pointer only if performance
    improves.
    """
    print("=== Continuous Training Trigger ===")

    # --- Step 1: Train a new candidate model -------------------------------
    # This produces a new versioned artifact in the models/ directory
    subprocess.call(["python3", "pipeline/train.py", str(DATASET)])

    # --- Step 2: Identify the newly produced model -------------------------
    # Select the latest model based on modification timestamp
    latest = max(MODELS.glob("model_v*.pkl"), key=lambda p: p.stat().st_mtime)

    # --- Step 3: Load the currently active production model ----------------
    # The active model is defined by a pointer file for controlled deployment
    with open(ACTIVE) as f:
        current = ROOT / f.read().strip()

    # --- Step 4: Evaluate both models --------------------------------------
    # Compare recall on the RISKY class for governance decisions
    new_recall = evaluate(latest)
    cur_recall = evaluate(current)

    # Print evaluation results for traceability and monitoring
    print(f"Current recall (RISKY): {cur_recall:.3f}")
    print(f"New recall (RISKY):     {new_recall:.3f}")

    # --- Step 5: Model promotion decision ----------------------------------
    # Promote the new model only if it improves the safety-oriented metric
    if new_recall >= cur_recall:
        # Update the active model pointer to deploy the new artifact
        with open(ACTIVE, "w") as f:
            f.write(str(latest.relative_to(ROOT)))
        print(f"New model promoted: {latest.name}")
    else:
        # Keep the current model in production if no improvement is observed
        print("New model rejected (performance did not improve).")

# Entry point for standalone execution
if __name__ == "__main__":
    main()
