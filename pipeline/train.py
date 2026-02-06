"""
Model training module for SpecLens-PML.

This script implements the supervised learning stage of the pipeline:

- Loads the generated dataset artifact (CSV).
- Splits data into training and evaluation sets.
- Trains a baseline Random Forest classifier.
- Reports standard ML metrics (accuracy and recall on risky functions).
- Saves a new versioned model artifact (model_vN.pkl).

The training process is fully reproducible and configuration-driven,
with hyperparameters centralized in config.yaml.

Each execution produces a new model version to support traceability,
rollback, and controlled deployment workflows.
"""

#!/usr/bin/env python3

import sys
import re
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def next_model_path(models_dir: Path) -> Path:
    """
    Compute the next available versioned model artifact path.

    Models are stored under the models/ directory using the naming scheme:

        model_v1.pkl
        model_v2.pkl
        ...

    This function scans existing artifacts, extracts the highest version
    number, and returns the next version path.

    :param models_dir: Directory containing stored model artifacts.
    :return: Path for the next model version to be saved.
    """
    # Retrieve all previously saved model artifacts
    existing = list(models_dir.glob("model_v*.pkl"))

    # If no model exists yet, start from version 1
    if not existing:
        return models_dir / "model_v1.pkl"

    versions = []

    # Extract numeric version identifiers from filenames
    for p in existing:
        m = re.search(r"model_v(\d+)\.pkl", p.name)
        if m:
            versions.append(int(m.group(1)))

    # Compute the next version number
    next_v = max(versions) + 1 if versions else 1

    return models_dir / f"model_v{next_v}.pkl"


def main():
    """
    Execute the training pipeline step.

    This function performs:

    1. Dataset loading
    2. Feature/label extraction
    3. Train/test split
    4. Model training
    5. Evaluation reporting
    6. Versioned artifact persistence

    The resulting model is stored in models/model_vN.pkl.
    """
    # Validate command-line usage
    if len(sys.argv) < 2:
        print("Usage: train.py <dataset.csv>")
        sys.exit(1)

    # Resolve dataset artifact path
    dataset_path = Path(sys.argv[1]).resolve()

    # Identify project root directory
    ROOT = Path(__file__).resolve().parents[1]

    # Load centralized configuration file
    CONFIG_PATH = ROOT / "config.yaml"
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Extract model hyperparameters from configuration
    model_cfg = config["model"]

    # --- Step 1: Load dataset artifact -------------------------------------
    df = pd.read_csv(dataset_path)

    # Define feature columns used for training and inference consistency
    FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]

    # Extract input feature matrix
    X = df[FEATURE_COLS]

    # Extract supervised target labels (SAFE vs RISKY)
    y = df["label"]

    # --- Step 2: Train/test split ------------------------------------------
    # Stratification is applied when possible to preserve label distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    # --- Step 3: Model initialization --------------------------------------
    # Random Forest is used as a robust baseline classifier
    model = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        class_weight=model_cfg.get("class_weight", None),
        random_state=42,
    )

    # --- Step 4: Model training --------------------------------------------
    model.fit(X_train, y_train)

    # --- Step 5: Evaluation ------------------------------------------------
    # Generate predictions on the held-out test set
    y_pred = model.predict(X_test)

    # Compute key operational metrics
    acc = accuracy_score(y_test, y_pred)

    # Recall on the risky class is safety-oriented: missing violations is costly
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Print evaluation report for transparency
    print("=== Evaluation Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.3f}")
    print(f"Recall (RISKY): {rec:.3f}")

    # --- Step 6: Model versioning and persistence --------------------------
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # Compute next artifact version path
    model_path = next_model_path(models_dir)

    # Save trained model as a versioned artifact
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


# Entry point for standalone execution
if __name__ == "__main__":
    main()
