"""
Model training module for SpecLens-PML.

This script implements the supervised learning stage of the pipeline.

It supports multiple candidate models (Champion/Challenger setup):

- Logistic Regression (baseline candidate)
- Random Forest (challenger candidate)

Hyperparameters are loaded from the central configuration file
``config.yaml``.

Each trained candidate model is saved separately under ``models/``:

    - models/logistic.pkl
    - models/forest.pkl

Continuous Training later evaluates all candidates on the held-out TEST set
and promotes the best-performing model based on **Recall on the RISKY class**.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# Configuration Loader
# ------------------------------------------------------------

CONFIG_PATH = Path("config.yaml")


def load_config():
    """
    Load the central YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary containing model parameters
        and governance policies.

    Raises
    ------
    FileNotFoundError
        If ``config.yaml`` is missing from the repository root.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing config.yaml")

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# Model Factory
# ------------------------------------------------------------

def build_model(model_type: str, cfg: dict):
    """
    Construct a candidate classifier given its type and YAML configuration.

    Parameters
    ----------
    model_type : str
        Candidate model identifier (``logistic`` or ``forest``).

    cfg : dict
        Configuration dictionary loaded from ``config.yaml``.

    Returns
    -------
    sklearn.base.BaseEstimator
        Instantiated scikit-learn classifier.

    Raises
    ------
    ValueError
        If the requested model type is not supported.
    """

    models_cfg = cfg.get("models", {})

    if model_type == "logistic":
        params = models_cfg.get("logistic", {})
        return LogisticRegression(
            max_iter=params.get("max_iter", 1000)
        )

    if model_type == "forest":
        params = models_cfg.get("forest", {})
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        )

    raise ValueError(f"Unknown model type: {model_type}")


# ------------------------------------------------------------
# Evaluation Helper
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on a validation or TEST dataset.

    The primary governance metric for SpecLens-PML is:

        Recall on the RISKY class (label = 1)

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Trained classifier.

    X_test : pandas.DataFrame
        Feature matrix.

    y_test : pandas.Series
        Ground-truth labels (0 = SAFE, 1 = RISKY).

    Returns
    -------
    float
        Recall score for the RISKY class.
    """

    y_pred = model.predict(X_test)

    print("\n=== Evaluation Report ===")
    print(classification_report(y_test, y_pred))

    recall_risky = recall_score(y_test, y_pred, pos_label=1)
    print(f"Recall (RISKY): {recall_risky:.3f}")

    return recall_risky


# ------------------------------------------------------------
# Training Procedure
# ------------------------------------------------------------

def train(dataset_path: Path, model_type: str):
    """
    Train a candidate model on the generated TRAIN dataset.

    This procedure:

    1. Loads the dataset artifact (datasets_train.csv)
    2. Splits it into training/validation subsets (diagnostic only)
    3. Trains the selected candidate model family
    4. Saves the trained artifact under ``models/``

    Parameters
    ----------
    dataset_path : Path
        Path to the generated TRAIN dataset CSV.

    model_type : str
        Candidate model type (``logistic`` or ``forest``).

    Returns
    -------
    float
        Validation Recall score on the RISKY class.
    """

    cfg = load_config()

    df = pd.read_csv(dataset_path)

    # Features = all numeric columns except metadata + label
    feature_cols = [
        c for c in df.columns
        if c not in ("name", "class", "source_file", "label")
    ]

    X = df[feature_cols]
    y = df["label"]

    # Internal validation split (inside TRAIN pool, diagnostic only)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Build candidate model instance
    model = build_model(model_type, cfg)

    # Fit model on TRAIN subset
    model.fit(X_train, y_train)

    # Evaluate on validation subset
    recall_risky = evaluate_model(model, X_val, y_val)

    # Save candidate artifact
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    out_path = models_dir / f"{model_type}.pkl"
    joblib.dump(model, out_path)

    print(f"\nCandidate model saved to: {out_path}")

    return recall_risky


# ------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a candidate model for SpecLens-PML."
    )

    parser.add_argument(
        "dataset",
        help="Path to the generated TRAIN dataset CSV"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["logistic", "forest"],
        help="Candidate model family to train"
    )

    args = parser.parse_args()

    train(Path(args.dataset), args.model)

