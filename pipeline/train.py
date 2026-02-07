"""
SpecLens-PML Model Training.

This script implements the supervised learning stage of the pipeline.

It trains multiple candidate models (baseline + challenger), including:

- Logistic Regression (baseline)
- Random Forest (challenger)

Each trained model is saved as an independent artifact so that the
Continuous Training trigger (ct_trigger.py) can later evaluate and
promote the best-performing one.

The key metric for promotion is:

    Recall on the RISKY class (label = 1)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_model(model_type: str):
    """
    Construct a classifier given its type.

    Supported models
    ----------------
    logistic : LogisticRegression baseline
    forest   : RandomForest challenger

    Parameters
    ----------
    model_type : str
        Identifier of the model type.

    Returns
    -------
    sklearn estimator
        Instantiated classifier ready for training.
    """

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000)

    if model_type == "forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
        )

    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Evaluation Helper (shared with ct_trigger.py)
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test) -> float:
    """
    Evaluate a trained model and return Recall on the RISKY class.

    Parameters
    ----------
    model :
        Trained classifier.
    X_test :
        Feature matrix for validation.
    y_test :
        Ground truth labels.

    Returns
    -------
    float
        Recall score for the RISKY class (label = 1).
    """

    y_pred = model.predict(X_test)

    print("\n=== Evaluation Report ===")
    print(classification_report(y_test, y_pred))

    recall_risky = recall_score(y_test, y_pred, pos_label=1)
    print(f"Recall (RISKY): {recall_risky:.3f}")

    return recall_risky


# ---------------------------------------------------------------------------
# Training Procedure
# ---------------------------------------------------------------------------

def train(dataset_path: Path, model_type: str) -> float:
    """
    Train a candidate model on the generated dataset.

    The trained artifact is saved as:

        models/{model_type}.pkl

    Parameters
    ----------
    dataset_path : Path
        Path to the CSV dataset produced by build_dataset.py.
    model_type : str
        Candidate model identifier ("logistic" or "forest").

    Returns
    -------
    float
        Recall score on the RISKY class.
    """

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    df = pd.read_csv(dataset_path)

    # Features = all numeric columns except metadata + label
    feature_cols = [
        c for c in df.columns
        if c not in ("name", "class", "source_file", "label")
    ]

    X = df[feature_cols]
    y = df["label"]

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------------
    # Build and train model
    # --------------------------------------------------------
    model = build_model(model_type)
    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Evaluate candidate
    # --------------------------------------------------------
    recall_risky = evaluate_model(model, X_test, y_test)

    # --------------------------------------------------------
    # Save trained artifact
    # --------------------------------------------------------
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    out_path = models_dir / f"{model_type}.pkl"
    joblib.dump(model, out_path)

    print(f"\nModel saved to {out_path}")

    return recall_risky


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a candidate model for SpecLens-PML."
    )

    parser.add_argument(
        "dataset",
        help="Path to the CSV dataset generated by build_dataset.py",
    )

    parser.add_argument(
        "--model",
        default="logistic",
        help="Model type: logistic or forest",
    )

    args = parser.parse_args()

    train(Path(args.dataset), args.model)

