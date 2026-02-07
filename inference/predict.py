"""
Inference module for SpecLens-PML.

This script performs inference using the promoted champion model
(`models/best_model.pkl`).

Given a Python source file annotated with PML contracts, it:

- Parses all contract-annotated functions
- Extracts the same feature schema used during training
- Predicts the probability of being RISKY
- Maps probabilities into operational risk levels:

    LOW / MEDIUM / HIGH

This module represents the *serving* component of the pipeline.
"""

from pathlib import Path
import joblib
import pandas as pd
import yaml

from pipeline.features import extract_features
from pml.parser import parse_file


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_thresholds() -> tuple[float, float]:
    """
    Load risk threshold values from the central configuration file.

    Thresholds are defined in ``config.yaml`` under:

    - ``risk_thresholds.low``
    - ``risk_thresholds.medium``

    Returns
    -------
    tuple[float, float]
        A tuple ``(low, medium)`` used to map probability scores
        into operational levels.
    """
    config = yaml.safe_load(Path("config.yaml").read_text())

    low = config["risk_thresholds"]["low"]
    medium = config["risk_thresholds"]["medium"]

    return low, medium


# ---------------------------------------------------------------------------
# Risk level mapping
# ---------------------------------------------------------------------------

def risk_level(score: float, low: float, medium: float) -> str:
    """
    Convert a probability score into an operational risk category.

    Parameters
    ----------
    score : float
        Predicted probability of the function being RISKY.
    low : float
        Threshold below which the function is considered LOW risk.
    medium : float
        Threshold below which the function is considered MEDIUM risk.

    Returns
    -------
    str
        One of: ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
    """
    if score < low:
        return "LOW"
    elif score < medium:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_file(path: Path) -> None:
    """
    Run inference on all contract-annotated functions in a Python file.

    The function:

    - Loads the promoted champion model (`best_model.pkl`)
    - Parses the input file using the PML parser
    - Extracts feature vectors
    - Predicts risk probabilities
    - Prints a per-function risk report

    Parameters
    ----------
    path : Path
        Path to the Python source file to analyze.
    """
    print(f"Analysis of {path.name}")
    print("(active model: best_model.pkl)\n")

    # Load thresholds from config.yaml
    low_t, med_t = load_thresholds()

    # Load champion model artifact
    model = joblib.load("models/best_model.pkl")

    # Parse contract-annotated functions from file
    functions = parse_file(path)

    for f in functions:
        feats = extract_features(f)

        X = pd.DataFrame([feats])
        score = model.predict_proba(X)[0][1]

        level = risk_level(score, low_t, med_t)

        print(f"- {f['name']} (line {f['line']})")
        print(f"  requires: {f['requires']}")
        print(f"  ensures:  {f['ensures']}")
        print(f"  invariant:{f['invariant']}")
        print(f"  → risk score: {score:.3f} [{level}]\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference/predict.py <file.py>")
        sys.exit(1)

    predict_file(Path(sys.argv[1]))

