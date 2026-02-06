#!/usr/bin/env python3

from pathlib import Path
import subprocess
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ACTIVE = MODELS / "active_model.txt"
DATASET = ROOT / "data" / "datasets_v1.csv"

FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]


def evaluate(model_path: Path) -> float:
    df = pd.read_csv(DATASET)

    X = df[FEATURE_COLS]
    y = df["label"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    model = joblib.load(model_path)
    ypred = model.predict(Xte)

    return recall_score(yte, ypred, zero_division=0)


def main():
    print("=== Continuous Training Trigger ===")

    # Train new model
    subprocess.call(["python3", "pipeline/train.py", str(DATASET)])

    # Latest model by modification time
    latest = max(MODELS.glob("model_v*.pkl"), key=lambda p: p.stat().st_mtime)

    # Current active model
    with open(ACTIVE) as f:
        current = ROOT / f.read().strip()

    new_recall = evaluate(latest)
    cur_recall = evaluate(current)

    print(f"Current recall (RISKY): {cur_recall:.3f}")
    print(f"New recall (RISKY):     {new_recall:.3f}")

    if new_recall >= cur_recall:
        with open(ACTIVE, "w") as f:
            f.write(str(latest.relative_to(ROOT)))
        print(f"New model promoted: {latest.name}")
    else:
        print("New model rejected (performance did not improve).")


if __name__ == "__main__":
    main()
