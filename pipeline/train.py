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
    existing = list(models_dir.glob("model_v*.pkl"))
    if not existing:
        return models_dir / "model_v1.pkl"

    versions = []
    for p in existing:
        m = re.search(r"model_v(\d+)\.pkl", p.name)
        if m:
            versions.append(int(m.group(1)))

    next_v = max(versions) + 1 if versions else 1
    return models_dir / f"model_v{next_v}.pkl"


def main():
    if len(sys.argv) < 2:
        print("Usage: train.py <dataset.csv>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1]).resolve()

    ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = ROOT / "config.yaml"

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]

    df = pd.read_csv(dataset_path)

    FEATURE_COLS = ["n_params", "n_requires", "n_ensures", "n_invariants", "n_loc"]
    X = df[FEATURE_COLS]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", None),
        class_weight=model_cfg.get("class_weight", None),
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("=== Evaluation Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.3f}")
    print(f"Recall (RISKY): {rec:.3f}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = next_model_path(models_dir)
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
