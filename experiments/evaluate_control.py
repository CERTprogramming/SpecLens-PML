"""Compare baseline and concept-aware controlled operational decisions.

This experiment is deliberately separate from Champion/Challenger promotion.
Champion selection evaluates ``model.predict()``.  Here we evaluate the
operational HIGH-risk decision produced by the configured probability threshold
before and after the human-defined concept-aware control layer.

Policy thresholds must be fixed without using the final held-out TEST set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import yaml

from governance.concepts import extract_concept_states
from governance.control import apply_control, load_policies, policy_concepts
from pipeline.features import get_model_feature_names, make_feature_matrix


DEFAULT_DATASET = ROOT / "data" / "processed" / "datasets_test.csv"
DEFAULT_MODEL = ROOT / "models" / "best_model.pkl"
DEFAULT_POLICIES = ROOT / "governance" / "policies.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "control_outputs"


def load_thresholds() -> tuple[float, float]:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    return (
        float(config["risk_thresholds"]["low"]),
        float(config["risk_thresholds"]["medium"]),
    )


def risk_level(score: float, low: float, high: float) -> str:
    if score < low:
        return "LOW"
    if score < high:
        return "MEDIUM"
    return "HIGH"


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    """Compute transparent RISKY-vs-not-HIGH operational metrics."""
    truth = y_true.astype(int)
    pred = y_pred.astype(int)

    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth) if len(truth) else 0.0

    return {
        "risky_precision": precision,
        "risky_recall": recall,
        "risky_f1": f1,
        "accuracy": accuracy,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def evaluate(
    *,
    dataset_path: Path,
    model_path: Path,
    policy_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"TEST dataset not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found: {model_path}")

    df = pd.read_csv(dataset_path)
    if "label" not in df.columns:
        raise ValueError("Evaluation dataset must contain a 'label' column.")

    model = joblib.load(model_path)
    X = make_feature_matrix(df, get_model_feature_names(model))
    scores = model.predict_proba(X)[:, 1]

    low_t, high_t = load_thresholds()
    policy_document = load_policies(policy_path)
    relevant_concepts = policy_concepts(policy_document)

    rows: list[dict[str, Any]] = []
    for position, (_, source_row) in enumerate(df.iterrows()):
        score = float(scores[position])
        original_level = risk_level(score, low_t, high_t)
        feature_values = source_row.to_dict()
        concept_states = extract_concept_states(feature_values, relevant_concepts)

        control = apply_control(
            score=score,
            original_level=original_level,
            concept_states=concept_states,
            feature_values=feature_values,
            policy_document=policy_document,
            low_threshold=low_t,
            high_threshold=high_t,
        )

        rows.append(
            {
                "row_index": int(position),
                "file": source_row.get("file", source_row.get("source_file", "")),
                "function": source_row.get("function", source_row.get("name", "")),
                "true_label": int(source_row["label"]),
                "model_score": score,
                "original_level": original_level,
                "controlled_level": control.controlled_level,
                "original_high_threshold": control.original_high_threshold,
                "controlled_high_threshold": control.controlled_high_threshold,
                "decision_changed": control.decision_changed,
                "review_required": control.review_required,
                "matched_policy_ids": ";".join(control.matched_policy_ids),
                "policy_evidence": json.dumps(
                    {
                        policy.policy_id: list(policy.evidence)
                        for policy in control.matched_policies
                    },
                    sort_keys=True,
                ),
                "concept_states": json.dumps(concept_states, sort_keys=True),
            }
        )

    predictions = pd.DataFrame(rows)
    predictions["baseline_risky"] = (predictions["original_level"] == "HIGH").astype(int)
    predictions["controlled_risky"] = (predictions["controlled_level"] == "HIGH").astype(int)

    baseline = binary_metrics(predictions["true_label"], predictions["baseline_risky"])
    controlled = binary_metrics(predictions["true_label"], predictions["controlled_risky"])

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "control_predictions.csv", index=False)
    predictions[predictions["decision_changed"]].to_csv(
        output_dir / "changed_decisions.csv",
        index=False,
    )

    policy_rows: list[dict[str, Any]] = []
    enabled_policies = [
        policy for policy in policy_document.get("policies", [])
        if policy.get("enabled", False)
    ]
    for policy in enabled_policies:
        policy_id = str(policy["id"])
        matched = predictions["matched_policy_ids"].fillna("").map(
            lambda value: policy_id in {item for item in str(value).split(";") if item}
        )
        policy_rows.append(
            {
                "policy_id": policy_id,
                "name": policy.get("name", policy_id),
                "high_risk_threshold": float(policy["high_risk_threshold"]),
                "matched_cases": int(matched.sum()),
                "changed_cases": int((matched & predictions["decision_changed"]).sum()),
                "review_cases": int((matched & predictions["review_required"]).sum()),
            }
        )
    pd.DataFrame(policy_rows).to_csv(output_dir / "policy_summary.csv", index=False)

    summary = {
        "dataset": str(dataset_path),
        "model": str(model_path),
        "policy_version": int(policy_document.get("version", 1)),
        "baseline_high_threshold": high_t,
        "n_cases": int(len(predictions)),
        "decisions_changed": int(predictions["decision_changed"].sum()),
        "human_review_cases": int(predictions["review_required"].sum()),
        "baseline": baseline,
        "controlled": controlled,
        "false_negative_reduction": int(baseline["false_negatives"] - controlled["false_negatives"]),
        "false_positive_increase": int(controlled["false_positives"] - baseline["false_positives"]),
    }

    lines = [
        "SpecLens-PML control-layer evaluation",
        "=====================================",
        "",
        f"Dataset: {dataset_path}",
        f"Model: {model_path}",
        f"Policy version: {summary['policy_version']}",
        f"Cases: {summary['n_cases']}",
        f"Decisions changed: {summary['decisions_changed']}",
        f"Human-review cases: {summary['human_review_cases']}",
        "",
        "Operational RISKY decision = risk level HIGH",
        "",
        "Metric                         Baseline   Controlled",
        "---------------------------------------------------",
        f"RISKY recall                   {baseline['risky_recall']:.3f}      {controlled['risky_recall']:.3f}",
        f"RISKY precision                {baseline['risky_precision']:.3f}      {controlled['risky_precision']:.3f}",
        f"RISKY F1                       {baseline['risky_f1']:.3f}      {controlled['risky_f1']:.3f}",
        f"False negatives                {baseline['false_negatives']:>3}        {controlled['false_negatives']:>3}",
        f"False positives                {baseline['false_positives']:>3}        {controlled['false_positives']:>3}",
        "",
        f"False-negative reduction: {summary['false_negative_reduction']}",
        f"False-positive increase: {summary['false_positive_increase']}",
        "",
        "Note: this evaluates configured operational thresholds and policies;",
        "it does not replace the Champion/Challenger model-selection evaluation.",
    ]
    (output_dir / "control_summary.txt").write_text("\n".join(lines) + "\n")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SpecLens-PML baseline vs controlled decisions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(
        dataset_path=args.dataset,
        model_path=args.model,
        policy_path=args.policies,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
