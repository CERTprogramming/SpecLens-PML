"""
Inference module for SpecLens-PML.

This script performs inference using the promoted champion model
(`models/best_model.pkl`).

Given a Python source file annotated with PML contracts, it:

- Parses all contract-annotated functions
- Extracts the same feature schema used during training
- Predicts the probability of being RISKY
- Preserves the original LOW / MEDIUM / HIGH model-supported decision
- Maps features into deterministic software-engineering concept states
- Applies human-defined concept-aware control policies
- Keeps the model score, original decision, policy intervention, and controlled
  decision separate and observable
- Appends policy interventions to the governance audit log

The trained model and its probability are never modified by the control layer.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import yaml

from governance.concepts import extract_concept_states
from governance.control import (
    append_control_event,
    apply_control,
    load_policies,
    policy_concepts,
)
from pipeline.features import extract_features, get_model_feature_names, make_feature_matrix
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

    The ``medium`` value is the baseline HIGH-risk boundary: scores below it
    are MEDIUM, while scores at or above it are HIGH.

    Returns
    -------
    tuple[float, float]
        A tuple ``(low, medium)`` used to map probability scores
        into operational levels.
    """
    config = yaml.safe_load((ROOT / "config.yaml").read_text())

    low = float(config["risk_thresholds"]["low"])
    medium = float(config["risk_thresholds"]["medium"])

    return low, medium


# ---------------------------------------------------------------------------
# Risk level mapping
# ---------------------------------------------------------------------------

def risk_level(score: float, low: float, medium: float) -> str:
    """Convert a probability score into LOW, MEDIUM, or HIGH risk."""
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
    """Run baseline inference followed by concept-aware operational control."""
    print(f"Analysis of {path.name}")
    print("(active model: best_model.pkl)\n")

    # Baseline thresholds and human-defined control policy configuration are
    # intentionally loaded from separate files.
    low_t, med_t = load_thresholds()
    policy_document = load_policies()
    relevant_concepts = policy_concepts(policy_document)

    # Load champion model artifact.
    model = joblib.load(ROOT / "models" / "best_model.pkl")

    # Parse contract-annotated functions from file.
    functions = parse_file(path)

    for f in functions:
        feats = extract_features(f)

        X = make_feature_matrix(
            pd.DataFrame([feats]),
            get_model_feature_names(model),
        )
        score = float(model.predict_proba(X)[0][1])

        # 1) Preserve the original model-supported operational decision.
        original_level = risk_level(score, low_t, med_t)

        # 2) Translate only policy-relevant features into shared concept states.
        concept_states = extract_concept_states(feats, relevant_concepts)

        # 3) Apply human-defined control policy without changing ``score``.
        control = apply_control(
            score=score,
            original_level=original_level,
            concept_states=concept_states,
            feature_values=feats,
            policy_document=policy_document,
            low_threshold=low_t,
            high_threshold=med_t,
        )

        # 4) Record actual policy interventions in an append-only audit log.
        event_id = append_control_event(
            control,
            function_name=f["name"],
            source_file=path.name,
            line=f.get("line"),
            concept_states=concept_states,
        )

        print(f"- {f['name']} (line {f['line']})")
        print(f"  requires: {f['requires']}")
        print(f"  ensures:  {f['ensures']}")
        print(f"  invariant:{f['invariant']}")
        print(f"  snapshots:{f.get('snapshots', {})}")
        print(f"  Model risk score: {score:.3f}")
        print(f"  Original risk level: {original_level}")

        print("  Relevant concepts:")
        if concept_states:
            for concept, state in concept_states.items():
                print(f"    {concept}: {state}")
        else:
            print("    none")

        print("  Applied policies:")
        if control.matched_policies:
            for policy in control.matched_policies:
                print(f"    {policy.policy_id} - {policy.name}")
                for evidence in policy.evidence:
                    print(f"      evidence: {evidence}")
                if policy.rationale:
                    print(f"      rationale: {policy.rationale}")
        else:
            print("    none")

        print(f"  Original HIGH threshold: {control.original_high_threshold:.3f}")
        print(f"  Controlled HIGH threshold: {control.controlled_high_threshold:.3f}")
        print(f"  Controlled risk level: {control.controlled_level}")
        print(f"  Decision changed: {'yes' if control.decision_changed else 'no'}")
        print(f"  Human review: {'required' if control.review_required else 'not required'}")
        print(f"  Governance event: {event_id or 'none'}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference/predict.py <file.py>")
        sys.exit(1)

    predict_file(Path(sys.argv[1]))
