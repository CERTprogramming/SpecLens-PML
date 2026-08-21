"""Concept-aware operational control and governance support for SpecLens-PML.

The control layer never modifies the trained model or its probability score.
Human-defined policies may only make the operational HIGH-risk threshold more
conservative.

Policies may use either:

1. deterministic concept states, for example::

       when:
         INDEX_ACCESS: present

2. DPG-derived feature predicates associated with software-engineering
   concepts, for example::

       when:
         - concept: CONTRACT_COVERAGE
           feature: n_contracts_total
           operator: "<="
           value: 2.5

The concept labels make policy intent explicit while feature predicates preserve
the quantitative evidence exposed by the Decision Predicate Graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from governance.concepts import CONCEPT_RULES


@dataclass(frozen=True)
class PolicyMatch:
    """One human-defined policy matched during operational control."""

    policy_id: str
    name: str
    high_risk_threshold: float
    require_review: bool
    rationale: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControlDecision:
    """Observable separation between model, policy control, and decision result."""

    model_score: float
    original_level: str
    original_high_threshold: float
    controlled_high_threshold: float
    controlled_level: str
    decision_changed: bool
    review_required: bool
    policy_version: int
    matched_policies: tuple[PolicyMatch, ...]

    @property
    def matched_policy_ids(self) -> tuple[str, ...]:
        return tuple(policy.policy_id for policy in self.matched_policies)


DEFAULT_POLICY_PATH = Path(__file__).with_name("policies.yaml")
DEFAULT_AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "governance"
    / "control_events.jsonl"
)

VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_OPERATORS = {"<=", "<", ">=", ">", "==", "!="}


def load_policies(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load and minimally validate the concept-aware policy configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise ValueError("Policy file must contain a YAML mapping.")

    version = document.get("version", 1)
    policies = document.get("policies", [])

    if not isinstance(version, int) or version < 1:
        raise ValueError("Policy version must be a positive integer.")

    if not isinstance(policies, list):
        raise ValueError("'policies' must be a YAML list.")

    document["version"] = version
    document["policies"] = policies
    return document


def policy_concepts(policy_document: dict[str, Any]) -> set[str]:
    """Return software-engineering concepts referenced by configured policies."""
    result: set[str] = set()

    for policy in policy_document.get("policies", []):
        conditions = policy.get("when", {})

        # Legacy/simple concept-state syntax.
        if isinstance(conditions, dict):
            result.update(str(concept) for concept in conditions)

        # Quantitative concept-aware predicate syntax.
        elif isinstance(conditions, list):
            for condition in conditions:
                if isinstance(condition, dict) and "concept" in condition:
                    result.add(str(condition["concept"]))

    return result


def _risk_level(
    score: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    if score < low_threshold:
        return "LOW"
    if score < high_threshold:
        return "MEDIUM"
    return "HIGH"


def _compare_numeric(
    actual: float,
    operator: str,
    expected: float,
) -> bool:
    if operator == "<=":
        return actual <= expected
    if operator == "<":
        return actual < expected
    if operator == ">=":
        return actual >= expected
    if operator == ">":
        return actual > expected
    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected

    raise ValueError(f"Unsupported policy operator: {operator}")


def _matches(
    conditions: Any,
    concept_states: dict[str, str],
    feature_values: dict[str, Any] | None,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate one policy and return match status plus readable evidence."""

    # Existing present/absent concept-state policy format.
    if isinstance(conditions, dict):
        evidence: list[str] = []

        for concept, expected in conditions.items():
            concept = str(concept)
            expected = str(expected)
            actual = concept_states.get(concept)

            if actual != expected:
                return False, ()

            evidence.append(f"{concept}={actual}")

        return True, tuple(evidence)

    # DPG-derived quantitative predicate format.
    if isinstance(conditions, list):
        # Allows older callers/tests that only provide concept states to keep
        # working. Quantitative policies simply cannot match without features.
        if feature_values is None:
            return False, ()

        evidence: list[str] = []

        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError("Each quantitative policy condition must be a mapping.")

            try:
                concept = str(condition["concept"])
                feature = str(condition["feature"])
                operator = str(condition["operator"])
                expected = float(condition["value"])
            except KeyError as exc:
                raise ValueError(
                    f"Missing quantitative policy field: {exc.args[0]}"
                ) from exc

            if operator not in VALID_OPERATORS:
                raise ValueError(f"Unsupported policy operator: {operator}")

            rule = CONCEPT_RULES.get(feature)
            if rule is None:
                raise ValueError(
                    f"Policy references unknown feature: {feature}"
                )

            if rule.concept != concept:
                raise ValueError(
                    f"Feature {feature!r} belongs to concept "
                    f"{rule.concept!r}, not {concept!r}."
                )

            if feature not in feature_values:
                return False, ()

            try:
                actual = float(feature_values[feature])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Policy feature {feature!r} is not numeric: "
                    f"{feature_values[feature]!r}"
                ) from exc

            if not _compare_numeric(actual, operator, expected):
                return False, ()

            evidence.append(
                f"{concept}: {feature}={actual:g} {operator} {expected:g}"
            )

        return True, tuple(evidence)

    raise ValueError(
        "Policy 'when' must be either a concept-state mapping "
        "or a list of quantitative concept predicates."
    )


def apply_control(
    *,
    score: float,
    original_level: str,
    concept_states: dict[str, str],
    policy_document: dict[str, Any],
    low_threshold: float,
    high_threshold: float,
    feature_values: dict[str, Any] | None = None,
) -> ControlDecision:
    """Apply human-defined conservative policies without changing model score."""

    if original_level not in VALID_LEVELS:
        raise ValueError(f"Invalid original risk level: {original_level}")

    controlled_high = float(high_threshold)
    matches: list[PolicyMatch] = []

    for raw_policy in policy_document.get("policies", []):
        if not raw_policy.get("enabled", False):
            continue

        policy_id = str(raw_policy.get("id", "")).strip()
        if not policy_id:
            raise ValueError("Enabled policies require a non-empty 'id'.")

        conditions = raw_policy.get("when", {})
        if not conditions:
            raise ValueError(f"Policy {policy_id} has no conditions.")

        matched, evidence = _matches(
            conditions,
            concept_states,
            feature_values,
        )

        if not matched:
            continue

        threshold = float(raw_policy["high_risk_threshold"])

        # Policies may only become more conservative.
        if threshold > high_threshold:
            raise ValueError(
                f"Policy {policy_id} is less conservative than the "
                f"baseline HIGH threshold ({threshold} > {high_threshold})."
            )

        if threshold < low_threshold:
            raise ValueError(
                f"Policy {policy_id} HIGH threshold ({threshold}) "
                f"cannot be lower than the LOW threshold ({low_threshold})."
            )

        matches.append(
            PolicyMatch(
                policy_id=policy_id,
                name=str(raw_policy.get("name", policy_id)),
                high_risk_threshold=threshold,
                require_review=bool(
                    raw_policy.get("require_review", False)
                ),
                rationale=str(
                    raw_policy.get("rationale", "")
                ).strip(),
                evidence=evidence,
            )
        )

        controlled_high = min(controlled_high, threshold)

    controlled_level = _risk_level(
        score,
        low_threshold,
        controlled_high,
    )

    decision_changed = controlled_level != original_level

    # Human attention is requested when a governance policy actually changes
    # the operational decision, rather than for every harmless policy match.
    review_required = (
        decision_changed
        and any(policy.require_review for policy in matches)
    )

    return ControlDecision(
        model_score=float(score),
        original_level=original_level,
        original_high_threshold=float(high_threshold),
        controlled_high_threshold=float(controlled_high),
        controlled_level=controlled_level,
        decision_changed=decision_changed,
        review_required=review_required,
        policy_version=int(policy_document.get("version", 1)),
        matched_policies=tuple(matches),
    )


def _new_event_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"EVT-{timestamp}-{uuid4().hex[:8]}"


def append_control_event(
    decision: ControlDecision,
    *,
    function_name: str,
    source_file: str,
    line: int | None,
    concept_states: dict[str, str],
    path: Path = DEFAULT_AUDIT_PATH,
) -> str | None:
    """Append an immutable audit record when at least one policy matched."""
    if not decision.matched_policies:
        return None

    event_id = _new_event_id()

    record = {
        "record_type": "control_event",
        "event_id": event_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "function": function_name,
        "source_file": source_file,
        "line": line,
        "model_score": decision.model_score,
        "original_level": decision.original_level,
        "concept_states": concept_states,
        "matched_policies": [
            asdict(policy)
            for policy in decision.matched_policies
        ],
        "original_high_threshold": decision.original_high_threshold,
        "controlled_high_threshold": decision.controlled_high_threshold,
        "controlled_level": decision.controlled_level,
        "decision_changed": decision.decision_changed,
        "review_required": decision.review_required,
        "policy_version": decision.policy_version,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, sort_keys=True) + "\n"
        )

    return event_id
