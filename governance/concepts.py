"""Shared software-engineering concept taxonomy for SpecLens-PML.

The taxonomy is deliberately deterministic.  It is shared by the DPG
interpretation experiment and by the concept-aware control layer so that the
same feature name always denotes the same software-engineering concept.

The mapping is descriptive and must not be interpreted as a causal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ConceptRule:
    """Deterministic mapping from a feature to a concept family."""

    concept: str
    description: str
    feature_kind: str = "numeric"


CONCEPT_RULES: dict[str, ConceptRule] = {
    # Function surface and structure.
    "n_params": ConceptRule(
        "INTERFACE_COMPLEXITY",
        "size of the function interface",
        "count",
    ),
    "has_self": ConceptRule(
        "METHOD_CONTEXT",
        "method-oriented context through self",
        "boolean",
    ),
    "n_loc": ConceptRule(
        "STRUCTURAL_SIZE",
        "function size measured in lines of code",
        "count",
    ),
    "n_branches": ConceptRule(
        "CONTROL_FLOW_COMPLEXITY",
        "branching structure in the function body",
        "count",
    ),
    "n_loops": ConceptRule(
        "CONTROL_FLOW_COMPLEXITY",
        "looping structure in the function body",
        "count",
    ),
    "n_returns": ConceptRule(
        "RETURN_STRUCTURE",
        "number of explicit return points",
        "count",
    ),
    "has_subscript": ConceptRule(
        "INDEX_ACCESS",
        "indexed or subscript-based access",
        "boolean",
    ),
    "has_division": ConceptRule(
        "ARITHMETIC_RISK",
        "division or arithmetic operation requiring safety conditions",
        "boolean",
    ),
    "has_mutation": ConceptRule(
        "STATE_MUTATION",
        "mutation of state or mutable values",
        "boolean",
    ),
    "has_method_call": ConceptRule(
        "CALL_BEHAVIOR",
        "method-call behavior inside the function",
        "boolean",
    ),
    "has_other": ConceptRule(
        "OTHER_OPERATION",
        "residual structural behavior not captured by the main AST indicators",
        "boolean",
    ),
    # Contract coverage.
    "n_requires": ConceptRule(
        "PRECONDITION_COVERAGE",
        "amount of explicit precondition coverage",
        "count",
    ),
    "n_ensures": ConceptRule(
        "POSTCONDITION_COVERAGE",
        "amount of explicit postcondition coverage",
        "count",
    ),
    "n_invariants": ConceptRule(
        "INVARIANT_COVERAGE",
        "amount of invariant coverage",
        "count",
    ),
    "n_contracts_total": ConceptRule(
        "CONTRACT_COVERAGE",
        "overall amount of lightweight contract specification",
        "count",
    ),
    "has_missing_requires": ConceptRule(
        "MISSING_PRECONDITION_COVERAGE",
        "absence of explicit preconditions where they may be expected",
        "boolean",
    ),
    "has_stateful_contract": ConceptRule(
        "STATEFUL_CONTRACT",
        "contract referring to stateful behavior",
        "boolean",
    ),
    # Contract expression content.
    "requires_has_cmp": ConceptRule(
        "PRECONDITION_COMPARISON",
        "comparison expressions in preconditions",
        "boolean",
    ),
    "ensures_has_cmp": ConceptRule(
        "POSTCONDITION_COMPARISON",
        "comparison expressions in postconditions",
        "boolean",
    ),
    "invariants_has_cmp": ConceptRule(
        "INVARIANT_COMPARISON",
        "comparison expressions in invariants",
        "boolean",
    ),
    "requires_has_arith": ConceptRule(
        "PRECONDITION_ARITHMETIC",
        "arithmetic expressions in preconditions",
        "boolean",
    ),
    "ensures_has_arith": ConceptRule(
        "POSTCONDITION_ARITHMETIC",
        "arithmetic expressions in postconditions",
        "boolean",
    ),
    "invariants_has_arith": ConceptRule(
        "INVARIANT_ARITHMETIC",
        "arithmetic expressions in invariants",
        "boolean",
    ),
    "requires_complexity": ConceptRule(
        "PRECONDITION_COMPLEXITY",
        "syntactic complexity of preconditions",
        "complexity",
    ),
    "ensures_complexity": ConceptRule(
        "POSTCONDITION_COMPLEXITY",
        "syntactic complexity of postconditions",
        "complexity",
    ),
    "invariants_complexity": ConceptRule(
        "INVARIANT_COMPLEXITY",
        "syntactic complexity of invariants",
        "complexity",
    ),
    "contract_complexity_total": ConceptRule(
        "CONTRACT_COMPLEXITY",
        "overall syntactic complexity of contract expressions",
        "complexity",
    ),
    # Pre-state and snapshot-aware reasoning.
    "n_old_refs": ConceptRule(
        "PRESTATE_REASONING",
        "explicit references to values before execution through old(...)",
        "count",
    ),
    "ensures_has_old": ConceptRule(
        "PRESTATE_REASONING",
        "postconditions referring to the pre-state through old(...)",
        "boolean",
    ),
    "invariants_has_old": ConceptRule(
        "PRESTATE_REASONING",
        "invariants referring to the pre-state through old(...)",
        "boolean",
    ),
    "has_prestate_reference": ConceptRule(
        "PRESTATE_REASONING",
        "presence of pre-state-aware contract reasoning",
        "boolean",
    ),
    "contract_has_prestate_reference": ConceptRule(
        "PRESTATE_REASONING",
        "contract-level use of pre-state references",
        "boolean",
    ),
    "n_snapshots": ConceptRule(
        "SNAPSHOT_USAGE",
        "number of named pre-state snapshots",
        "count",
    ),
    "has_snapshot": ConceptRule(
        "SNAPSHOT_USAGE",
        "presence of named pre-state snapshots",
        "boolean",
    ),
    "ensures_uses_snapshot": ConceptRule(
        "SNAPSHOT_BASED_POSTCONDITION",
        "postconditions using named pre-state snapshots",
        "boolean",
    ),
    "snapshot_complexity": ConceptRule(
        "PRESTATE_CONTRACT_COMPLEXITY",
        "syntactic complexity of snapshot expressions",
        "complexity",
    ),
}


CONCEPT_ORDER = [
    "MISSING_PRECONDITION_COVERAGE",
    "PRECONDITION_COVERAGE",
    "POSTCONDITION_COVERAGE",
    "CONTRACT_COVERAGE",
    "INDEX_ACCESS",
    "ARITHMETIC_RISK",
    "STATE_MUTATION",
    "PRESTATE_REASONING",
    "SNAPSHOT_USAGE",
    "SNAPSHOT_BASED_POSTCONDITION",
    "PRESTATE_CONTRACT_COMPLEXITY",
    "PRECONDITION_COMPLEXITY",
    "POSTCONDITION_COMPLEXITY",
    "CONTRACT_COMPLEXITY",
    "CONTROL_FLOW_COMPLEXITY",
    "STRUCTURAL_SIZE",
    "INTERFACE_COMPLEXITY",
]


def _feature_is_present(value: Any, feature_kind: str) -> bool:
    """Convert a numeric feature value into a binary concept-presence signal."""
    if value is None:
        return False

    if feature_kind == "boolean":
        return bool(value)

    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return bool(value)


def extract_concept_states(
    features: dict[str, Any],
    concepts: Iterable[str] | None = None,
) -> dict[str, str]:
    """Aggregate feature values into deterministic ``present``/``absent`` states.

    Several features may map to the same concept (for example, the pre-state
    features).  A concept is considered ``present`` when at least one of its
    mapped feature values is active/non-zero.

    Parameters
    ----------
    features:
        Feature dictionary produced by :func:`pipeline.features.extract_features`.
    concepts:
        Optional concept subset.  When omitted, all mapped concepts represented
        by the supplied feature dictionary are returned.
    """
    requested = set(concepts) if concepts is not None else None
    present: dict[str, bool] = {}

    for feature, value in features.items():
        rule = CONCEPT_RULES.get(feature)
        if rule is None:
            continue
        if requested is not None and rule.concept not in requested:
            continue

        present.setdefault(rule.concept, False)
        present[rule.concept] = present[rule.concept] or _feature_is_present(
            value,
            rule.feature_kind,
        )

    if requested is not None:
        for concept in requested:
            present.setdefault(concept, False)

    return {
        concept: "present" if is_present else "absent"
        for concept, is_present in sorted(present.items())
    }
