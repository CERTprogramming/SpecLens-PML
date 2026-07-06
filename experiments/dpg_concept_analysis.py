"""
Concept-level analysis for SpecLens-PML DPG communities.

This script reads the CSV artifacts produced by ``experiments/dpg_explain_forest.py``
and adds an interpretation layer that maps DPG predicates to software-engineering
concept families.

The mapping is intentionally deterministic and transparent. It should be treated
as a first taxonomy for project-level interpretation, not as a causal model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import argparse
import math
import re
import textwrap

import pandas as pd


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "dpg_outputs"

DEFAULT_TOP_CONCEPTS = 8
DEFAULT_TOP_PREDICATES_PER_CONCEPT = 5
DEFAULT_INTERPRETATION_TOP_CONCEPTS = 6

PREDICATE_RE = re.compile(
    r"(?P<feature>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<operator><=|>=|>|<|==)\s*"
    r"(?P<threshold>-?\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Concept taxonomy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptRule:
    """
    Deterministic mapping from a feature name to a concept family.

    Parameters
    ----------
    concept : str
        Software-engineering concept family.
    description : str
        Human-readable explanation of the concept.
    feature_kind : str
        Feature type used to interpret thresholds. Supported values are
        ``boolean``, ``count``, ``complexity`` and ``numeric``.
    """

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


# ---------------------------------------------------------------------------
# Predicate parsing and interpretation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredicateInterpretation:
    """
    Parsed and interpreted representation of a DPG predicate.

    Parameters
    ----------
    raw_predicate : str
        Original predicate label extracted from DPG.
    feature : str
        Feature name on the left-hand side of the predicate.
    operator : str
        Comparison operator.
    threshold : float
        Numeric threshold used by the tree split.
    concept : str
        Software-engineering concept assigned by the deterministic taxonomy.
    polarity : str
        Qualitative interpretation of the predicate direction.
    interpretation : str
        Human-readable reading of the predicate.
    mapped : bool
        Whether the feature was explicitly mapped by the taxonomy.
    """

    raw_predicate: str
    feature: str
    operator: str
    threshold: float
    concept: str
    polarity: str
    interpretation: str
    mapped: bool


def parse_predicate(label: Any) -> Optional[tuple[str, str, float]]:
    """
    Parse a DPG predicate label into feature, operator and threshold.

    Parameters
    ----------
    label : Any
        Predicate label generated by DPG.

    Returns
    -------
    Optional[tuple[str, str, float]]
        Parsed tuple, or ``None`` when the label does not look like a predicate.
    """
    match = PREDICATE_RE.search(str(label))
    if not match:
        return None
    return (
        match.group("feature"),
        match.group("operator"),
        float(match.group("threshold")),
    )


def is_upper_side(operator: str) -> bool:
    """
    Return whether a predicate selects the high side of a threshold.

    Parameters
    ----------
    operator : str
        Predicate comparison operator.

    Returns
    -------
    bool
        ``True`` for ``>`` and ``>=``; ``False`` otherwise.
    """
    return operator in {">", ">="}


def integer_boundary(operator: str, threshold: float) -> str:
    """
    Convert a tree threshold into an integer-oriented reading when possible.

    Parameters
    ----------
    operator : str
        Predicate comparison operator.
    threshold : float
        Numeric tree threshold.

    Returns
    -------
    str
        Human-readable integer boundary, such as ``at least 3`` or ``at most 2``.
    """
    if operator in {">", ">="}:
        value = (
            math.floor(threshold) + 1
            if float(threshold).is_integer()
            else math.ceil(threshold)
        )
        return f"at least {value}"
    if operator in {"<", "<="}:
        value = math.floor(threshold)
        return f"at most {value}"
    return f"equal to {threshold:g}"


def interpret_boolean(feature: str, operator: str) -> tuple[str, str]:
    """
    Interpret a boolean feature split as present or absent.

    Parameters
    ----------
    feature : str
        Feature name.
    operator : str
        Predicate comparison operator.

    Returns
    -------
    tuple[str, str]
        Polarity and short predicate reading.
    """
    if is_upper_side(operator):
        return "present", f"{feature} is present"
    return "absent", f"{feature} is absent"


def interpret_count(feature: str, operator: str, threshold: float) -> tuple[str, str]:
    """
    Interpret a count-based split.

    Parameters
    ----------
    feature : str
        Count feature name.
    operator : str
        Predicate comparison operator.
    threshold : float
        Numeric tree threshold.

    Returns
    -------
    tuple[str, str]
        Polarity and short predicate reading.
    """
    boundary = integer_boundary(operator, threshold)
    polarity = "higher" if is_upper_side(operator) else "lower"
    if boundary == "at most 0":
        return "none_or_low", f"{feature} is absent or zero"
    return polarity, f"{feature} is {boundary}"


def interpret_numeric(
    feature: str,
    operator: str,
    threshold: float,
    feature_kind: str,
) -> tuple[str, str]:
    """
    Interpret numeric and complexity-based predicates.

    Parameters
    ----------
    feature : str
        Feature name.
    operator : str
        Predicate comparison operator.
    threshold : float
        Numeric tree threshold.
    feature_kind : str
        Feature kind from the concept taxonomy.

    Returns
    -------
    tuple[str, str]
        Polarity and short predicate reading.
    """
    if feature_kind == "boolean":
        return interpret_boolean(feature, operator)
    if feature_kind == "count":
        return interpret_count(feature, operator, threshold)
    polarity = "higher" if is_upper_side(operator) else "lower"
    relation = "above" if is_upper_side(operator) else "at or below"
    return polarity, f"{feature} is {relation} {threshold:g}"


def interpret_predicate(label: Any) -> PredicateInterpretation:
    """
    Map a DPG predicate into a software-engineering concept.

    Parameters
    ----------
    label : Any
        Original DPG predicate label.

    Returns
    -------
    PredicateInterpretation
        Parsed predicate, concept family and textual interpretation.
    """
    raw = str(label)
    parsed = parse_predicate(raw)
    if parsed is None:
        return PredicateInterpretation(
            raw_predicate=raw,
            feature="",
            operator="",
            threshold=float("nan"),
            concept="UNPARSED_PREDICATE",
            polarity="unknown",
            interpretation="Predicate could not be parsed automatically.",
            mapped=False,
        )

    feature, operator, threshold = parsed
    rule = CONCEPT_RULES.get(
        feature,
        ConceptRule(
            "UNMAPPED_FEATURE",
            "feature not yet assigned to a software-engineering concept",
            "numeric",
        ),
    )
    polarity, reading = interpret_numeric(
        feature,
        operator,
        threshold,
        rule.feature_kind,
    )
    interpretation = f"{reading}; concept: {rule.description}"
    return PredicateInterpretation(
        raw_predicate=raw,
        feature=feature,
        operator=operator,
        threshold=threshold,
        concept=rule.concept,
        polarity=polarity,
        interpretation=interpretation,
        mapped=feature in CONCEPT_RULES,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def require_file(path: Path) -> None:
    """
    Raise a helpful error if a required artifact is missing.

    Parameters
    ----------
    path : Path
        Artifact path expected by the concept analysis.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required DPG artifact not found: {path}. "
            "Run experiments/dpg_explain_forest.py first."
        )


def load_community_artifacts(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DPG community summary and predicate membership artifacts.

    Parameters
    ----------
    output_dir : Path
        Directory containing DPG CSV outputs.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Community class summary and community predicate table.
    """
    summary_path = output_dir / "community_class_summary.csv"
    predicates_path = output_dir / "community_predicates.csv"
    require_file(summary_path)
    require_file(predicates_path)
    return pd.read_csv(summary_path), pd.read_csv(predicates_path)


# ---------------------------------------------------------------------------
# Community concept analysis
# ---------------------------------------------------------------------------

def concept_sort_key(concept: str) -> tuple[int, str]:
    """
    Return a stable sort key for concept families.

    Parameters
    ----------
    concept : str
        Concept family name.

    Returns
    -------
    tuple[int, str]
        Numeric priority and fallback lexical key.
    """
    try:
        return CONCEPT_ORDER.index(concept), concept
    except ValueError:
        return len(CONCEPT_ORDER), concept


def enrich_predicates_with_concepts(predicates: pd.DataFrame) -> pd.DataFrame:
    """
    Add concept taxonomy columns to community predicate rows.

    Parameters
    ----------
    predicates : pandas.DataFrame
        Table produced by ``dpg_explain_forest.py``.

    Returns
    -------
    pandas.DataFrame
        Predicate rows enriched with parsed feature, concept, polarity and
        interpretation columns.
    """
    if "Label" not in predicates.columns:
        raise ValueError("community_predicates.csv must contain a Label column.")

    enriched = predicates.copy()
    interpretations = [interpret_predicate(label) for label in enriched["Label"]]
    enriched["Feature"] = [item.feature for item in interpretations]
    enriched["Operator"] = [item.operator for item in interpretations]
    enriched["Threshold"] = [item.threshold for item in interpretations]
    enriched["Concept"] = [item.concept for item in interpretations]
    enriched["Polarity"] = [item.polarity for item in interpretations]
    enriched["Concept interpretation"] = [
        item.interpretation for item in interpretations
    ]
    enriched["Mapped concept"] = [item.mapped for item in interpretations]
    return enriched


def top_joined_values(values: pd.Series, limit: int) -> str:
    """
    Join unique values into a compact semicolon-separated string.

    Parameters
    ----------
    values : pandas.Series
        Values to summarize.
    limit : int
        Maximum number of unique values.

    Returns
    -------
    str
        Compact textual list.
    """
    result: list[str] = []
    for value in values.dropna().astype(str):
        if value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return "; ".join(result)


def build_community_concept_summary(
    community_summary: pd.DataFrame,
    enriched_predicates: pd.DataFrame,
    top_predicates_per_concept: int,
) -> pd.DataFrame:
    """
    Aggregate predicate-level concepts for each DPG community.

    Parameters
    ----------
    community_summary : pandas.DataFrame
        Class-aware community summary produced by the DPG experiment.
    enriched_predicates : pandas.DataFrame
        Community predicate rows enriched with concept taxonomy columns.
    top_predicates_per_concept : int
        Number of representative predicates to retain for each concept.

    Returns
    -------
    pandas.DataFrame
        One row per community and concept family.
    """
    required_columns = {"Community", "Concept", "Label"}
    missing = required_columns - set(enriched_predicates.columns)
    if missing:
        raise ValueError(f"Missing predicate columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for (community, concept), group in enriched_predicates.groupby(
        ["Community", "Concept"],
        sort=False,
    ):
        ranked = group.copy()
        if "Local reaching centrality" in ranked.columns:
            ranked["Local reaching centrality"] = pd.to_numeric(
                ranked["Local reaching centrality"],
                errors="coerce",
            ).fillna(0.0)
        if "Betweenness centrality" in ranked.columns:
            ranked["Betweenness centrality"] = pd.to_numeric(
                ranked["Betweenness centrality"],
                errors="coerce",
            ).fillna(0.0)
        if "Degree" in ranked.columns:
            ranked["Degree"] = pd.to_numeric(
                ranked["Degree"],
                errors="coerce",
            ).fillna(0.0)

        rank_columns = [
            column for column in [
                "Local reaching centrality",
                "Betweenness centrality",
                "Degree",
                "Label",
            ]
            if column in ranked.columns
        ]
        ascending = [False] * (len(rank_columns) - 1) + [True]
        ranked = ranked.sort_values(rank_columns, ascending=ascending)

        rows.append(
            {
                "Community": community,
                "Concept": concept,
                "Predicate count": len(group),
                "Mapped predicates": int(group["Mapped concept"].sum()),
                "Distinct features": top_joined_values(group["Feature"], 10),
                "Representative predicates": top_joined_values(
                    ranked["Label"],
                    top_predicates_per_concept,
                ),
                "Interpretations": top_joined_values(
                    ranked["Concept interpretation"],
                    top_predicates_per_concept,
                ),
                "Sum local reaching": float(
                    ranked.get("Local reaching centrality", pd.Series(dtype=float)).sum()
                ),
                "Sum betweenness": float(
                    ranked.get("Betweenness centrality", pd.Series(dtype=float)).sum()
                ),
                "Sum degree": float(ranked.get("Degree", pd.Series(dtype=float)).sum()),
            }
        )

    concept_summary = pd.DataFrame(rows)
    concept_summary["_concept_order"] = concept_summary["Concept"].map(
        lambda value: concept_sort_key(str(value))[0]
    )
    concept_summary = concept_summary.sort_values(
        ["Community", "Predicate count", "Sum local reaching", "_concept_order"],
        ascending=[True, False, False, True],
    ).drop(columns=["_concept_order"])

    join_columns = [
        column for column in [
            "Community",
            "Display label",
            "Dominant class",
            "SAFE association score",
            "RISKY association score",
            "SAFE edge weight",
            "RISKY edge weight",
        ]
        if column in community_summary.columns
    ]
    if join_columns:
        concept_summary = concept_summary.merge(
            community_summary[join_columns],
            on="Community",
            how="left",
        )
        front = [column for column in join_columns if column in concept_summary.columns]
        remaining = [column for column in concept_summary.columns if column not in front]
        concept_summary = concept_summary[front + remaining]

    return concept_summary


def write_concept_taxonomy(path: Path) -> Path:
    """
    Save the deterministic concept taxonomy as CSV.

    Parameters
    ----------
    path : Path
        Output CSV path.

    Returns
    -------
    Path
        Written output path.
    """
    rows = [
        {
            "Feature": feature,
            "Concept": rule.concept,
            "Feature kind": rule.feature_kind,
            "Description": rule.description,
        }
        for feature, rule in sorted(CONCEPT_RULES.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_community_concept_report(
    path: Path,
    community_summary: pd.DataFrame,
    concept_summary: pd.DataFrame,
    top_concepts: int,
) -> Path:
    """
    Write a readable concept-level report for DPG communities.

    Parameters
    ----------
    path : Path
        Output text path.
    community_summary : pandas.DataFrame
        Class-aware DPG community summary.
    concept_summary : pandas.DataFrame
        Concept aggregation table.
    top_concepts : int
        Maximum number of concept families shown per community.

    Returns
    -------
    Path
        Written report path.
    """
    lines = [
        "SpecLens-PML DPG concept-level community analysis",
        "=================================================",
        "",
        "This report maps DPG predicates to deterministic software-engineering",
        "concept families. The mapping is descriptive and reproducible; it does",
        "not imply causality between a concept and a class label.",
        "",
    ]

    for _, community in community_summary.iterrows():
        community_id = community["Community"]
        display = community.get("Display label", f"Community {community_id}")
        dominant = community.get("Dominant class", "n/a")
        safe_score = float(community.get("SAFE association score", 0.0))
        risky_score = float(community.get("RISKY association score", 0.0))
        predicate_count = int(community.get("Predicate count", 0))
        lines.extend(
            [
                str(display),
                "-" * len(str(display)),
                f"DPG cluster: {community_id}",
                f"Dominant class: {dominant}",
                f"Predicates: {predicate_count}",
                "Class association: "
                f"SAFE {safe_score * 100:.1f}% | RISKY {risky_score * 100:.1f}%",
                "",
                "Main concept families:",
            ]
        )

        subset = concept_summary[concept_summary["Community"] == community_id].copy()
        subset = subset.sort_values(
            ["Predicate count", "Sum local reaching", "Sum betweenness"],
            ascending=[False, False, False],
        ).head(top_concepts)
        if subset.empty:
            lines.append("- No mapped concepts available.")
        for _, concept in subset.iterrows():
            predicates = str(concept["Representative predicates"])
            wrapped = textwrap.wrap(predicates, width=88)
            lines.append(
                f"- {concept['Concept']} "
                f"({int(concept['Predicate count'])} predicates)"
            )
            if wrapped:
                lines.append(f"  Examples: {wrapped[0]}")
                for continuation in wrapped[1:]:
                    lines.append(f"            {continuation}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path



# ---------------------------------------------------------------------------
# Project-level interpretation summary
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float while tolerating missing CSV fields.

    Parameters
    ----------
    value : Any
        Input value.
    default : float, optional
        Fallback value when conversion is not possible.

    Returns
    -------
    float
        Converted numeric value or fallback.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percentage(value: Any) -> str:
    """
    Format an association score as a percentage.

    Parameters
    ----------
    value : Any
        Numeric score in the range ``[0, 1]`` when available.

    Returns
    -------
    str
        Percentage string with one decimal digit.
    """
    return f"{safe_float(value) * 100:.1f}%"


def top_concepts_for_community(
    concept_summary: pd.DataFrame,
    community_id: Any,
    limit: int,
) -> pd.DataFrame:
    """
    Select the most representative concept families for one community.

    Parameters
    ----------
    concept_summary : pandas.DataFrame
        Concept aggregation table.
    community_id : Any
        Community identifier used by DPG.
    limit : int
        Maximum number of concepts to return.

    Returns
    -------
    pandas.DataFrame
        Ranked concept rows for the requested community.
    """
    subset = concept_summary[concept_summary["Community"] == community_id].copy()
    if subset.empty:
        return subset
    return subset.sort_values(
        ["Predicate count", "Sum local reaching", "Sum betweenness"],
        ascending=[False, False, False],
    ).head(limit)


def top_central_concepts_for_community(
    concept_summary: pd.DataFrame,
    community_id: Any,
    limit: int,
) -> pd.DataFrame:
    """
    Select concept families that are structurally central in one community.

    Parameters
    ----------
    concept_summary : pandas.DataFrame
        Concept aggregation table.
    community_id : Any
        Community identifier used by DPG.
    limit : int
        Maximum number of concepts to return.

    Returns
    -------
    pandas.DataFrame
        Concept rows ranked by aggregated centrality metrics.
    """
    subset = concept_summary[concept_summary["Community"] == community_id].copy()
    if subset.empty:
        return subset
    return subset.sort_values(
        ["Sum betweenness", "Sum local reaching", "Sum degree", "Predicate count"],
        ascending=[False, False, False, False],
    ).head(limit)


def append_concept_family_lines(
    lines: list[str],
    concepts: pd.DataFrame,
    title: str,
    include_centrality: bool,
) -> None:
    """
    Append formatted concept-family rows to a text report.

    Parameters
    ----------
    lines : list[str]
        Report lines to extend.
    concepts : pandas.DataFrame
        Ranked concept rows.
    title : str
        Section title to add before the concept rows.
    include_centrality : bool
        Whether to include aggregated centrality values in each row.
    """
    lines.extend(["", title])
    if concepts.empty:
        lines.append("- No mapped concepts available.")
        return

    for _, concept in concepts.iterrows():
        representatives = str(concept["Representative predicates"])
        metric_text = ""
        if include_centrality:
            metric_text = (
                f"; betweenness={safe_float(concept['Sum betweenness']):.4f}"
                f"; local_reaching={safe_float(concept['Sum local reaching']):.4f}"
                f"; degree={safe_float(concept['Sum degree']):.0f}"
            )
        predicate_count = int(concept["Predicate count"])
        predicate_label = "predicate" if predicate_count == 1 else "predicates"
        lines.append(
            f"- {concept['Concept']}: "
            f"{predicate_count} {predicate_label}; "
            f"features: {concept['Distinct features']}"
            f"{metric_text}"
        )
        wrapped = textwrap.wrap(representatives, width=88)
        if wrapped:
            lines.append(f"  Representative predicates: {wrapped[0]}")
            for continuation in wrapped[1:]:
                lines.append(f"                             {continuation}")


def build_interpretation_note(
    dominant: str,
    safe_score: float,
    risky_score: float,
    concepts: pd.DataFrame,
) -> list[str]:
    """
    Build a cautious interpretation for a class-aware DPG community.

    Parameters
    ----------
    dominant : str
        Dominant class assigned by the class-aware community analysis.
    safe_score : float
        Normalized SAFE association score.
    risky_score : float
        Normalized RISKY association score.
    concepts : pandas.DataFrame
        Ranked concept rows for the community.

    Returns
    -------
    list[str]
        Paragraphs suitable for an interpretation report.
    """
    concept_names = concepts["Concept"].astype(str).tolist()
    concept_text = ", ".join(concept_names[:6]) if concept_names else "no mapped concepts"

    if dominant == "RISKY" and risky_score >= 0.80:
        lead = (
            "This community appears to capture a compact risky region of the "
            "DPG. Its association with RISKY is strong, and the dominant "
            "concept families suggest that the model combines contract "
            "complexity, contract coverage, structural information and "
            "risk-sensitive program constructs."
        )
    elif dominant == "SAFE" and risky_score >= 0.25:
        lead = (
            "This community appears broader and more heterogeneous. It is "
            "mostly associated with SAFE decisions, but it still contains a "
            "non-negligible RISKY component, suggesting that SAFE is not "
            "represented by a single simple structural pattern."
        )
    elif dominant == "SAFE":
        lead = (
            "This community is mostly associated with SAFE decisions. Its "
            "concept families should still be interpreted as combinations of "
            "predicates along decision paths, not as isolated indicators of "
            "software correctness."
        )
    else:
        lead = (
            "This community has no clear class-specific interpretation from "
            "the current association scores alone. Its predicates should be "
            "studied through local paths before drawing stronger conclusions."
        )

    caution = (
        "The main concept families are: "
        f"{concept_text}. These concepts are descriptive associations derived "
        "from DPG predicates; they do not imply causality by themselves."
    )
    path_note = (
        "A robust interpretation should therefore compare this global "
        "community-level view with SAFE and RISKY local paths, where the same "
        "predicates can be checked as concrete decision combinations."
    )
    return [lead, caution, path_note]


def write_interpretation_summary(
    path: Path,
    community_summary: pd.DataFrame,
    concept_summary: pd.DataFrame,
    top_concepts: int,
) -> Path:
    """
    Write a compact interpretation summary for project discussion.

    Parameters
    ----------
    path : Path
        Output text path.
    community_summary : pandas.DataFrame
        Class-aware DPG community summary.
    concept_summary : pandas.DataFrame
        Concept aggregation table.
    top_concepts : int
        Maximum number of concept families shown per community.

    Returns
    -------
    Path
        Written text path.
    """
    lines = [
        "SpecLens-PML DPG community interpretation summary",
        "==================================================",
        "",
        "This report provides a compact, project-oriented reading of the DPG",
        "communities. It combines class association scores with the deterministic",
        "predicate-to-concept mapping. The interpretation is descriptive and",
        "should be validated against local SAFE and RISKY paths.",
        "",
    ]

    for _, community in community_summary.iterrows():
        community_id = community["Community"]
        display = str(community.get("Display label", f"Community {community_id}"))
        dominant = str(community.get("Dominant class", "n/a"))
        safe_score = safe_float(community.get("SAFE association score", 0.0))
        risky_score = safe_float(community.get("RISKY association score", 0.0))
        safe_weight = safe_float(community.get("SAFE edge weight", 0.0))
        risky_weight = safe_float(community.get("RISKY edge weight", 0.0))
        predicate_count = int(safe_float(community.get("Predicate count", 0)))
        concepts = top_concepts_for_community(
            concept_summary,
            community_id,
            top_concepts,
        )
        central_concepts = top_central_concepts_for_community(
            concept_summary,
            community_id,
            top_concepts,
        )

        lines.extend(
            [
                display,
                "-" * len(display),
                f"Class leaf association: {community_id}",
                f"Dominant class: {dominant}",
                f"Predicates: {predicate_count}",
                "Class association: "
                f"SAFE {format_percentage(safe_score)} "
                f"(w={safe_weight:g}) | "
                f"RISKY {format_percentage(risky_score)} "
                f"(w={risky_weight:g})",
            ]
        )

        append_concept_family_lines(
            lines,
            concepts,
            "Concept families by predicate count:",
            include_centrality=False,
        )
        append_concept_family_lines(
            lines,
            central_concepts,
            "Concept families by structural centrality:",
            include_centrality=True,
        )

        lines.extend(["", "Interpretation:"])
        for paragraph in build_interpretation_note(
            dominant,
            safe_score,
            risky_score,
            concepts,
        ):
            wrapped = textwrap.wrap(paragraph, width=88)
            lines.extend(wrapped)
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Local path concept analysis
# ---------------------------------------------------------------------------

def extract_predicates_from_row(row: pd.Series) -> list[str]:
    """
    Extract predicate-like substrings from a local-path CSV row.

    Parameters
    ----------
    row : pandas.Series
        One row from a local explanation artifact.

    Returns
    -------
    list[str]
        Predicate strings found in textual cells.
    """
    predicates: list[str] = []
    for value in row.astype(str):
        for match in PREDICATE_RE.finditer(value):
            predicates.append(match.group(0))
    return predicates


def analyze_local_paths(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize concept families appearing in SAFE and RISKY local paths.

    Parameters
    ----------
    output_dir : Path
        DPG output directory containing local path CSV files.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Detailed predicate occurrences and aggregated concept counts. Empty
        tables are returned when local path files are missing.
    """
    detail_rows: list[dict[str, Any]] = []
    for local_label, filename in [
        ("SAFE", "local_safe_paths.csv"),
        ("RISKY", "local_risky_paths.csv"),
    ]:
        path = output_dir / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        for row_index, row in table.iterrows():
            for predicate in extract_predicates_from_row(row):
                interpretation = interpret_predicate(predicate)
                detail_rows.append(
                    {
                        "Local explanation": local_label,
                        "Row": row_index,
                        "Predicate": predicate,
                        "Feature": interpretation.feature,
                        "Concept": interpretation.concept,
                        "Polarity": interpretation.polarity,
                        "Interpretation": interpretation.interpretation,
                    }
                )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()

    summary = (
        detail.groupby(["Local explanation", "Concept"], sort=False)
        .agg(
            Occurrences=("Predicate", "count"),
            Features=("Feature", lambda values: top_joined_values(values, 10)),
            Examples=("Predicate", lambda values: top_joined_values(values, 8)),
        )
        .reset_index()
        .sort_values(
            ["Local explanation", "Occurrences", "Concept"],
            ascending=[True, False, True],
        )
    )
    return detail, summary


def write_local_path_report(path: Path, local_summary: pd.DataFrame) -> Path:
    """
    Write a short report comparing concept occurrence in local paths.

    Parameters
    ----------
    path : Path
        Output text path.
    local_summary : pandas.DataFrame
        Aggregated concept occurrence table for local paths.

    Returns
    -------
    Path
        Written text path.
    """
    lines = [
        "SpecLens-PML DPG local-path concept summary",
        "============================================",
        "",
        "This report counts concept families extracted from representative local",
        "SAFE and RISKY DPG paths. Counts are occurrences in local path artifacts,",
        "not global feature importances.",
        "",
    ]
    if local_summary.empty:
        lines.append("No local path concepts were available.")
    else:
        for label, group in local_summary.groupby("Local explanation", sort=False):
            lines.append(f"{label} local paths")
            lines.append("-" * (len(str(label)) + 12))
            for _, row in group.head(10).iterrows():
                lines.append(
                    f"- {row['Concept']}: {int(row['Occurrences'])} occurrences"
                )
                lines.append(f"  Examples: {row['Examples']}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_analysis(
    output_dir: Path,
    top_concepts: int,
    top_predicates_per_concept: int,
) -> list[Path]:
    """
    Run the concept-level analysis over existing DPG artifacts.

    Parameters
    ----------
    output_dir : Path
        Directory containing DPG outputs.
    top_concepts : int
        Number of concept families shown per community in the text report.
    top_predicates_per_concept : int
        Number of representative predicates shown per concept family.

    Returns
    -------
    list[Path]
        Paths written by the analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    community_summary, community_predicates = load_community_artifacts(output_dir)

    enriched = enrich_predicates_with_concepts(community_predicates)
    concept_summary = build_community_concept_summary(
        community_summary,
        enriched,
        top_predicates_per_concept,
    )

    outputs: list[Path] = []

    enriched_path = output_dir / "community_concept_predicates.csv"
    enriched.to_csv(enriched_path, index=False)
    outputs.append(enriched_path)

    concept_summary_path = output_dir / "community_concept_summary.csv"
    concept_summary.to_csv(concept_summary_path, index=False)
    outputs.append(concept_summary_path)

    report_path = write_community_concept_report(
        output_dir / "community_concept_summary.txt",
        community_summary,
        concept_summary,
        top_concepts,
    )
    outputs.append(report_path)

    interpretation_path = write_interpretation_summary(
        output_dir / "community_interpretation_summary.txt",
        community_summary,
        concept_summary,
        min(top_concepts, DEFAULT_INTERPRETATION_TOP_CONCEPTS),
    )
    outputs.append(interpretation_path)

    taxonomy_path = write_concept_taxonomy(output_dir / "concept_taxonomy.csv")
    outputs.append(taxonomy_path)

    local_detail, local_summary = analyze_local_paths(output_dir)
    if not local_detail.empty:
        local_detail_path = output_dir / "local_path_concept_predicates.csv"
        local_detail.to_csv(local_detail_path, index=False)
        outputs.append(local_detail_path)

        local_summary_path = output_dir / "local_path_concept_summary.csv"
        local_summary.to_csv(local_summary_path, index=False)
        outputs.append(local_summary_path)

        local_report_path = write_local_path_report(
            output_dir / "local_path_concept_summary.txt",
            local_summary,
        )
        outputs.append(local_report_path)

    unmapped = sorted(
        enriched.loc[~enriched["Mapped concept"], "Feature"].dropna().unique()
    )
    if unmapped:
        print("Unmapped features found:")
        for feature in unmapped:
            print(f"- {feature}")
        print("Consider extending CONCEPT_RULES before relying on the report.")

    print("Concept-level community analysis completed.")
    print(f"Communities analyzed: {community_summary['Community'].nunique()}")
    print(f"Predicate rows analyzed: {len(enriched)}")
    print(f"Concept families found: {concept_summary['Concept'].nunique()}")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for concept-level analysis.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Map SpecLens-PML DPG community predicates to deterministic "
            "software-engineering concepts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing DPG output artifacts.",
    )
    parser.add_argument(
        "--top-concepts",
        type=int,
        default=DEFAULT_TOP_CONCEPTS,
        help="Number of concept families shown per community in the text report.",
    )
    parser.add_argument(
        "--top-predicates-per-concept",
        type=int,
        default=DEFAULT_TOP_PREDICATES_PER_CONCEPT,
        help="Number of representative predicates shown for each concept family.",
    )
    return parser


def main() -> None:
    """Run the CLI entry point for concept-level DPG analysis."""
    args = build_arg_parser().parse_args()
    if args.top_concepts <= 0:
        raise ValueError("--top-concepts must be positive.")
    if args.top_predicates_per_concept <= 0:
        raise ValueError("--top-predicates-per-concept must be positive.")

    outputs = run_analysis(
        args.output_dir,
        args.top_concepts,
        args.top_predicates_per_concept,
    )
    print("\nSaved artifacts:")
    for path in outputs:
        try:
            print(f"- {path.relative_to(ROOT)}")
        except ValueError:
            print(f"- {path}")


if __name__ == "__main__":
    main()
