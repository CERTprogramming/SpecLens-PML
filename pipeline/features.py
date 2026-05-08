"""
SpecLens-PML Feature Extraction.

This module defines the "single source of truth" for feature engineering.

Both dataset generation (build_dataset.py) and inference (predict.py)
must rely on the exact same feature extraction logic to ensure that:

- training and prediction schemas stay aligned
- models remain compatible across pipeline stages
- feature drift is avoided

All extracted features are simple numeric proxies derived from:

- function structure
- parameter patterns
- contract complexity
- pre-state references through ``old(...)`` and ``@snapshot``
"""

from __future__ import annotations

import ast
import re

import pandas as pd


# Metadata columns may be present in generated CSVs, but must never be passed
# to scikit-learn as model inputs.
METADATA_COLUMNS = frozenset({
    "name",
    "class",
    "source_file",
    "file",
    "function",
    "label",
})


# ---------------------------------------------------------------------------
# Feature Column Selection
# ---------------------------------------------------------------------------

def select_numeric_feature_columns(
    df: pd.DataFrame,
    metadata_cols: set[str] | frozenset[str] = METADATA_COLUMNS,
) -> list[str]:
    """Return numeric feature columns after excluding known metadata fields."""
    return [
        c for c in df.columns
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_feature_matrix(
    df: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Build a numeric feature matrix, optionally aligned to a trained schema.

    Missing feature columns are filled with zero so older and newer model
    artifacts can be handled predictably during inference and CT evaluation.
    """
    if feature_names is None:
        feature_names = select_numeric_feature_columns(df)

    X = df.reindex(columns=list(feature_names), fill_value=0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X


def get_model_feature_names(model) -> list[str] | None:
    """Return feature names stored on a trained model, when available."""
    if hasattr(model, "spec_lens_feature_names"):
        return list(model.spec_lens_feature_names)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


# ---------------------------------------------------------------------------
# Contract Pattern Helpers
# ---------------------------------------------------------------------------

def _parse_expr(expr: str) -> ast.AST | None:
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError:
        return None


def _expr_complexity(expr: str) -> int:
    """Use AST node count as a compact expression complexity proxy."""
    tree = _parse_expr(expr)
    if tree is None:
        return len(expr)
    return sum(1 for _ in ast.walk(tree))


def _name_used(expr: str, name: str) -> bool:
    tree = _parse_expr(expr)
    if tree is None:
        return name in expr
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(tree))


def _count_old_refs(expr: str) -> int:
    tree = _parse_expr(expr)
    if tree is None:
        return len(re.findall(r"\bold\s*\(", expr))
    return sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "old"
    )


def _has_self_reference(expr: str) -> bool:
    tree = _parse_expr(expr)
    if tree is None:
        return "self." in expr
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(tree)
    )


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(func_info: dict) -> dict:
    """
    Extract a numeric ML feature representation from parsed function metadata.

    This function must remain consistent across:

    - dataset generation
    - model training
    - inference

    Parameters
    ----------
    func_info : dict
        Parsed function metadata produced by
        :func:`pml.parser.parse_file`.

    Returns
    -------
    dict
        Dictionary of numeric features ready for ML models.
    """

    requires = func_info["requires"]
    ensures = func_info["ensures"]
    invariants = func_info.get("invariant", [])
    snapshots = func_info.get("snapshots", {})
    params = func_info["params"]
    all_contracts = requires + ensures + invariants
    structural = func_info.get("structural", {})

    n_old_refs = sum(_count_old_refs(contract) for contract in all_contracts)
    n_snapshots = len(snapshots)
    snapshot_names = set(snapshots)
    ensures_uses_snapshot = any(
        _name_used(expr, name)
        for expr in ensures
        for name in snapshot_names
    )
    has_prestate_reference = bool(n_old_refs or n_snapshots or ensures_uses_snapshot)

    return {
        "n_params": len(params),
        "n_requires": len(requires),
        "n_ensures": len(ensures),
        "n_invariants": len(invariants),
        "n_contracts_total": len(all_contracts),
        "n_loc": func_info["n_loc"],
        "has_self": int(bool(params) and params[0] == "self"),
        "has_other": int("other" in params),
        "requires_complexity": sum(_expr_complexity(r) for r in requires),
        "ensures_complexity": sum(_expr_complexity(e) for e in ensures),
        "invariants_complexity": sum(_expr_complexity(i) for i in invariants),
        "has_missing_requires": int(len(requires) == 0),
        "has_stateful_contract": int(any(_has_self_reference(c) for c in all_contracts)),
        "n_old_refs": n_old_refs,
        "ensures_has_arith": int(any(op in e for e in ensures for op in ["+", "-", "*", "/", "//"])),
        "ensures_has_cmp": int(any(op in e for e in ensures for op in [">", "<", "=="])),
        "ensures_has_old": int(any(_count_old_refs(e) for e in ensures)),
        "invariants_has_old": int(any(_count_old_refs(i) for i in invariants)),
        "invariants_has_cmp": int(any(op in inv for inv in invariants for op in [">", "<", "=="])),
        "n_snapshots": n_snapshots,
        "has_snapshot": int(n_snapshots > 0),
        "snapshot_complexity": sum(_expr_complexity(expr) for expr in snapshots.values()),
        "ensures_uses_snapshot": int(ensures_uses_snapshot),
        "contract_has_prestate_reference": int(has_prestate_reference),
        "has_prestate_reference": int(has_prestate_reference),
        "n_branches": structural.get("n_branches", 0),
        "n_loops": structural.get("n_loops", 0),
        "n_returns": structural.get("n_returns", 0),
        "has_subscript": structural.get("has_subscript", 0),
        "has_division": structural.get("has_division", 0),
        "has_mutation": structural.get("has_mutation", 0),
        "has_method_call": structural.get("has_method_call", 0),
    }
