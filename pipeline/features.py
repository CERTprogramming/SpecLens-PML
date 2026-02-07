"""
SpecLens-PML Feature Extraction.

This module defines the *single source of truth* for feature engineering.

Both dataset generation (build_dataset.py) and inference (predict.py)
must rely on the exact same feature extraction logic to ensure that:

- training and prediction schemas stay aligned
- models remain compatible across pipeline stages
- feature drift is avoided

All extracted features are simple numeric proxies derived from:

- function structure
- parameter patterns
- contract complexity
"""

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
    params = func_info["params"]

    return {
        # ----------------------------------------------------
        # Structural features
        # ----------------------------------------------------
        "n_params": len(params),
        "n_requires": len(requires),
        "n_ensures": len(ensures),

        # Class-level invariants (if any)
        "n_invariants": len(func_info.get("invariant", [])),

        # Lines of code proxy
        "n_loc": func_info["n_loc"],

        # ----------------------------------------------------
        # Parameter-level hints
        # ----------------------------------------------------
        "has_self": int(bool(params) and params[0] == "self"),
        "has_other": int("other" in params),

        # ----------------------------------------------------
        # Contract complexity proxies
        # ----------------------------------------------------
        "requires_complexity": sum(len(r) for r in requires),
        "ensures_complexity": sum(len(e) for e in ensures),

        # ----------------------------------------------------
        # Semantic contract patterns
        # ----------------------------------------------------

        # Arithmetic operators inside ensures clauses
        "ensures_has_arith": int(
            any(op in e for e in ensures for op in ["+", "-", "*", "/", "//"])
        ),

        # Comparison operators inside ensures clauses
        "ensures_has_cmp": int(
            any(op in e for e in ensures for op in [">", "<", "=="])
        ),
    }

