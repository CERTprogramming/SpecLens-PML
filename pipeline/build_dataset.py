"""
SpecLens-PML Dataset Builder.

This module is part of the SpecLens demo pipeline:
it implements the dataset generation stage.

Annotated Python programs are treated as structured training data:

- Functions and methods are parsed from source files.
- PML contracts (@requires / @ensures / @invariant) are extracted.
- Structural and semantic features are computed.
- Functions are dynamically executed on generated inputs.
- Contract violations or runtime failures are labeled as RISKY.

The output is a supervised dataset ready for ML training.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pandas as pd

from pipeline.features import extract_features
from pml.parser import parse_file


# ---------------------------------------------------------------------------
# Module Loading Helper
# ---------------------------------------------------------------------------

def load_module(path: Path):
    """
    Dynamically import a Python source file as a module.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Contract Evaluation Helper
# ---------------------------------------------------------------------------

def eval_expr(expr: str, env: dict) -> bool:
    """
    Evaluate a boolean PML expression inside a restricted environment.

    If evaluation fails (syntax/runtime), the expression is treated as False.
    """
    try:
        return bool(eval(expr, {}, env))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Input Generation (Lightweight Fuzzing)
# ---------------------------------------------------------------------------

def generate_argument(param_name: str, obj=None):
    """
    Simple type-aware argument generator.

    Supports:
    - integers (default)
    - strings (s, text, name)
    - lists (lst, values, items)
    - Account-like objects (other)
    """

    # Object parameter (e.g., Account.transfer_to)
    if param_name == "other" and obj is not None:
        return obj.__class__(10)

    # String-like parameters
    if param_name in ("s", "text", "name"):
        return random.choice(["a", "hello", "XYZ"])

    # List-like parameters
    if param_name in ("lst", "values", "items"):
        return random.choice([[1, 2, 3], [0], [5, -1]])

    # Default: integer fuzzing
    return random.randint(-5, 5)


# ---------------------------------------------------------------------------
# Dynamic Labeling (SAFE vs RISKY)
# ---------------------------------------------------------------------------

def label_function(func_info, module, trials: int = 20) -> int:
    """
    Assign a supervised label by executing the function dynamically.

    Labels:
    - 0 → SAFE   (no observed contract violation)
    - 1 → RISKY  (runtime error or contract violation)
    """

    func = None
    obj = None

    # --------------------------------------------------------
    # Resolve function target (top-level or class method)
    # --------------------------------------------------------
    if func_info.get("class"):
        cls_name = func_info["class"]
        cls = getattr(module, cls_name, None)

        if cls is None:
            return 0

        try:
            obj = cls(10)
            func = getattr(obj, func_info["name"], None)
        except Exception:
            return 0
    else:
        func = getattr(module, func_info["name"], None)

    if func is None:
        return 0

    # Remove "self" from method parameters
    params = func_info["params"]
    if obj is not None and params and params[0] == "self":
        params = params[1:]

    # --------------------------------------------------------
    # Randomized execution trials
    # --------------------------------------------------------
    for _ in range(trials):

        # Generate fuzzed arguments
        args = [generate_argument(p, obj=obj) for p in params]

        # Build evaluation environment
        env = dict(zip(params, args))

        # Expose self for method contracts
        if obj is not None:
            env["self"] = obj

        # ----------------------------------------------------
        # Step 1: Check preconditions
        # ----------------------------------------------------
        if any(not eval_expr(r, env) for r in func_info["requires"]):
            continue

        # ----------------------------------------------------
        # Step 2: Execute function
        # ----------------------------------------------------
        try:
            result = func(*args)
        except Exception:
            return 1

        env["result"] = result

        # ----------------------------------------------------
        # Step 3: Check postconditions
        # ----------------------------------------------------
        for e in func_info["ensures"]:
            if not eval_expr(e, env):
                return 1

    return 0


# ---------------------------------------------------------------------------
# Dataset Construction
# ---------------------------------------------------------------------------

def build_dataset(raw_dir: Path, out_path: Path):
    """
    Build a full labeled dataset from annotated Python programs.
    """

    rows = []

    for py_file in raw_dir.glob("*.py"):

        module = load_module(py_file)
        functions = parse_file(py_file)

        for f in functions:

            # Skip Python special methods (__init__, __repr__, ...)
            if f["name"].startswith("__") and f["name"].endswith("__"):
                continue

            feats = extract_features(f)
            label = label_function(f, module)

            feats["label"] = label
            feats["source_file"] = py_file.name

            rows.append(feats)

    df = pd.DataFrame(rows)

    # Save dataset artifact
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_dataset.py <raw_dir> <output.csv>")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    df = build_dataset(raw_dir, out_path)

    print("Dataset created:")
    print(df)

