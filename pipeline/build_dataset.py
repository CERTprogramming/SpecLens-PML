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

from pathlib import Path
from pipeline.features import extract_features
from pml.parser import parse_file

import importlib.util
import pandas as pd
import random
import sys


# ---------------------------------------------------------------------------
# Module Loading Helper
# ---------------------------------------------------------------------------

def load_module(path: Path):
    """
    Dynamically import a Python source file as a module.

    Parameters
    ----------
    path : Path
        Path to the Python source file.

    Returns
    -------
    module
        Imported Python module object.
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
    Evaluate a boolean PML contract expression.

    The expression is evaluated in a restricted environment containing
    only the variables in ``env``. If evaluation fails due to syntax
    errors or runtime exceptions, the expression is treated as False.

    Parameters
    ----------
    expr : str
        Boolean expression extracted from a PML contract.
    env : dict
        Environment mapping variable names to runtime values.

    Returns
    -------
    bool
        True if the expression holds, False otherwise.
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
    Generate a randomized argument value for lightweight fuzzing.

    The generator uses simple heuristics based on parameter names.

    Supported cases include:

    - integers (default)
    - strings (``s``, ``text``, ``name``)
    - lists (``lst``, ``values``, ``items``)
    - Account-like objects (``other``)

    Parameters
    ----------
    param_name : str
        Name of the function parameter.
    obj : optional
        Instance of the enclosing object (for methods).

    Returns
    -------
    object
        Randomly generated argument value.
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
    Assign a supervised label to a function via dynamic execution.

    The function is executed multiple times with randomized inputs.
    If any runtime failure or contract violation is observed, the
    function is labeled as RISKY.

    Labels
    ------
    0
        SAFE (no observed violations)
    1
        RISKY (runtime error or contract violation)

    Parameters
    ----------
    func_info : dict
        Parsed function metadata produced by the PML parser.
    module : module
        Imported module containing the executable function.
    trials : int, default=20
        Number of randomized execution attempts.

    Returns
    -------
    int
        Supervised label (0 SAFE, 1 RISKY).
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
    Build a labeled dataset from annotated Python programs.

    The builder scans a directory of Python files, extracts contract-
    annotated functions, computes feature vectors, dynamically labels
    them as SAFE/RISKY, and writes the resulting dataset to CSV.

    Parameters
    ----------
    raw_dir : Path
        Directory containing annotated Python examples.
    out_path : Path
        Output path for the generated dataset CSV file.

    Returns
    -------
    pandas.DataFrame
        Generated dataset containing features, labels, and source file
        information.
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

