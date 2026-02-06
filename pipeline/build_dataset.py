"""
Dataset generation module for SpecLens-PML.

This script implements the data engineering stage of the pipeline.

It treats annotated Python programs as a source of structured data:

- Parses functions and methods annotated with PML contracts.
- Extracts simple structural features (parameters, contracts, LOC).
- Dynamically executes functions on generated inputs.
- Labels functions as SAFE or RISKY depending on observed violations.
- Produces a tabular dataset ready for supervised ML training.

This stage represents the "data-driven specification mining" component
of the overall MLOps lifecycle.
"""

from pathlib import Path
from pml.parser import parse_file

import importlib.util
import pandas as pd
import random
import sys

from pml.parser import parse_file


def extract_features(func_info):
    """
    Extract a numeric feature representation from parsed function metadata.

    Features are intentionally simple and interpretable, including:

    - number of parameters
    - number of requires clauses
    - number of ensures clauses
    - number of invariants
    - number of lines of code

    :param func_info: Dictionary describing a parsed function or method.
    :return: Feature dictionary suitable for ML training.
    """
    return {
        "name": func_info["name"],
        "class": func_info["class"] or "",
        "n_params": len(func_info["params"]),
        "n_requires": len(func_info["requires"]),
        "n_ensures": len(func_info["ensures"]),
        "n_invariants": len(func_info["invariant"]),
        "n_loc": func_info["n_loc"],
    }


def load_module(path: Path):
    """
    Dynamically import a Python source file as an executable module.

    This enables runtime execution of functions during dataset labeling.

    :param path: Path to a Python source file.
    :return: Imported module object.
    """
    # Build an import specification from the file location
    spec = importlib.util.spec_from_file_location(path.stem, path)

    # Create a new module instance
    module = importlib.util.module_from_spec(spec)

    # Execute the module code in memory
    spec.loader.exec_module(module)

    return module


def eval_expr(expr: str, env: dict):
    """
    Evaluate a PML boolean expression inside a given environment.

    Expressions are evaluated safely with restricted globals.

    If evaluation fails (e.g., runtime error), the expression is treated
    as False.

    :param expr: Contract expression as a string.
    :param env: Variable environment mapping parameter names to values.
    :return: Boolean evaluation result.
    """
    try:
        return bool(eval(expr, {}, env))
    except Exception:
        return False


def generate_inputs(n):
    """
    Generate random integer inputs for dynamic testing.

    This is a lightweight test generator used only for educational purposes.

    :param n: Number of arguments required.
    :return: List of randomly generated integers.
    """
    # very simple generator for demo purposes
    return [random.randint(-5, 5) for _ in range(n)]


def label_function(func_info, module, trials=20):
    """
    Assign a supervised label to a function by dynamic contract checking.

    The labeling procedure is:

    - Generate random inputs
    - Check preconditions (@requires)
    - Execute the function
    - Check postconditions (@ensures)
    - Mark as RISKY if a violation is observed

    Labels:

    - 0 → SAFE (no violations observed)
    - 1 → RISKY (contract violation or runtime failure)

    :param func_info: Parsed metadata for the function.
    :param module: Imported module containing the executable function.
    :param trials: Number of random executions attempted.
    :return: Integer label (0 safe, 1 risky).
    """
    # Retrieve the actual callable object from the module
    func = getattr(module, func_info["name"], None)

    # If the function cannot be found, assume safe by default
    if func is None:
        return 0  # cannot test → assume safe

    # Perform multiple randomized trials
    for _ in range(trials):
        # Generate candidate arguments
        args = generate_inputs(len(func_info["params"]))

        # Build evaluation environment for contracts
        env = dict(zip(func_info["params"], args))

        # --- Step 1: Check preconditions -----------------------------------
        if any(not eval_expr(r, env) for r in func_info["requires"]):
            continue  # precondition not satisfied, skip case

        # --- Step 2: Execute function --------------------------------------
        try:
            result = func(*args)
        except Exception:
            return 1  # runtime failure → violation

        # Add result into environment for postcondition evaluation
        env["result"] = result

        # --- Step 3: Check postconditions ----------------------------------
        for e in func_info["ensures"]:
            if not eval_expr(e, env):
                return 1

    # If no violations were found across trials, mark as safe
    return 0


def build_dataset(raw_dir: Path, out_path: Path):
    """
    Build a full dataset from a directory of annotated Python programs.

    For each file:

    - Parse contract-annotated functions
    - Extract features
    - Execute dynamic labeling
    - Store results into a CSV dataset

    :param raw_dir: Directory containing raw annotated Python files.
    :param out_path: Output CSV file path.
    :return: Pandas DataFrame containing the generated dataset.
    """
    rows = []

    # Iterate over all Python files in the raw directory
    for py_file in raw_dir.glob("*.py"):
        # Load module dynamically for execution
        module = load_module(py_file)

        # Parse functions and their PML contracts
        functions = parse_file(py_file)

        # Process each extracted function unit
        for f in functions:
            # Extract static structural features
            feats = extract_features(f)

            # Assign label through dynamic testing
            label = label_function(f, module)

            # Store dataset row metadata
            feats["label"] = label
            feats["source_file"] = py_file.name

            rows.append(feats)

    # Convert collected rows into a tabular dataset
    df = pd.DataFrame(rows)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist dataset artifact as CSV
    df.to_csv(out_path, index=False)

    return df


# Standalone CLI entry point
if __name__ == "__main__":
    # Validate command-line usage
    if len(sys.argv) != 3:
        print("Usage: python build_dataset.py <raw_dir> <output.csv>")
        sys.exit(1)

    # Read input and output paths from CLI
    raw_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    # Execute dataset generation pipeline step
    df = build_dataset(raw_dir, out_path)

    # Print dataset preview for transparency
    print("Dataset created:")
    print(df)
