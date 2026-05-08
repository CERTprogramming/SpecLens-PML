"""
SpecLens-PML Dataset Builder.

This module is part of the SpecLens demo pipeline:
it implements the dataset generation stage.

Annotated Python programs are treated as structured training data:

- Functions and methods are parsed from source files
- PML contracts (@requires / @ensures / @invariant) are extracted
- Structural and semantic features are computed
- Functions are dynamically executed on generated inputs
- Contract violations, snapshot-evaluation failures, or runtime failures are
  labeled as RISKY

The output is a supervised dataset ready for ML training.
"""

from __future__ import annotations

from pathlib import Path

import ast
import importlib.util
import pandas as pd
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.features import extract_features
from pml.parser import parse_file


# ---------------------------------------------------------------------------
# Helper: Module Loading
# ---------------------------------------------------------------------------

def load_module(path: Path):
    """Dynamically import a Python source file as a module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Expression Rewriting for old(...)
# ---------------------------------------------------------------------------

class _OldTransformer(ast.NodeTransformer):
    """Rewrite old(expr) calls into lookups on the captured pre-state."""

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "old" and len(node.args) == 1:
            inner = ast.unparse(node.args[0])
            return ast.Subscript(
                value=ast.Name(id="__old__", ctx=ast.Load()),
                slice=ast.Constant(value=inner),
                ctx=ast.Load(),
            )
        return node


def _rewrite_old_calls(expr: str) -> ast.AST:
    """Parse an expression and rewrite old(...) calls for evaluation."""
    tree = ast.parse(expr, mode="eval")
    tree = _OldTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def collect_old_expressions(expressions: list[str]) -> list[str]:
    """Return unique inner expressions referenced through old(...)."""
    found: list[str] = []
    for expr in expressions:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "old" and len(node.args) == 1:
                item = ast.unparse(node.args[0])
                if item not in found:
                    found.append(item)
    return found


def capture_old_environment(expressions: list[str], env: dict) -> dict:
    """Evaluate and store the pre-state values needed by old(...)."""
    snapshots: dict[str, object] = {}
    for inner in collect_old_expressions(expressions):
        try:
            snapshots[inner] = eval(inner, {}, env)
        except Exception:
            snapshots[inner] = None
    return snapshots


def capture_named_snapshots(snapshots: dict[str, str], env: dict) -> dict:
    """Evaluate ``@snapshot`` bindings in the pre-call environment."""
    values: dict[str, object] = {}
    snapshot_env = dict(env)

    for name, expr in snapshots.items():
        try:
            values[name] = eval(expr, {}, snapshot_env)
        except Exception:
            values[name] = None
        snapshot_env[name] = values[name]

    return values


# ---------------------------------------------------------------------------
# Helper: Contract Evaluation
# ---------------------------------------------------------------------------

def eval_expr(expr: str, env: dict) -> bool:
    """
    Evaluate a boolean PML contract expression.

    The expression is evaluated in a restricted environment containing
    only the variables in ``env``. ``old(...)`` and named ``@snapshot``
    values are supported in postconditions and invariants as references
    to the pre-state.
    If evaluation fails due to syntax errors or runtime exceptions,
    the expression is treated as False.
    """
    try:
        code = compile(_rewrite_old_calls(expr), "<pml>", "eval")
        return bool(eval(code, {}, env))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Input Generation (Lightweight Fuzzing)
# ---------------------------------------------------------------------------

def generate_argument(param_name: str, obj=None):
    """Generate a randomized argument value for lightweight fuzzing."""
    if param_name == "other" and obj is not None:
        return obj.__class__(10)
    if param_name in ("s", "text", "name"):
        return random.choice(["a", "hello", "XYZ"])
    if param_name in ("lst", "values", "items"):
        return random.choice([[1, 2, 3], [0], [5, -1]])
    return random.randint(-5, 5)


# ---------------------------------------------------------------------------
# Dynamic Labeling (SAFE vs RISKY)
# ---------------------------------------------------------------------------

def _build_instance_for_method(cls, method_name: str):
    """Best-effort instance construction for simple demo classes."""
    candidates = [10, 1, 5]
    signatures = [(10,), (1,), (0,), (5, 10), (0, 10), tuple()]
    for args in signatures:
        try:
            obj = cls(*args)
            if hasattr(obj, method_name):
                return obj
        except Exception:
            continue
    try:
        obj = cls(candidates[0])
        if hasattr(obj, method_name):
            return obj
    except Exception:
        return None
    return None


def label_function(func_info, module, trials: int = 20) -> int:
    """
    Assign a supervised label to a function via dynamic execution.

    The function is executed multiple times with randomized inputs.
    If any runtime failure or contract violation is observed, the
    function is labeled as RISKY.
    """

    func = None
    obj = None

    if func_info.get("class"):
        cls_name = func_info["class"]
        cls = getattr(module, cls_name, None)
        if cls is None:
            return 0
        obj = _build_instance_for_method(cls, func_info["name"])
        if obj is None:
            return 0
        func = getattr(obj, func_info["name"], None)
    else:
        func = getattr(module, func_info["name"], None)

    if func is None:
        return 0

    params = func_info["params"]
    if obj is not None and params and params[0] == "self":
        params = params[1:]

    post_state_contracts = list(func_info["ensures"]) + list(func_info.get("invariant", []))

    for _ in range(trials):
        args = [generate_argument(p, obj=obj) for p in params]
        env = dict(zip(params, args))
        if obj is not None:
            env["self"] = obj

        if any(not eval_expr(r, env) for r in func_info["requires"]):
            continue

        env["__old__"] = capture_old_environment(post_state_contracts, env)
        env.update(capture_named_snapshots(func_info.get("snapshots", {}), env))

        try:
            result = func(*args)
        except Exception:
            return 1

        env["result"] = result

        for e in func_info["ensures"]:
            if not eval_expr(e, env):
                return 1

        for inv in func_info.get("invariant", []):
            if not eval_expr(inv, env):
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
    """
    rows = []

    for py_file in sorted(raw_dir.glob("*.py")):
        mod = load_module(py_file)
        funcs = parse_file(py_file)

        for f in funcs:
            feats = extract_features(f)
            label = label_function(f, mod)
            feats.update({
                "file": py_file.name,
                "function": f["name"],
                "label": label,
            })
            rows.append(feats)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pipeline/build_dataset.py <raw_dir> <out.csv>")
        sys.exit(1)

    build_dataset(Path(sys.argv[1]), Path(sys.argv[2]))
