"""
SpecLens-PML Contract Parser.

This module implements a lightweight parser for extracting functions and
methods annotated with PML-style contracts.

Supported annotations
---------------------
Contracts are expressed as Python comments:

    # @requires  <expr>
    # @ensures   <expr>
    # @invariant <expr>
    # @snapshot  <name> = <expr>

The expression language also supports ``old(...)`` inside postconditions
to refer to values captured in the pre-state of the current function call.
Named snapshots provide a readable way to bind pre-state expressions for
later use in postconditions.

Contracts may appear:

1. Immediately above a function or method definition
2. Anywhere inside the function body (after docstrings, comments, or code)

This design makes the parser robust across all demo examples.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ast


# ---------------------------------------------------------------------------
# Contract Extraction Helpers
# ---------------------------------------------------------------------------

def _parse_snapshot(payload: str) -> Optional[Tuple[str, str]]:
    """
    Parse a lightweight ``@snapshot name = expr`` annotation.

    Invalid snapshot declarations are ignored by the parser. If a later
    contract references the missing name, normal contract evaluation treats
    that unresolved reference as a failed expression.
    """
    body = payload[len("@snapshot"):].strip()
    if "=" not in body:
        return None

    name, expr = body.split("=", 1)
    name = name.strip()
    expr = expr.strip()

    if not name.isidentifier() or not expr:
        return None

    return name, expr


def _extract_contracts(lines: List[str]) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """
    Extract all contract annotations from a list of comment lines.

    Parameters
    ----------
    lines : list[str]
        Comment lines potentially containing PML annotation tags.

    Returns
    -------
    (requires, ensures, invariants, snapshots) : tuple[list[str], list[str], list[str], dict[str, str]]
        Extracted contract clauses.
    """

    requires: List[str] = []
    ensures: List[str] = []
    invariants: List[str] = []
    snapshots: Dict[str, str] = {}

    for raw in lines:
        line = raw.strip()

        if not line.startswith("#"):
            continue

        payload = line[1:].strip()

        if payload.startswith("@requires"):
            requires.append(payload[len("@requires"):].strip())

        elif payload.startswith("@ensures"):
            ensures.append(payload[len("@ensures"):].strip())

        elif payload.startswith("@invariant"):
            invariants.append(payload[len("@invariant"):].strip())

        elif payload.startswith("@snapshot"):
            snapshot = _parse_snapshot(payload)
            if snapshot is not None:
                name, expr = snapshot
                snapshots[name] = expr

    return requires, ensures, invariants, snapshots


# ---------------------------------------------------------------------------
# Comment Block Utilities
# ---------------------------------------------------------------------------

def _comment_block_above(lines: List[str], lineno: int) -> List[str]:
    """
    Collect contiguous comment lines immediately above a definition.

    This captures contracts written directly before a function / class header.

    Parameters
    ----------
    lines : list[str]
        Full source file split into lines.
    lineno : int
        AST line number where the definition starts.

    Returns
    -------
    list[str]
        The contiguous block of comment lines above the definition.
    """

    i = lineno - 2
    block: List[str] = []

    while i >= 0 and lines[i].strip() == "":
        i -= 1

    while i >= 0 and lines[i].lstrip().startswith("#"):
        block.append(lines[i])
        i -= 1

    block.reverse()
    return block


# ---------------------------------------------------------------------------
# In-Function Comment Scanner
# ---------------------------------------------------------------------------

def _all_comments_inside_function(lines: List[str], node: ast.FunctionDef) -> List[str]:
    """
    Collect all comment lines inside a function body.

    SpecLens examples may place contracts after docstrings or executable code,
    so scanning the full body is the most robust strategy.

    Parameters
    ----------
    lines : list[str]
        Full source file split into lines.
    node : ast.FunctionDef
        Function node.

    Returns
    -------
    list[str]
        Comment lines found inside the function body.
    """

    start = node.lineno - 1
    end = getattr(node, "end_lineno", start)

    body_lines = lines[start:end]
    return [l for l in body_lines if l.strip().startswith("#")]


def _all_comments_inside_class(lines: List[str], node: ast.ClassDef) -> List[str]:
    """
    Collect class-level comment lines without descending into methods.

    This captures invariants written as the first statements inside a class
    body, which is the style used by the demo examples.
    """
    start = node.lineno - 1
    end = getattr(node, "end_lineno", start)
    method_lines: set[int] = set()

    for child in node.body:
        if isinstance(child, ast.FunctionDef):
            child_start = child.lineno - 1
            child_end = getattr(child, "end_lineno", child_start)
            method_lines.update(range(child_start, child_end))

    comments: List[str] = []
    for idx in range(start, end):
        if idx in method_lines:
            continue
        if lines[idx].strip().startswith("#"):
            comments.append(lines[idx])

    return comments


# ---------------------------------------------------------------------------
# Helper: Lines of Code (LOC)
# ---------------------------------------------------------------------------

def _node_loc(node: ast.AST) -> int:
    """
    Approximate the number of lines of code (LOC) of an AST node.

    Uses the AST attribute end_lineno (available in Python ≥ 3.8)
    to estimate the full size of the function in lines of code.

    Returns
    -------
    int
        Approximate LOC for the node.
    """

    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)

    if lineno and end_lineno:
        return max(1, end_lineno - lineno + 1)

    return 1


def _structural_features(node: ast.FunctionDef) -> Dict[str, int]:
    """Extract simple AST-based implementation structure metrics."""
    nodes = list(ast.walk(node))

    n_branches = sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in nodes)
    n_loops = sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in nodes)
    n_returns = sum(isinstance(n, ast.Return) for n in nodes)

    has_division = any(
        isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod))
        for n in nodes
    )
    has_mutation = any(
        isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete))
        for n in nodes
    )
    has_method_call = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        for n in nodes
    )

    return {
        "n_branches": int(n_branches),
        "n_loops": int(n_loops),
        "n_returns": int(n_returns),
        "has_subscript": int(any(isinstance(n, ast.Subscript) for n in nodes)),
        "has_division": int(has_division),
        "has_mutation": int(has_mutation),
        "has_method_call": int(has_method_call),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> List[Dict]:
    """
    Parse a Python source file and extract all annotated functions / methods.

    Each extracted entry includes:

    - name        : function/method name
    - class       : enclosing class name (or None)
    - params      : parameter names
    - requires    : list of preconditions
    - ensures     : list of postconditions
    - invariant   : list of class invariants (if any)
    - snapshots   : dict mapping snapshot names to pre-state expressions
    - line        : definition line number
    - n_loc       : approximate LOC

    Parameters
    ----------
    path : Path
        Path to the Python source file.

    Returns
    -------
    list[dict]
        Parsed descriptors for all functions and methods.
    """

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)

    results: List[Dict] = []

    for node in tree.body:

        if isinstance(node, ast.FunctionDef):
            above = _comment_block_above(lines, node.lineno)
            inside = _all_comments_inside_function(lines, node)

            req1, ens1, _, snap1 = _extract_contracts(above)
            req2, ens2, _, snap2 = _extract_contracts(inside)
            snapshots = {**snap1, **snap2}

            results.append({
                "name": node.name,
                "class": None,
                "params": [a.arg for a in node.args.args],
                "requires": req1 + req2,
                "ensures": ens1 + ens2,
                "invariant": [],
                "snapshots": snapshots,
                "line": node.lineno,
                "n_loc": _node_loc(node),
                "structural": _structural_features(node),
            })

        elif isinstance(node, ast.ClassDef):
            above_class = _comment_block_above(lines, node.lineno)
            inside_class = _all_comments_inside_class(lines, node)
            _, _, class_invs1, class_snapshots1 = _extract_contracts(above_class)
            _, _, class_invs2, class_snapshots2 = _extract_contracts(inside_class)
            class_invs = class_invs1 + class_invs2
            class_snapshots = {**class_snapshots1, **class_snapshots2}

            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue

                above = _comment_block_above(lines, child.lineno)
                inside = _all_comments_inside_function(lines, child)

                req1, ens1, _, snap1 = _extract_contracts(above)
                req2, ens2, _, snap2 = _extract_contracts(inside)
                snapshots = {**class_snapshots, **snap1, **snap2}

                results.append({
                    "name": child.name,
                    "class": node.name,
                    "params": [a.arg for a in child.args.args],
                    "requires": req1 + req2,
                    "ensures": ens1 + ens2,
                    "invariant": class_invs,
                    "snapshots": snapshots,
                    "line": child.lineno,
                    "n_loc": _node_loc(child),
                    "structural": _structural_features(child),
                })

    return results
