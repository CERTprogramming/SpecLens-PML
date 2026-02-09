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

Contracts may appear:

1. Immediately above a function or method definition
2. Anywhere inside the function body (after docstrings, comments, or code)

This design makes the parser robust across all SpecLens demo examples.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import ast

# ---------------------------------------------------------------------------
# Contract Extraction Helpers
# ---------------------------------------------------------------------------

def _extract_contracts(lines: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract all contract annotations from a list of comment lines.

    Parameters
    ----------
    lines : list[str]
        Comment lines potentially containing @requires/@ensures/@invariant tags.

    Returns
    -------
    (requires, ensures, invariants) : tuple[list[str], list[str], list[str]]
        Extracted contract clauses.
    """

    requires: List[str] = []
    ensures: List[str] = []
    invariants: List[str] = []

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

    return requires, ensures, invariants


# ---------------------------------------------------------------------------
# Comment Block Utilities
# ---------------------------------------------------------------------------

def _comment_block_above(lines: List[str], lineno: int) -> List[str]:
    """
    Collect contiguous comment lines immediately above a definition.

    This captures contracts written directly before a function/class header.

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

    # Skip blank lines immediately above
    while i >= 0 and lines[i].strip() == "":
        i -= 1

    # Collect comment lines
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
    Collect ALL comment lines inside a function body.

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


# ---------------------------------------------------------------------------
# LOC Helper
# ---------------------------------------------------------------------------

def _node_loc(node: ast.AST) -> int:
    """
    Approximate the number of lines of code (LOC) of an AST node.

    Uses end_lineno when available (Python 3.8+).

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_file(path: Path) -> List[Dict]:
    """
    Parse a Python source file and extract all annotated functions/methods.

    Each extracted entry includes:

    - name        : function/method name
    - class       : enclosing class name (or None)
    - params      : parameter names
    - requires    : list of preconditions
    - ensures     : list of postconditions
    - invariant   : list of class invariants (if any)
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

    # -----------------------------------------------------------------------
    # Traverse top-level AST nodes
    # -----------------------------------------------------------------------

    for node in tree.body:

        # -------------------------------------------------------------------
        # Top-level functions
        # -------------------------------------------------------------------

        if isinstance(node, ast.FunctionDef):

            above = _comment_block_above(lines, node.lineno)
            inside = _all_comments_inside_function(lines, node)

            req1, ens1, _ = _extract_contracts(above)
            req2, ens2, _ = _extract_contracts(inside)

            results.append({
                "name": node.name,
                "class": None,
                "params": [a.arg for a in node.args.args],
                "requires": req1 + req2,
                "ensures": ens1 + ens2,
                "invariant": [],
                "line": node.lineno,
                "n_loc": _node_loc(node),
            })

        # -------------------------------------------------------------------
        # Classes and methods
        # -------------------------------------------------------------------

        elif isinstance(node, ast.ClassDef):

            # Extract class-level invariants
            above_class = _comment_block_above(lines, node.lineno)
            _, _, class_invs = _extract_contracts(above_class)

            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue

                above = _comment_block_above(lines, child.lineno)
                inside = _all_comments_inside_function(lines, child)

                req1, ens1, _ = _extract_contracts(above)
                req2, ens2, _ = _extract_contracts(inside)

                results.append({
                    "name": child.name,
                    "class": node.name,
                    "params": [a.arg for a in child.args.args],
                    "requires": req1 + req2,
                    "ensures": ens1 + ens2,
                    "invariant": class_invs,
                    "line": child.lineno,
                    "n_loc": _node_loc(child),
                })

    return results

