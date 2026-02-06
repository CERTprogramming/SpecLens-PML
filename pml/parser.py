"""
PML parsing module for SpecLens-PML.

This module implements the static analysis front-end of the project.

It combines:

- Python AST parsing (via the built-in ast module)
- Lightweight contract extraction from comments

Supported PML annotations are embedded as Python comments:

    # @requires <expr>
    # @ensures  <expr>
    # @invariant <expr>

The parser extracts, for each function or method:

- signature information (name, parameters, location)
- associated preconditions and postconditions
- inherited class invariants (if defined)

This component enables treating annotated programs as structured data,
which is the foundation of the data-driven MLOps pipeline.
"""

# pml/parser.py

import ast
import re
from pathlib import Path
from typing import List, Dict

# Regular expression pattern matching PML contract clauses in comments
PML_PATTERN = re.compile(r"#\s*@(?P<kind>requires|ensures|invariant)\s+(?P<expr>.+)")


def extract_pml_from_lines(lines: List[str]) -> Dict[str, List[str]]:
    """
    Extract PML contract clauses from a list of source code lines.

    Each line is scanned for annotations of the form:

        # @requires ...
        # @ensures ...
        # @invariant ...

    The result is grouped into three categories.

    :param lines: List of source code lines.
    :return: Dictionary containing lists of extracted expressions.
    """
    # Initialize clause containers
    clauses = {"requires": [], "ensures": [], "invariant": []}

    # Scan each line for contract annotations
    for line in lines:
        m = PML_PATTERN.search(line)
        if m:
            kind = m.group("kind")
            expr = m.group("expr").strip()

            # Append extracted expression under its clause type
            clauses[kind].append(expr)

    return clauses


def collect_leading_comments(lines, start_line):
    """
    Collect contiguous comment lines located immediately above a node.

    This is used to associate contract clauses written before a function
    or class definition.

    :param lines: Full source file lines.
    :param start_line: Line index (0-based) where the node begins.
    :return: List of comment lines above the node.
    """
    collected = []
    i = start_line - 1

    # Traverse upward while comments continue
    while i >= 0 and lines[i].strip().startswith("#"):
        collected.append(lines[i])
        i -= 1

    # Restore original ordering
    collected.reverse()
    return collected


def collect_body_comments(lines, node):
    """
    Collect comment lines at the beginning of a node body.

    This supports contracts written inside a class or function block,
    immediately before the first executable statement.

    :param lines: Full source file lines.
    :param node: AST node (FunctionDef or ClassDef).
    :return: List of leading comment lines inside the node body.
    """
    comments = []

    # If node has no body, nothing to extract
    if not node.body:
        return comments

    # Identify the first statement in the node body
    first_stmt = node.body[0]
    start = first_stmt.lineno - 2  # 0-based index of line before statement

    i = start

    # Traverse upward while contiguous comments exist
    while i >= 0 and lines[i].strip().startswith("#"):
        comments.append(lines[i])
        i -= 1

    # Restore original ordering
    comments.reverse()
    return comments


def parse_file(path: Path):
    """
    Parse a Python source file annotated with PML contracts.

    The parsing process performs two passes:

    1. Collect class invariants (@invariant) defined in classes.
    2. Collect functions and methods, attaching:
       - requires clauses
       - ensures clauses
       - inherited invariants from enclosing classes

    The output is a list of dictionaries, each representing one
    analyzable unit.

    :param path: Path to the Python source file.
    :return: List of extracted function/method metadata dictionaries.
    """
    # Load source file contents
    source = path.read_text()
    lines = source.splitlines()

    # Parse into an abstract syntax tree
    tree = ast.parse(source)

    functions = []

    # Map class names to their invariant clauses
    class_invariants = {}

    # --- First pass: collect class invariants ------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Extract comments above and inside the class definition
            start_line = node.lineno - 1
            above = collect_leading_comments(lines, start_line)
            inside = collect_body_comments(lines, node)

            # Extract invariant clauses from those comments
            clauses = extract_pml_from_lines(above + inside)
            class_invariants[node.name] = clauses["invariant"]

    # --- Second pass: collect functions and methods ------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Identify function start line for comment association
            start_line = node.lineno - 1

            # Extract comments above and inside the function definition
            above = collect_leading_comments(lines, start_line)
            inside = collect_body_comments(lines, node)

            # Extract requires/ensures clauses
            clauses = extract_pml_from_lines(above + inside)

            # --- Determine enclosing class context -------------------------
            parent_class = None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    if node in parent.body:
                        parent_class = parent.name
                        break

            # Attach invariants inherited from enclosing class (if any)
            invariants = []
            if parent_class and parent_class in class_invariants:
                invariants.extend(class_invariants[parent_class])

            # Build structured representation of the function unit
            func_info = {
                "name": node.name,
                "class": parent_class,
                "params": [arg.arg for arg in node.args.args],
                "lineno": node.lineno,
                "requires": clauses["requires"],
                "ensures": clauses["ensures"],
                "invariant": invariants,
                "n_loc": len(node.body),
            }

            # Add extracted unit to result list
            functions.append(func_info)

    return functions


# Standalone CLI entry point for debugging the parser
if __name__ == "__main__":
    import sys

    # Validate command-line usage
    if len(sys.argv) != 2:
        print("Usage: python parser.py <file.py>")
        sys.exit(1)

    # Parse file passed as argument
    path = Path(sys.argv[1])
    parsed = parse_file(path)

    # Print extracted units for inspection
    for f in parsed:
        print(f)
