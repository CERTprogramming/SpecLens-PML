# pml/parser.py

import ast
import re
from pathlib import Path
from typing import List, Dict

PML_PATTERN = re.compile(r"#\s*@(?P<kind>requires|ensures|invariant)\s+(?P<expr>.+)")

def extract_pml_from_lines(lines: List[str]) -> Dict[str, List[str]]:
    clauses = {"requires": [], "ensures": [], "invariant": []}
    for line in lines:
        m = PML_PATTERN.search(line)
        if m:
            kind = m.group("kind")
            expr = m.group("expr").strip()
            clauses[kind].append(expr)
    return clauses


def collect_leading_comments(lines, start_line):
    """Collect contiguous comment lines above start_line (0-based)."""
    collected = []
    i = start_line - 1
    while i >= 0 and lines[i].strip().startswith("#"):
        collected.append(lines[i])
        i -= 1
    collected.reverse()
    return collected


def collect_body_comments(lines, node):
    """Collect comment lines at the beginning of a node body."""
    comments = []
    if not node.body:
        return comments

    first_stmt = node.body[0]
    start = first_stmt.lineno - 2  # 0-based, line before first statement

    i = start
    while i >= 0 and lines[i].strip().startswith("#"):
        comments.append(lines[i])
        i -= 1

    comments.reverse()
    return comments


def parse_file(path: Path):
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)

    functions = []

    class_invariants = {}

    # First pass: collect class invariants
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start_line = node.lineno - 1
            above = collect_leading_comments(lines, start_line)
            inside = collect_body_comments(lines, node)

            clauses = extract_pml_from_lines(above + inside)
            class_invariants[node.name] = clauses["invariant"]

    # Second pass: functions and methods
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1

            above = collect_leading_comments(lines, start_line)
            inside = collect_body_comments(lines, node)

            clauses = extract_pml_from_lines(above + inside)

            # Determine enclosing class (if any)
            parent_class = None
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    if node in parent.body:
                        parent_class = parent.name
                        break

            invariants = []
            if parent_class and parent_class in class_invariants:
                invariants.extend(class_invariants[parent_class])

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

            functions.append(func_info)

    return functions


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parser.py <file.py>")
        sys.exit(1)

    path = Path(sys.argv[1])
    parsed = parse_file(path)

    for f in parsed:
        print(f)
