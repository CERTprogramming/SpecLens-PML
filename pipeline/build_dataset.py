# pipeline/build_dataset.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import importlib.util
import pandas as pd
import random

from pml.parser import parse_file


def extract_features(func_info):
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
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def eval_expr(expr: str, env: dict):
    try:
        return bool(eval(expr, {}, env))
    except Exception:
        return False


def generate_inputs(n):
    # very simple generator for demo purposes
    return [random.randint(-5, 5) for _ in range(n)]


def label_function(func_info, module, trials=20):
    func = getattr(module, func_info["name"], None)
    if func is None:
        return 0  # cannot test → assume safe

    for _ in range(trials):
        args = generate_inputs(len(func_info["params"]))
        env = dict(zip(func_info["params"], args))

        # check requires
        if any(not eval_expr(r, env) for r in func_info["requires"]):
            continue  # precondition not satisfied, skip case

        try:
            result = func(*args)
        except Exception:
            return 1  # runtime failure → violation

        env["result"] = result

        # check ensures
        for e in func_info["ensures"]:
            if not eval_expr(e, env):
                return 1

    return 0


def build_dataset(raw_dir: Path, out_path: Path):
    rows = []

    for py_file in raw_dir.glob("*.py"):
        module = load_module(py_file)
        functions = parse_file(py_file)

        for f in functions:
            feats = extract_features(f)
            label = label_function(f, module)
            feats["label"] = label
            feats["source_file"] = py_file.name
            rows.append(feats)

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_dataset.py <raw_dir> <output.csv>")
        sys.exit(1)

    raw_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    df = build_dataset(raw_dir, out_path)
    print("Dataset created:")
    print(df)
