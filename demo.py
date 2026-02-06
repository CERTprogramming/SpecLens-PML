# demo.py

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # make project root importable

RAW_DIR = ROOT / "data" / "raw"
DATASET = ROOT / "data" / "datasets_v1.csv"


def run(cmd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Command failed, stopping demo.")
        sys.exit(1)


def main():
    print("=== SpecLens-PML Demo ===")

    run([
        sys.executable,
        str(ROOT / "pipeline" / "build_dataset.py"),
        str(RAW_DIR),
        str(DATASET),
    ])

    run([
        sys.executable,
        str(ROOT / "pipeline" / "train.py"),
        str(DATASET),
    ])

    for py_file in RAW_DIR.glob("*.py"):
        run([
            sys.executable,
            str(ROOT / "inference" / "predict.py"),
            str(py_file),
        ])

    print("\n=== Demo completed successfully ===")


if __name__ == "__main__":
    main()
