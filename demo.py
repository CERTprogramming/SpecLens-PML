"""
End-to-end demo script for SpecLens-PML.

This module orchestrates the full MLOps workflow in a single execution:

1. Dataset generation from annotated Python source files
2. Model training and artifact versioning
3. Inference and risk scoring on all available raw examples

It is intended as the main entry point for quickly demonstrating the
complete pipeline lifecycle.
"""

from pathlib import Path

import subprocess
import sys

# Define the root directory of the repository
ROOT = Path(__file__).parent

# Define key input/output paths used across the pipeline
RAW_DIR = ROOT / "data" / "raw"
DATASET = ROOT / "data" / "datasets_v1.csv"

def run(cmd):
    """
    Execute a command as a subprocess and stop the demo on failure.

    This helper function provides consistent logging and error handling
    when running the different pipeline stages.

    :param cmd: List of command arguments to execute.
    """
    # Print the command being executed for traceability
    print("\n>>>", " ".join(cmd))

    # Run the subprocess command
    result = subprocess.run(cmd)

    # Stop the pipeline immediately if any stage fails
    if result.returncode != 0:
        print("Command failed, stopping demo.")
        sys.exit(1)

def main():
    """
    Run the complete SpecLens-PML demo pipeline.

    This function sequentially executes dataset generation, training,
    and inference over all example Python files.
    """
    print("=== SpecLens-PML Demo ===")

    # --- Step 1: Dataset generation ----------------------------------------
    # Build a labeled dataset from PML-annotated Python files
    run([
        sys.executable,
        str(ROOT / "pipeline" / "build_dataset.py"),
        str(RAW_DIR),
        str(DATASET),
    ])

    # --- Step 2: Model training --------------------------------------------
    # Train a new versioned model artifact using the generated dataset
    run([
        sys.executable,
        str(ROOT / "pipeline" / "train.py"),
        str(DATASET),
    ])

    # --- Step 3: Inference -------------------------------------------------
    # Run risk prediction on each raw Python example in the dataset folder
    for py_file in RAW_DIR.glob("*.py"):
        run([
            sys.executable,
            str(ROOT / "inference" / "predict.py"),
            str(py_file),
        ])

    print("\n=== Demo completed successfully ===")


# Entry point for standalone execution
if __name__ == "__main__":
    main()
