"""
SpecLens-PML Demo Pipeline (Continuous Learning).

This script runs a realistic MLOps-style workflow:

1. Build TRAIN dataset from:
      data/raw_train + data/raw_feedback

2. Build TEST dataset from:
      data/raw_test

3. Train multiple candidate models:
      logistic regression + random forest

4. Promote the best model using Recall on the RISKY class.

5. Run inference on UNSEEN examples:
      data/raw_unseen

6. If risky functions are detected, the file is added to the feedback pool:
      data/raw_feedback

This implements a simplified but realistic continuous learning loop.
"""

import subprocess
import sys
import shutil
from pathlib import Path


# ------------------------------------------------------------
# Helper: run a pipeline step safely
# ------------------------------------------------------------

def run_step(cmd):
    """
    Execute a pipeline command as a subprocess.

    Stops immediately if any step fails.
    """
    print(f"\n>>> {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Step failed, stopping demo.")
        sys.exit(1)


# ------------------------------------------------------------
# Main Demo Pipeline
# ------------------------------------------------------------

def main():
    print("=== SpecLens-PML Demo (Continuous Learning) ===")

    root = Path(__file__).parent

    train_dir = root / "data" / "raw_train"
    feedback_dir = root / "data" / "raw_feedback"
    test_dir = root / "data" / "raw_test"
    unseen_dir = root / "data" / "raw_unseen"

    dataset_train = root / "data" / "datasets_train.csv"
    dataset_test = root / "data" / "datasets_test.csv"

    feedback_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Prepare training pool (train + feedback)
    # --------------------------------------------------------
    print("\n=== Step 1: Preparing TRAIN dataset (with feedback) ===")

    tmp_train = root / "data" / "_tmp_train"

    if tmp_train.exists():
        shutil.rmtree(tmp_train)

    tmp_train.mkdir()

    for folder in [train_dir, feedback_dir]:
        for py_file in folder.glob("*.py"):
            shutil.copy(py_file, tmp_train / py_file.name)

    run_step([
        sys.executable,
        str(root / "pipeline" / "build_dataset.py"),
        str(tmp_train),
        str(dataset_train)
    ])

    # --------------------------------------------------------
    # Step 2: Build TEST dataset (fixed evaluation split)
    # --------------------------------------------------------
    print("\n=== Step 2: Building TEST dataset ===")

    run_step([
        sys.executable,
        str(root / "pipeline" / "build_dataset.py"),
        str(test_dir),
        str(dataset_test)
    ])

    # --------------------------------------------------------
    # Step 3: Train candidate models
    # --------------------------------------------------------
    print("\n=== Step 3: Training candidate models ===")

    for model_name in ["logistic", "forest"]:
        run_step([
            sys.executable,
            str(root / "pipeline" / "train.py"),
            str(dataset_train),
            "--model", model_name
        ])

    # --------------------------------------------------------
    # Step 4: Promote best model using TEST dataset
    # --------------------------------------------------------
    print("\n=== Step 4: Continuous Training Promotion ===")

    run_step([
        sys.executable,
        str(root / "ct_trigger.py"),
        str(dataset_test)
    ])

    # --------------------------------------------------------
    # Step 5: Inference on unseen examples + feedback collection
    # --------------------------------------------------------
    print("\n=== Step 5: Inference on UNSEEN examples ===")

    for py_file in sorted(unseen_dir.glob("*.py")):

        print(f"\n--- Analyzing {py_file.name} ---")

        result = subprocess.run(
            [
                sys.executable,
                str(root / "inference" / "predict.py"),
                str(py_file)
            ],
            capture_output=True,
            text=True
        )

        print(result.stdout)

        # Feedback rule:
        # if HIGH risk appears, add file into feedback directory
        if "[HIGH]" in result.stdout:

            print("High-risk function detected.")
            print("Adding example to feedback dataset.")

            shutil.copy(py_file, feedback_dir / py_file.name)

    print("\n=== Demo completed successfully ===")
    print("Feedback pool updated in: data/raw_feedback/")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()

