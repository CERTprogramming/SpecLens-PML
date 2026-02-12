"""
SpecLens-PML Demo Pipeline (Continuous Learning).

This script runs a full end-to-end MLOps workflow:

1. Prepare the TRAIN dataset (raw_train + feedback pool)
2. Build the TEST dataset (held-out evaluation set)
3. Train candidate models (logistic + forest)
4. Promote the best candidate via Continuous Training Trigger
5. Run inference on UNSEEN examples
6. Collect high-risk unseen files into raw_feedback/

This implements a simplified continuous learning loop:

    train → test → promote → unseen → feedback → retrain
"""

from pathlib import Path

import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helper: Safe pipeline execution
# ---------------------------------------------------------------------------

def run_step(cmd: list[str]) -> None:
    """
    Execute a pipeline command and stop the demo if it fails.

    Parameters
    ----------
    cmd : list[str]
        Command arguments passed to ``subprocess.run``.

    Raises
    ------
    SystemExit
        If the executed command returns a non-zero exit code.
    """
    print(f"\n>>> {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Pipeline step failed. Stopping demo.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helper: Training pool preparation
# ---------------------------------------------------------------------------

def prepare_training_pool(
    train_dir: Path,
    feedback_dir: Path,
    tmp_dir: Path,
) -> None:
    """
    Prepare the effective training pool for dataset generation.

    The staging directory is built by merging:

    - ``raw_train/``: base annotated training examples
    - ``raw_feedback/``: collected high-risk unseen examples

    The merged pool is copied into a temporary folder (``_tmp_train/``),
    which is then used by ``build_dataset.py``.

    Parameters
    ----------
    train_dir : Path
        Directory containing the base training examples.
    feedback_dir : Path
        Directory containing feedback examples (may be empty).
    tmp_dir : Path
        Temporary staging directory created for dataset generation.
    """
    # Reset staging directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True)

    # Copy base training examples
    for f in train_dir.glob("*.py"):
        shutil.copy(f, tmp_dir / f.name)

    # Copy feedback examples if available (avoid duplicates)
    if feedback_dir.exists():
        for f in feedback_dir.glob("*.py"):
            target = tmp_dir / f.name
            if not target.exists():
                shutil.copy(f, target)


# ---------------------------------------------------------------------------
# Main Demo Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full SpecLens-PML continuous learning demo.

    This orchestrates dataset generation, candidate training,
    champion promotion, inference on unseen examples, and
    feedback collection.
    """
    print("=== SpecLens-PML Demo (Continuous Learning) ===")

    root = Path(__file__).parent

    # Data folders
    train_dir = root / "data" / "raw_train"
    test_dir = root / "data" / "raw_test"
    unseen_dir = root / "data" / "raw_unseen"
    feedback_dir = root / "data" / "raw_feedback"

    # Temporary staging directory (regenerated every run)
    tmp_train_dir = root / "data" / "_tmp_train"

    # Dataset outputs
    train_csv = root / "data" / "datasets_train.csv"
    test_csv = root / "data" / "datasets_test.csv"

    # -----------------------------------------------------------------------
    # Step 1: Prepare TRAIN pool (with feedback, if available)
    # -----------------------------------------------------------------------

    print("\n=== Step 1: Preparing TRAIN dataset (with feedback, if available) ===")

    prepare_training_pool(train_dir, feedback_dir, tmp_train_dir)

    run_step([
        sys.executable,
        str(root / "pipeline" / "build_dataset.py"),
        str(tmp_train_dir),
        str(train_csv),
    ])

    # -----------------------------------------------------------------------
    # Step 2: Build TEST dataset (held-out evaluation set)
    # -----------------------------------------------------------------------

    print("\n=== Step 2: Building TEST dataset ===")

    run_step([
        sys.executable,
        str(root / "pipeline" / "build_dataset.py"),
        str(test_dir),
        str(test_csv),
    ])

    # -----------------------------------------------------------------------
    # Step 3: Train candidate models
    # -----------------------------------------------------------------------

    print("\n=== Step 3: Training candidate models ===")

    run_step([
        sys.executable,
        str(root / "pipeline" / "train.py"),
        str(train_csv),
        "--model",
        "logistic",
    ])

    run_step([
        sys.executable,
        str(root / "pipeline" / "train.py"),
        str(train_csv),
        "--model",
        "forest",
    ])

    # -----------------------------------------------------------------------
    # Step 4: Champion promotion
    # -----------------------------------------------------------------------

    print("\n=== Step 4: Continuous Training Promotion ===")

    run_step([
        sys.executable,
        str(root / "ct_trigger.py"),
        str(test_csv),
    ])

    # -----------------------------------------------------------------------
    # Step 5: Inference on UNSEEN examples + feedback collection
    # -----------------------------------------------------------------------

    print("\n=== Step 5: Inference on UNSEEN examples ===")

    feedback_dir.mkdir(exist_ok=True)

    for py_file in sorted(unseen_dir.glob("*.py")):

        print(f"\n--- Analyzing {py_file.name} ---")

        result = subprocess.run(
            [
                sys.executable,
                str(root / "inference" / "predict.py"),
                str(py_file),
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        # Feedback rule: collect example if HIGH risk appears
        if "[HIGH]" in result.stdout:

            target = feedback_dir / py_file.name

            if not target.exists():
                print("High-risk function detected.")
                print("Adding example to feedback dataset.")
                shutil.copy(py_file, target)
            else:
                print("High-risk example already present in feedback pool.")

    print("\n=== Demo completed successfully ===")
    print(f"Feedback pool updated in: {feedback_dir}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

