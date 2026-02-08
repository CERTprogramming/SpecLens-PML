"""
app.py

Streamlit web interface for SpecLens-PML.

This module provides a lightweight graphical frontend to:

- Execute the full MLOps demo pipeline (train/test/promotion/inference)
- Upload and analyze new Python files annotated with PML contracts

The Streamlit GUI is intentionally minimal:
all MLOps logic remains implemented in the CLI scripts
(e.g., demo.py, ct_trigger.py, inference/predict.py).

The goal is to provide a simple operational entry point for exam demos.
"""

from pathlib import Path
import subprocess

import streamlit as st


# ---------------------------------------------------------------------------
# Repository root configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

st.set_page_config(page_title="SpecLens-PML", layout="wide")


# ---------------------------------------------------------------------------
# Main page header
# ---------------------------------------------------------------------------

st.title("SpecLens-PML")
st.caption("Data-driven Software Correctness with MLOps")


# ---------------------------------------------------------------------------
# Sidebar: Minimal pipeline execution
# ---------------------------------------------------------------------------

st.sidebar.header("Pipeline Control")

if st.sidebar.button("Run full pipeline (demo.py)"):
    with st.spinner("Running full continuous learning pipeline..."):
        result = subprocess.run(
            ["python3", "demo.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    st.sidebar.success("Pipeline completed")
    st.sidebar.code(result.stdout)


# ---------------------------------------------------------------------------
# Sidebar: Active champion model display
# ---------------------------------------------------------------------------

champion_path = ROOT / "models" / "best_model.pkl"

st.sidebar.markdown("### Active Champion Model")

if champion_path.exists():
    st.sidebar.code("models/best_model.pkl")
else:
    st.sidebar.warning("No champion model found yet. Run the pipeline first.")


# ---------------------------------------------------------------------------
# Main panel: Inference on uploaded Python files
# ---------------------------------------------------------------------------

st.header("Analyze Python Code")

uploaded = st.file_uploader(
    "Upload a Python file annotated with PML",
    type="py",
)

if uploaded:
    tmp_path = ROOT / "tmp_uploaded.py"
    tmp_path.write_bytes(uploaded.read())

    if not champion_path.exists():
        st.error("No trained model available. Please run the pipeline first.")
    else:
        with st.spinner("Analyzing code with champion model..."):
            result = subprocess.run(
                ["python3", "inference/predict.py", str(tmp_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        st.subheader("Analysis Result")
        st.code(result.stdout)
