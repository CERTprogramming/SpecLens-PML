"""
app.py

Streamlit web interface for SpecLens-PML.

This module provides a lightweight graphical frontend to:

- Execute the full MLOps demo pipeline (train/test/promotion/inference)
- Trigger the Continuous Training promotion step
- Upload and analyze new Python files annotated with PML contracts

The Streamlit GUI is a presentation layer only:
all MLOps logic remains implemented in the CLI scripts
(e.g., demo.py, ct_trigger.py, inference/predict.py).
"""

from pathlib import Path
import subprocess

import streamlit as st


# ---------------------------------------------------------------------------
# Repository root configuration
# ---------------------------------------------------------------------------

#: Root directory of the SpecLens-PML repository.
ROOT = Path(__file__).parent

# Configure the Streamlit page layout and browser tab title.
st.set_page_config(page_title="SpecLens-PML", layout="wide")


# ---------------------------------------------------------------------------
# Main page header
# ---------------------------------------------------------------------------

st.title("SpecLens-PML")
st.caption("Data-driven Software Correctness with MLOps")


# ---------------------------------------------------------------------------
# Sidebar controls: Pipeline execution
# ---------------------------------------------------------------------------

st.sidebar.header("MLOps Control")

# Run the full end-to-end demo workflow:
# dataset generation → training → promotion → unseen inference.
if st.sidebar.button("Run full pipeline (demo)"):
    with st.spinner("Running full pipeline..."):
        result = subprocess.run(
            ["python3", "demo.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    st.sidebar.success("Pipeline completed")
    st.sidebar.text(result.stdout)

# Trigger the Continuous Training promotion step only.
if st.sidebar.button("Trigger Continuous Training"):
    with st.spinner("Triggering retraining..."):
        result = subprocess.run(
            ["python3", "ct_trigger.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    st.sidebar.success("Continuous Training completed")
    st.sidebar.text(result.stdout)


# ---------------------------------------------------------------------------
# Sidebar: Active champion model display
# ---------------------------------------------------------------------------

#: Pointer file storing the name of the currently promoted champion model.
active_model_path = ROOT / "models" / "active_model.txt"

if active_model_path.exists():
    active_model = active_model_path.read_text().strip()

    st.sidebar.markdown("### Active model")
    st.sidebar.code(active_model)


# ---------------------------------------------------------------------------
# Main panel: Inference on uploaded Python files
# ---------------------------------------------------------------------------

st.header("Analyze Python Code")

# Upload a Python file annotated with PML specifications.
uploaded = st.file_uploader(
    "Upload a Python file annotated with PML",
    type="py",
)

if uploaded:
    # Save the uploaded file temporarily inside the repository.
    tmp_path = ROOT / "tmp_uploaded.py"
    tmp_path.write_bytes(uploaded.read())

    # Run inference using the promoted champion model.
    with st.spinner("Analyzing code..."):
        result = subprocess.run(
            ["python3", "inference/predict.py", str(tmp_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    # Display the prediction output produced by inference/predict.py.
    st.subheader("Analysis Result")
    st.code(result.stdout)

