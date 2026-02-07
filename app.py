"""
Streamlit web application for SpecLens-PML.

This module is part of the SpecLens demo project:
it provides an interactive graphical user interface (GUI)
to run the full MLOps pipeline (dataset generation, training, inference)
and analyze new Python files annotated with PML specifications.

The goal is not to replace the command-line tools,
but to showcase how the pipeline can be executed and explored
through a lightweight frontend.
"""

from pathlib import Path
import subprocess

import streamlit as st


# ---------------------------------------------------------------------------
# Project root configuration
# ---------------------------------------------------------------------------

# Define the root directory of the repository
ROOT = Path(__file__).parent

# Configure the Streamlit page layout and browser tab title
st.set_page_config(page_title="SpecLens-PML", layout="wide")


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("SpecLens-PML")
st.caption("Data-driven Software Correctness with MLOps")


# ---------------------------------------------------------------------------
# Sidebar: MLOps controls
# ---------------------------------------------------------------------------

st.sidebar.header("MLOps Control")

# Run the full end-to-end demo pipeline (dataset → training → inference)
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

# Trigger continuous training and model promotion workflow
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
# Sidebar: Active model display
# ---------------------------------------------------------------------------

# The active model pointer defines which model is used for inference
active_model_path = ROOT / "models" / "active_model.txt"

if active_model_path.exists():
    active_model = active_model_path.read_text().strip()

    st.sidebar.markdown("### Active model")
    st.sidebar.code(active_model)


# ---------------------------------------------------------------------------
# Main panel: Inference on uploaded files
# ---------------------------------------------------------------------------

st.header("Analyze Python Code")

# Upload a Python file annotated with PML contracts
uploaded = st.file_uploader(
    "Upload a Python file annotated with PML",
    type="py",
)

if uploaded:
    # Save the uploaded file temporarily inside the repository
    tmp_path = ROOT / "tmp_uploaded.py"
    tmp_path.write_bytes(uploaded.read())

    with st.spinner("Analyzing code..."):
        result = subprocess.run(
            ["python3", "inference/predict.py", str(tmp_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    # Display the inference output produced by predict.py
    st.subheader("Analysis Result")
    st.code(result.stdout)

