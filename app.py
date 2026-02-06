"""
Streamlit web application for SpecLens-PML.

This module provides an interactive graphical user interface (GUI)
to run the full MLOps pipeline (dataset generation, training, inference)
and analyze new Python files annotated with PML specifications.

The app acts as a lightweight frontend on top of the command-line tools
already available in the repository.
"""

from pathlib import Path

import streamlit as st
import subprocess

# Define the root directory of the project (repository base path)
ROOT = Path(__file__).parent

# Configure the Streamlit page layout and browser tab title
st.set_page_config(page_title="SpecLens-PML", layout="wide")

# Display the main title and a short project tagline
st.title("SpecLens-PML")
st.caption("Data-driven Software Correctness with MLOps")

# --- Sidebar: MLOps controls -------------------------------------------------
# The sidebar provides operational controls for running pipeline stages.

st.sidebar.header("MLOps Control")

# Button to execute the full end-to-end pipeline (demo.py)
if st.sidebar.button("Run full pipeline (demo)"):
    with st.spinner("Running full pipeline..."):
        # Run the demo script as a subprocess from the project root
        result = subprocess.run(
            ["python3", "demo.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    # Notify the user once execution is complete
    st.sidebar.success("Pipeline completed")
    # Display the pipeline output logs in the sidebar
    st.sidebar.text(result.stdout)

# Button to trigger the continuous training and model promotion workflow
if st.sidebar.button("Trigger Continuous Training"):
    with st.spinner("Triggering retraining..."):
        # Run the continuous training trigger script (ct_trigger.py)
        result = subprocess.run(
            ["python3", "ct_trigger.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    # Notify the user once retraining has finished
    st.sidebar.success("Continuous Training completed")
    # Display retraining logs and evaluation output
    st.sidebar.text(result.stdout)

# Show active model
# The active model pointer defines which model artifact is used in production.
active_model_path = ROOT / "models" / "active_model.txt"
if active_model_path.exists():
    # Read the currently deployed model path from the pointer file
    active_model = active_model_path.read_text().strip()
    st.sidebar.markdown("### Active model")
    # Display the active model path in a formatted code block
    st.sidebar.code(active_model)

# --- Main area: Inference ----------------------------------------------------
# The main panel allows users to upload code and run inference.

st.header("Analyze Python Code")

# File uploader widget for Python source files annotated with PML contracts
uploaded = st.file_uploader("Upload a Python file annotated with PML", type="py")

if uploaded:
    # Save the uploaded file temporarily inside the project directory
    tmp_path = ROOT / "tmp_uploaded.py"
    tmp_path.write_bytes(uploaded.read())

    with st.spinner("Analyzing code..."):
        # Run the inference script on the uploaded file
        result = subprocess.run(
            ["python3", "inference/predict.py", str(tmp_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    # Display the inference output produced by predict.py
    st.subheader("Analysis Result")
    st.code(result.stdout)
