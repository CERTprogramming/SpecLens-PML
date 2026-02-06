# app.py

import streamlit as st
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent

st.set_page_config(page_title="SpecLens-PML", layout="wide")

st.title("SpecLens-PML")
st.caption("Data-driven Software Correctness with MLOps")

# --- Sidebar: MLOps controls -------------------------------------------------

st.sidebar.header("MLOps Control")

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

# Show active model
active_model_path = ROOT / "models" / "active_model.txt"
if active_model_path.exists():
    active_model = active_model_path.read_text().strip()
    st.sidebar.markdown("### Active model")
    st.sidebar.code(active_model)

# --- Main area: Inference ----------------------------------------------------

st.header("Analyze Python Code")

uploaded = st.file_uploader("Upload a Python file annotated with PML", type="py")

if uploaded:
    tmp_path = ROOT / "tmp_uploaded.py"
    tmp_path.write_bytes(uploaded.read())

    with st.spinner("Analyzing code..."):
        result = subprocess.run(
            ["python3", "inference/predict.py", str(tmp_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    st.subheader("Analysis Result")
    st.code(result.stdout)
