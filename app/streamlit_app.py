"""CANGuard TARA - Streamlit front end.  Run:  streamlit run app/streamlit_app.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import pages_live, pages_ml, pages_ops, pages_overview, ui  # noqa: E402
from src.config import METRICS_PATH  # noqa: E402
from src.detect import artifacts_ready, get_detector  # noqa: E402

st.set_page_config(page_title="CANGuard TARA", page_icon="\U0001F6E1", layout="wide")
ui.inject_css()

PAGES = {
    "Overview": pages_overview.render, "Data & EDA": pages_ml.render_eda, "Features & Models": pages_ml.render_models,
    "Evaluation": pages_ml.render_eval, "Live Detection": pages_live.render_live, "TARA Report": pages_live.render_tara,
    "Responsible AI & MLOps": pages_ops.render,
}


@st.cache_data(show_spinner=False)
def load_metrics(mtime: float) -> dict:
    return json.loads(METRICS_PATH.read_text())


def train_now() -> None:
    with st.spinner("Generating data and training (fast mode, a few minutes)..."):
        res = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--fast"], cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        st.error("Training failed:\n" + res.stderr[-1500:])
        return
    st.cache_data.clear()
    get_detector.cache_clear()
    st.rerun()


with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">CG</div><div><div class="t">CANGuard TARA</div><div class="s">CAN intrusion detection x ISO/SAE 21434</div></div></div>', unsafe_allow_html=True)
    ready = artifacts_ready()
    if ready:
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
        m = load_metrics(METRICS_PATH.stat().st_mtime)
        st.markdown(f'<div class="card"><p><b>Model</b> v{m["meta"]["version"]}<br>{m["meta"]["best_model"]}<br>F1 {m["final"]["scores"]["f1_macro"]:.3f}</p></div>', unsafe_allow_html=True)
        with st.expander("LLM settings (optional)"):
            key = st.text_input("Ollama API key", type="password", help="Only kept in this server process.")
            if key:
                os.environ["OLLAMA_API_KEY"] = key
            os.environ["OLLAMA_MODEL"] = st.text_input("Model", os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"))
        if st.button("Retrain (fast)"):
            train_now()
    st.caption("Decision support only. Outputs need review by a qualified engineer.")

if not ready:
    ui.hero("Welcome to CANGuard TARA", "No trained model found yet. One click generates synthetic CAN traffic, runs the full ML pipeline and opens the dashboard.")
    if st.button("Generate data & train", type="primary"):
        train_now()
    st.code("python run_pipeline.py        # same thing from the terminal (add --fast for a quicker run)", language="bash")
else:
    PAGES[page](m)
