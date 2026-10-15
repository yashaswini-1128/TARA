"""Responsible AI and MLOps page."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app import ui
from src.config import BASELINE_PATH, DATA_PATH, EDA_PATH, METRICS_PATH, MODEL_PATH


def render(m: dict) -> None:
    meta, fin = m["meta"], m["final"]["scores"]
    ui.hero("Responsible AI & MLOps", "Where the system can fail, how that is controlled, and how it is packaged and monitored.")
    t1, t2, t3 = st.tabs(["Responsible AI", "Model card", "MLOps & deployment"])

    with t1:
        ui.section("Risks and controls")
        ui.table(pd.DataFrame([
            ("Missed attack (false negative)", "Safety-relevant: an undetected spoof can reach actuators.", "Recall-first metrics, anomaly detector as safety net, preventive controls (SecOC, gateway) do not depend on the model."),
            ("False alarm (alert fatigue)", "Too many alerts get ignored.", "Debounced alerts, confidence shown, false-alarm rate tracked as a headline KPI."),
            ("Synthetic training data", "Real buses are messier; reported scores will not transfer 1:1.", "Schema-compatible loader for the real HCRL dataset; report real-data numbers separately."),
            ("Adversarial evasion", "An attacker can mimic timing and payload.", "Treat the IDS as a detective layer only; red-team review; authenticated messages."),
            ("Data drift", "New ECU software / variants change normal traffic.", "PSI drift monitor, versioned models, scheduled revalidation."),
            ("Automation bias", "Engineers over-trust generated text.", "Deterministic rating, cited evidence, draft banner, human approval before any use."),
            ("Standards / IP", "Using unlicensed standards text.", "Tables are illustrative and labelled; calibrate with the authorised copy."),
            ("Privacy & confidentiality", "Architecture details are sensitive.", "Local data only; the LLM receives a structured summary, never raw traffic.")],
            columns=["Risk", "Why it matters", "Control"]))
        a, b = st.columns(2)
        a.markdown(ui.card("Principles", "<ul><li>The model <b>assists</b>; engineers decide.</li><li>Risk values come from fixed tables, not from an LLM.</li>"
                   "<li>Every output carries its evidence and limitations.</li><li>No autonomous action on a vehicle.</li></ul>"), unsafe_allow_html=True)
        b.markdown(ui.card("Out of scope", "<ul><li>Active exploitation or attack tooling.</li><li>Blocking traffic on a production vehicle.</li><li>Autonomous cybersecurity acceptance.</li><li>Replacing a qualified review.</li></ul>"), unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)
        c1.markdown(ui.card("Model details", f"<ul><li>Name: <b>CANGuard classifier v{meta['version']}</b></li><li>Algorithm: <b>{meta['best_model']}</b></li>"
                    f"<li>Hyper-parameters: <span class='mono'>{meta['best_params']}</span></li><li>Features: {meta['n_features']} online features</li>"
                    f"<li>Trained (UTC): {meta['trained_utc']}</li><li>Seed: {meta['seed']}, scikit-learn {meta['sklearn']}, Python {meta['python']}</li>"
                    f"<li>Train / test frames: {meta['train_rows']:,} / {meta['test_rows']:,}</li></ul>"), unsafe_allow_html=True)
        c2.markdown(ui.card("Performance (held-out future window)", f"<ul><li>Accuracy: <b>{fin['accuracy']:.4f}</b></li><li>Macro F1: <b>{fin['f1_macro']:.4f}</b></li>"
                    f"<li>Attack recall: <b>{fin['attack_recall']:.2%}</b></li><li>False-alarm rate: <b>{fin['false_alarm_rate']:.3%}</b></li>"
                    f"<li>Latency: <b>{meta['latency_ms_per_frame']:.3f} ms/frame</b></li></ul>"), unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.markdown(ui.card("Intended use", "<p>Decision support for cybersecurity engineers: detecting known CAN attack patterns on test benches and in recorded drives, and "
                    "attaching evidence to TARA entries.</p>"), unsafe_allow_html=True)
        d2.markdown(ui.card("Limitations", "<ul><li>Trained on simulated traffic unless retrained on real captures.</li><li>Four attack families only.</li>"
                    "<li>Not validated for series-production vehicles.</li><li>Rating tables are illustrative.</li></ul>"), unsafe_allow_html=True)

    with t3:
        ui.section("Delivery pipeline")
        ui.flow([("DATA", "Generate / load", "run_pipeline.py"), ("TRAIN", "Fit + tune + evaluate", "seeded, reproducible"), ("REGISTER", "Versioned artifacts", "model, baseline, metrics"),
                 ("SERVE", "FastAPI + Streamlit", "same code path"), ("MONITOR", "PSI drift + alerts", "retrain trigger")])
        rows = []
        for p in (DATA_PATH, MODEL_PATH, BASELINE_PATH, METRICS_PATH, EDA_PATH):
            if p.exists():
                rows.append((p.name, f"{p.stat().st_size / 1e6:.2f} MB", dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")))
        ui.section("Artifacts on disk")
        ui.table(pd.DataFrame(rows, columns=["File", "Size", "Last modified"]))
        ui.section("REST API")
        ui.table(pd.DataFrame([("GET", "/health", "Liveness + model readiness"), ("GET", "/model/info", "Version, metrics, comparison"), ("POST", "/predict", "JSON frames -> predictions, alerts, TARA summary, drift"),
                               ("POST", "/predict/csv", "Upload a capture"), ("GET", "/tara", "Whole threat register"), ("GET", "/tara/{attack}?mitigations=M1,M2", "One assessment with what-if controls")],
                              columns=["Method", "Path", "Purpose"]))
        st.code("uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs\n"
                "curl -F file=@capture.csv http://127.0.0.1:8000/predict/csv\n"
                "docker build -t canguard-tara . && docker run -p 8501:8501 -p 8000:8000 canguard-tara", language="bash")
        ui.callout("Reproducibility: fixed seeds, pinned minimum versions, artifacts carry their version and training date, and <span class='mono'>pytest</span> covers features, TARA logic and the API.")
