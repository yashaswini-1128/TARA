"""Overview page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app import ui


def render(m: dict) -> None:
    meta, fin, eda = m["meta"], m["final"]["scores"], m["eda"]
    ui.hero("CANGuard TARA",
            "ML-powered CAN-bus intrusion detection that feeds an ISO/SAE 21434-style Threat Analysis and Risk Assessment. "
            "Detection is learned from data; the risk rating stays deterministic and auditable; an engineer approves every output.",
            ["Case Study 3 - Cybersecurity TARA", "Classical ML (scikit-learn)", "FastAPI + Streamlit", "Explainable", "Human-in-the-loop"])
    ui.kpis([
        ("Macro F1 (held-out future data)", f"{fin['f1_macro']:.3f}", f"{meta['best_model']} - tuned", "#22D3EE"),
        ("Attack recall", f"{fin['attack_recall']:.1%}", "attack frames caught", "#34D399"),
        ("False-alarm rate", f"{fin['false_alarm_rate']:.2%}", "normal frames flagged", "#FBBF24"),
        ("Inference latency", f"{meta['latency_ms_per_frame']:.3f} ms", "per frame, CPU", "#8B5CF6"),
        ("Frames analysed", f"{eda['n_frames']:,}", f"{eda['unique_ids']} CAN IDs - {eda['frames_per_s']:,.0f} frames/s", "#F472B6"),
    ])

    ui.section("Why this problem matters")
    c1, c2, c3 = st.columns(3)
    c1.markdown(ui.card("The threat", "<p>CAN has no sender authentication. Anyone who reaches the bus (OBD-II port, a compromised telematics unit) "
                "can flood it, inject random frames or forge safety-relevant signals such as engine speed or gear position.</p>"), unsafe_allow_html=True)
    c2.markdown(ui.card("The engineering gap", "<p>ISO/SAE 21434 requires a TARA for every item, but TARA is manual and expert-heavy, and "
                "disconnected from what attacks actually look like on the bus.</p>"), unsafe_allow_html=True)
    c3.markdown(ui.card("This project", "<p>An intrusion detector learns attack signatures from bus traffic. Each confirmed alert is mapped to a "
                "threat scenario, rated with impact x feasibility, and paired with mitigations, tests and a linked safety hazard.</p>"), unsafe_allow_html=True)

    ui.section("How it works", "From raw frames to an approved TARA entry")
    ui.flow([("01", "CAN frames", "timestamp - ID - DLC - 8 bytes"), ("02", "Feature engineering", f"{meta['n_features']} online features, baseline from train only"),
             ("03", "Classifier", "5 classes: Normal + 4 attacks"), ("04", "Alert debouncing", "min frames per window"),
             ("05", "TARA engine", "rule-based risk 1-5"), ("06", "Engineer review", "approve - edit - reject")])

    ui.section("What goes beyond the reference document")
    a, b = st.columns(2)
    a.markdown(ui.card("Technical novelty", "<ul><li><b>Leakage-aware validation</b>: time-ordered split + group CV, so whole attack bursts never straddle train and validation.</li>"
               "<li><b>Zero-day test</b>: leave-one-attack-out experiment comparing a supervised model with an unsupervised Isolation Forest.</li>"
               "<li><b>Drift monitor</b> (PSI) on live traffic against the training-normal baseline.</li></ul>"), unsafe_allow_html=True)
    b.markdown(ui.card("Engineering novelty", "<ul><li><b>Evidence-linked TARA</b>: detection results flow into the risk assessment.</li>"
               "<li><b>Mitigation what-if simulator</b>: toggle controls and watch residual risk change.</li>"
               "<li><b>Safety-security bridge</b>: every threat names a candidate hazard for the HARA team.</li></ul>"), unsafe_allow_html=True)

    ui.section("Evaluation-rubric coverage", "Where each scoring parameter is demonstrated")
    rows = [("Problem understanding & automotive context", 5, "Overview, TARA Report"), ("AI/ML fundamentals & approach selection", 5, "Features & Models (why each model)"),
            ("Dataset understanding, preprocessing & EDA", 15, "Data & EDA"), ("Feature engineering, selection & dimensionality reduction", 8, "Features & Models"),
            ("Model development using syllabus techniques", 18, "Features & Models (6 supervised + Isolation Forest)"),
            ("Evaluation, validation & optimisation", 15, "Evaluation, tuning table"), ("Completeness & working demonstration", 15, "Live Detection, TARA Report, REST API"),
            ("Python implementation & code quality", 6, "Modular src/, type hints, tests"), ("Deployment & MLOps basics", 5, "Responsible AI & MLOps, Dockerfile, API"),
            ("Responsible AI & deployment ethics", 3, "Responsible AI & MLOps"), ("Technical Q&A", 3, "docs/VIVA_QA.md"), ("Communication & presentation", 2, "This interface, README")]
    ui.table(pd.DataFrame(rows, columns=["Parameter", "Marks", "Where to see it"]))
    ui.callout("Data note: the demo runs on <b>synthetic CAN traffic</b> that mirrors the public HCRL Car-Hacking schema. "
               "Use <span class='mono'>python run_pipeline.py --real-dir &lt;folder&gt;</span> to retrain on the real dataset for final reported numbers.", "#FBBF24")
