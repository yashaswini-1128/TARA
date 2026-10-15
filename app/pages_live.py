"""Live detection and TARA report pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, recall_score

from app import ui
from src import tara
from src.config import ATTACKS, CLASSES, COLORS
from src.detect import alerts, drift_report, get_detector
from src.llm import narrative
from src.simulate import generate

SCENARIOS = ["Normal traffic", *ATTACKS, "Multi-attack"]


def goto(page: str, attack: str | None = None) -> None:
    st.session_state["nav"] = page
    if attack:
        st.session_state["tara_attack"] = attack


def _run(upload, scenario: str, duration: int, intensity: float, seed: int) -> None:
    det = get_detector()
    if upload is not None:
        df = pd.read_csv(upload).head(300_000)
        scenario = "Uploaded capture"
    else:
        atk = [] if scenario == "Normal traffic" else ATTACKS if scenario == "Multi-attack" else [scenario]
        df = generate(float(duration), int(seed), attacks=atk, bursts_per_attack=2, intensity=float(intensity))
    pred = det.predict(df)
    st.session_state["live"] = {"pred": pred, "alerts": alerts(pred), "drift": drift_report(pred.attrs["features"], det.baseline), "scenario": scenario}


def _timeline(pred: pd.DataFrame, al: list[dict]) -> go.Figure:
    t = pred.timestamp.to_numpy()
    bins = np.arange(t.min(), t.max() + 0.1, 0.1)
    load, _ = np.histogram(t, bins)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.28, 0.72], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=bins[:-1], y=load, fill="tozeroy", line=dict(color="#22D3EE", width=1.5), name="bus load", showlegend=False), row=1, col=1)
    for c in CLASSES:
        s = pred[pred.pred == c]
        if len(s) == 0:
            continue
        s = s.sample(2500, random_state=0) if len(s) > 2500 else s
        fig.add_trace(go.Scattergl(x=s.timestamp, y=s.can_id, mode="markers", name=c,
                                   marker=dict(color=COLORS[c], size=4 if c == "Normal" else 5, opacity=.5 if c == "Normal" else .9)), row=2, col=1)
    if "label" in pred and (pred.label != "unknown").any():
        err = pred[pred.pred != pred.label]
        err = err.sample(800, random_state=0) if len(err) > 800 else err
        if len(err):
            fig.add_trace(go.Scattergl(x=err.timestamp, y=err.can_id, mode="markers", name="misclassified", marker=dict(symbol="x", color="#FFFFFF", size=7)), row=2, col=1)
    for a in al:
        fig.add_vrect(x0=a["start"], x1=a["end"], fillcolor=COLORS[a["class"]], opacity=.16, line_width=0, row="all", col=1)
    fig.update_yaxes(title_text="frames / 100 ms", row=1, col=1)
    fig.update_yaxes(title_text="CAN ID", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    return ui.style(fig, 520).update_layout(title="Bus timeline - colour = model prediction, shaded = raised alert")


def render_live(m: dict) -> None:
    ui.hero("Live Detection", "Replay fresh traffic (a different random seed than training) through the detector, or upload your own capture. "
            "Frame-level predictions are debounced into alerts, and each alert links to its TARA entry.")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, .8])
        scenario = c1.selectbox("Scenario", SCENARIOS, index=1)
        duration = c2.slider("Duration (s)", 10, 40, 20)
        intensity = c3.slider("Attack intensity", 0.2, 1.0, 0.8, 0.05, help="Lower = stealthier (sparser injections)")
        seed = c4.number_input("Seed", 0, 9999, 7)
        up = st.file_uploader("...or upload a capture (CSV: timestamp, can_id, dlc, d0..d7 [, label])", type=["csv"])
        run = st.button("Run detection", type="primary")
    if run:
        with st.spinner("Building features and classifying frames..."):
            try:
                _run(up, scenario, duration, intensity, seed)
            except Exception as e:
                st.error(f"Could not process the capture: {e}")
    live = st.session_state.get("live")
    if not live:
        ui.callout("Pick a scenario and press <b>Run detection</b>.")
        return
    pred, al = live["pred"], live["alerts"]
    flagged = int((pred.pred != "Normal").sum())
    has_truth = "label" in pred and (pred.label != "unknown").any()
    items = [("Frames", f"{len(pred):,}", live["scenario"], "#22D3EE"), ("Frames flagged", f"{flagged:,}", f"{flagged / len(pred):.1%} of traffic", "#F87171"),
             ("Alerts raised", len(al), "debounced bursts", "#FBBF24")]
    if has_truth:
        atk = pred.label != "Normal"
        rec = float(((pred.pred != "Normal") & atk).sum() / max(atk.sum(), 1))
        f1 = f1_score(pred.label, pred.pred, average="macro", labels=[c for c in CLASSES if c in set(pred.label)], zero_division=0)
        items += [("Attack recall (this run)", f"{rec:.1%}" if atk.any() else "n/a", "ground truth available", "#34D399"), ("Macro F1 (this run)", f"{f1:.3f}", "present classes", "#8B5CF6")]
    ui.kpis(items)
    ui.show(_timeline(pred, al))

    ui.section("Alerts", "An alert needs at least 5 flagged frames of one class within 50 ms gaps - a simple debounce that suppresses one-off misclassifications")
    if not al:
        ui.callout("No alerts raised - traffic looks normal.", "#34D399")
    for i, a in enumerate(al[:10]):
        t = tara.assess(a["class"], [])
        with st.container(border=True):
            c = st.columns([1.1, 1.5, .8, .9, 1.3, 1.3])
            c[0].markdown(ui.badge(a["class"], COLORS[a["class"]]), unsafe_allow_html=True)
            c[1].markdown(f"**{a['start']:.2f}s - {a['end']:.2f}s**")
            c[2].markdown(f"{a['frames']:,} frames")
            c[3].markdown(f"conf. {a['mean_confidence']:.0%}")
            c[4].markdown(ui.badge(f"Risk {t['risk']} - {t['risk_label']}", tara.RISK_COLOR[t["risk"]]), unsafe_allow_html=True)
            c[5].button("Open TARA entry", key=f"open_{i}", on_click=goto, args=("TARA Report", a["class"]))

    ui.section("Drift monitor", "Population Stability Index of live features vs the training-normal baseline")
    d = live["drift"]
    col = {"stable": "#34D399", "watch": "#FBBF24", "shift": "#F87171"}
    f = go.Figure(go.Bar(x=d.feature, y=d.psi, marker_color=[col[s] for s in d.status], text=d.status, textposition="outside"))
    f.add_hline(y=.1, line_dash="dot", line_color="#FBBF24"); f.add_hline(y=.25, line_dash="dot", line_color="#F87171")
    ui.show(ui.style(f, 330, False).update_layout(title="Feature drift (PSI)", yaxis_title="PSI"))
    ui.callout("PSI &lt; 0.1 stable, 0.1-0.25 watch, &gt; 0.25 shift. During an attack a shift is <i>expected</i> - it is the signature. "
               "<b>High drift with no alerts</b> is the real warning: the bus changed (new ECU software, new variant) and the model needs review or retraining.")
    st.download_button("Download predictions (CSV)", pred[["timestamp", "can_id", "pred", "confidence"]].to_csv(index=False), "predictions.csv")


def _matrix_fig(points: list[tuple]) -> go.Figure:
    z = [[tara.RISK_MATRIX[i][j] for j in range(4)] for i in tara.IMPACT_LEVELS]
    cs = [[0, "#34D399"], [.25, "#A3E635"], [.5, "#FBBF24"], [.75, "#FB923C"], [1, "#F87171"]]
    fig = go.Figure(go.Heatmap(z=z, x=tara.FEAS_LEVELS, y=tara.IMPACT_LEVELS, zmin=1, zmax=5, colorscale=cs, text=z, texttemplate="%{text}",
                               textfont=dict(size=15, color="#0A0F1C"), showscale=False, xgap=3, ygap=3, opacity=.85))
    for label, imp, fea, sym, colr in points:
        fig.add_trace(go.Scatter(x=[fea], y=[imp], mode="markers+text", text=[label], textposition="top center", showlegend=False,
                                 marker=dict(symbol=sym, size=24, color=colr, line=dict(color="#FFFFFF", width=2))))
    fig.update_xaxes(title="attack feasibility"); fig.update_yaxes(title="impact")
    return ui.style(fig, 380, False)


def _potential_fig(score: int, res: int) -> go.Figure:
    fig = go.Figure()
    for a, b, lab, c in [(0, 14, "High", "#F87171"), (14, 20, "Medium", "#FBBF24"), (20, 25, "Low", "#A3E635"), (25, 45, "Very low", "#34D399")]:
        fig.add_shape(type="rect", x0=a, x1=b, y0=0, y1=1, fillcolor=c, opacity=.32, line_width=0)
        fig.add_annotation(x=(a + b) / 2, y=.5, text=f"feasibility<br>{lab}", showarrow=False, font=dict(size=11, color="#F1F5F9"))
    fig.add_vline(x=score + .5, line_width=4, line_color="#FFFFFF", annotation_text=f"inherent {score}", annotation_position="top left")
    if res != score:
        fig.add_vline(x=min(res, 44) + .5, line_width=4, line_dash="dash", line_color="#22D3EE", annotation_text=f"residual {res}", annotation_position="top right")
    fig.update_xaxes(range=[0, 45], title="attack potential needed (higher = harder for the attacker)"); fig.update_yaxes(visible=False, range=[0, 1])
    return ui.style(fig, 200, False)


def render_tara(m: dict) -> None:
    ui.hero("TARA Report", "Deterministic ISO/SAE 21434-style assessment: impact x attack feasibility -> risk value, treatment, mitigations and verification. "
            "Generated text is a draft; a qualified engineer approves.")
    live = st.session_state.get("live")
    live_alerts = live["alerts"] if live else []
    detected = sorted({a["class"] for a in live_alerts})
    if "tara_attack" not in st.session_state:
        st.session_state["tara_attack"] = detected[0] if detected else "DoS"
    t1, t2 = st.tabs(["Threat assessment", "Threat register"])

    with t1:
        attack = st.selectbox("Threat scenario", list(tara.THREATS), key="tara_attack",
                              format_func=lambda k: f"{tara.THREATS[k]['id']}  -  {tara.THREATS[k]['name']}" + ("   [detected in last run]" if k in detected else ""))
        base = tara.assess(attack, [])
        sel = [mm["id"] for mm in base["mitigations"] if st.session_state.get(f"mit_{attack}_{mm['id']}")]
        a = tara.assess(attack, sel)
        r = a["residual"]
        ev = max((x for x in live_alerts if x["class"] == attack), key=lambda x: x["frames"], default=None)

        st.markdown(f"### {a['id']} - {a['name']}")
        st.markdown(ui.badge(f"Risk {a['risk']} - {a['risk_label']}", tara.RISK_COLOR[a["risk"]]) + "&nbsp;&nbsp;" + ui.badge(a["stride"], "#8B5CF6") +
                    "&nbsp;&nbsp;" + ui.badge(f"Feasibility {a['feasibility']}", "#22D3EE") + "&nbsp;&nbsp;" + ui.badge(f"Impact {a['impact']}", "#F472B6"), unsafe_allow_html=True)
        l, rr = st.columns([1.1, 1])
        with l:
            imp = "".join(f"<li>{k}: <b>{v}</b></li>" for k, v in a["impact_by_category"].items())
            st.markdown(ui.card("Scenario", f"<p><b>Asset:</b> {a['asset']}</p><p><b>Attack interface:</b> {a['interface']}</p><p><b>Damage scenario:</b> {a['damage']}</p>"
                                f"<p><b>Impact by category</b></p><ul>{imp}</ul><p><b>Treatment:</b> {a['treatment']}</p>"
                                f"<p><b>Cybersecurity goal:</b> {a['goal']}</p><p><b>Linked safety hazard:</b> {a['hazard']}</p>"), unsafe_allow_html=True)
        with rr:
            pts = [("Inherent", a["impact"], a["feasibility"], "circle", "#0A0F1C")]
            if sel:
                pts.append(("Residual", r["impact"], r["feasibility"], "diamond", "#22D3EE"))
            ui.show(_matrix_fig(pts).update_layout(title="Risk matrix (illustrative table)"))

        ui.section("Attack feasibility", "Attack-potential factors add up; a higher total means the attack is harder")
        ui.show(_potential_fig(a["potential_score"], r["potential_score"]))
        ui.table(pd.DataFrame([(f, lvl, tara.POTENTIAL[f][lvl]) for f, lvl in a["potential"].items()] + [("TOTAL", "", a["potential_score"])], columns=["Factor", "Level", "Points"]))

        ui.section("Mitigation what-if simulator", "Tick controls to see how the residual risk changes (uplift values are illustrative)")
        mc = st.columns(2)
        for i, mm in enumerate(base["mitigations"]):
            with mc[i % 2]:
                st.checkbox(f"{mm['id']}  {mm['control']}", key=f"mit_{attack}_{mm['id']}")
                st.caption(f"{mm['layer']} - {mm['detail']} (attack potential +{mm['uplift']}, impact -{mm['impact_drop']} level)")
        ui.kpis([("Inherent risk", f"{a['risk']} - {a['risk_label']}", f"{a['impact']} impact x {a['feasibility']} feasibility", tara.RISK_COLOR[a["risk"]]),
                 ("Residual risk", f"{r['risk']} - {tara.RISK_LABEL[r['risk']]}", f"{r['impact']} impact x {r['feasibility']} feasibility", tara.RISK_COLOR[r["risk"]]),
                 ("Risk reduction", f"{a['risk'] - r['risk']} level(s)", f"{len(sel)} control(s) applied", "#22D3EE")])

        ui.section("Evidence, verification and report")
        e1, e2 = st.columns(2)
        with e1:
            if ev:
                body = f"<ul><li>Frames flagged: <b>{ev['frames']:,}</b></li><li>Window: <b>{ev['start']:.2f}s - {ev['end']:.2f}s</b></li><li>Mean confidence: <b>{ev['mean_confidence']:.0%}</b></li></ul>"
            else:
                body = "<p>No alert of this class in the last live run. Run <b>Live Detection</b> to attach evidence.</p>"
            st.markdown(ui.card("Detection evidence", body), unsafe_allow_html=True)
        with e2:
            st.markdown(ui.card("Verification tests", "<ul>" + "".join(f"<li>{t}</li>" for t in a["tests"]) + "</ul>"), unsafe_allow_html=True)
        if st.button("Generate AI narrative", help="Uses Ollama (set the key in the sidebar). Falls back to a template."):
            with st.spinner("Drafting..."):
                st.session_state[f"narr_{attack}"] = narrative(a, ev)
        if f"narr_{attack}" in st.session_state:
            txt, src = st.session_state[f"narr_{attack}"]
            st.markdown(ui.card("AI-assisted narrative (draft)", f"<p>{txt}</p><p style='color:#94A3B8;font-size:.78rem'>Source: {src}. Facts come from the structured record above.</p>"), unsafe_allow_html=True)
        st.download_button("Download TARA entry (.md)", tara.to_markdown(a, ev), f"{a['id']}_{attack}.md")

    with t2:
        rows = []
        for k in tara.THREATS:
            x, y = tara.assess(k, []), tara.assess(k)
            rows.append({"ID": x["id"], "Threat": x["name"], "STRIDE": x["stride"], "Impact": x["impact"], "Attack potential": x["potential_score"], "Feasibility": x["feasibility"],
                         "Inherent risk": x["risk"], "Residual (all controls)": y["residual"]["risk"], "Treatment": x["treatment"], "Linked hazard": x["hazard"]})
        reg = pd.DataFrame(rows)
        ui.table(reg)
        pts = [(tara.THREATS[k]["id"], tara.assess(k, [])["impact"], tara.assess(k, [])["feasibility"], "circle", "#0A0F1C") for k in tara.THREATS]
        ui.show(_matrix_fig(pts).update_layout(title="All threats on the risk matrix"))
        st.download_button("Download register (CSV)", reg.to_csv(index=False), "tara_register.csv")
        ui.callout("Tables are <b>illustrative</b>. Calibrate impact criteria, feasibility thresholds and the risk matrix with your authorised copy of ISO/SAE 21434 and your organisation's risk policy.", "#FBBF24")
