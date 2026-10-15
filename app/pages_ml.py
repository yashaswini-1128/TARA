"""EDA, features/models and evaluation pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import ui
from src.config import CLASSES, COLORS, EDA_PATH
from src.features import FEATURE_GROUPS, FEATURES


@st.cache_data(show_spinner=False)
def _eda_sample(mtime: float) -> pd.DataFrame:
    return pd.read_csv(EDA_PATH)


def render_eda(m: dict) -> None:
    e = m["eda"]
    ui.hero("Data & EDA", "What the bus traffic looks like, how imbalanced it is, and which signals separate attacks from normal behaviour.")
    ui.kpis([("Frames", f"{e['n_frames']:,}", f"{e['duration_s']:.0f} s capture", "#22D3EE"), ("Unique CAN IDs", e["unique_ids"], "legitimate set + injected", "#8B5CF6"),
             ("Missing values", e["missing_values"], "none to impute", "#34D399"), ("Duplicate rows", f"{e['duplicate_rows']:,}", "kept: periodic frames repeat legitimately", "#FBBF24"),
             ("Imbalance ratio", f"{e['imbalance_ratio']:.0f}:1", "largest / smallest class", "#F87171")])
    df = _eda_sample(EDA_PATH.stat().st_mtime)
    t1, t2, t3, t4 = st.tabs(["Class balance", "Timing", "ID behaviour", "Correlation & PCA"])

    with t1:
        counts = e["class_counts"]
        c1, c2 = st.columns([1, 1.3])
        pie = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.62, marker=dict(colors=[COLORS[c] for c in counts]), textinfo="percent"))
        bar = go.Figure(go.Bar(x=list(counts), y=list(counts.values()), marker_color=[COLORS[c] for c in counts], text=[f"{v:,}" for v in counts.values()], textposition="outside"))
        bar.update_yaxes(type="log", title="frames (log scale)")
        with c1:
            ui.show(ui.style(pie, 340).update_layout(title="Share of frames"))
        with c2:
            ui.show(ui.style(bar, 340, False).update_layout(title="Frames per class (log scale)"))
        sp = pd.DataFrame({"Class": CLASSES, "Train": [m["split"]["train"].get(c, 0) for c in CLASSES], "Test (future)": [m["split"]["test"].get(c, 0) for c in CLASSES]})
        ui.table(sp)
        ui.callout("<b>Imbalance handling:</b> class-weighted losses (<span class='mono'>class_weight='balanced'</span>), macro-averaged metrics and per-class recall - "
                   "accuracy alone would hide missed Spoof attacks because they are under 2% of frames. SMOTE was avoided on purpose: synthetic oversampling of time-series frames "
                   "would invent timing patterns that never occur on a real bus.")

    with t2:
        d = df.assign(log_ratio=np.log10(df.dt_ratio.clip(lower=1e-3)))
        f = go.Figure()
        for c in CLASSES:
            f.add_trace(go.Box(y=d[d.label == c].log_ratio, name=c, marker_color=COLORS[c], boxpoints=False))
        f.update_yaxes(title="log10( gap since same ID / expected period )")
        ui.show(ui.style(f, 400, False).update_layout(title="Timing regularity: who breaks the expected period?"))
        med = pd.DataFrame(e["class_medians"]).round(5).reset_index().rename(columns={"index": "class"})
        ui.table(med)
        ui.callout("Normal frames sit near <b>0</b> (arrive on schedule). DoS frames arrive thousands of times faster than the ID's period; Spoof frames arrive roughly "
                   "3x too often on a legitimate ID - invisible to an ID whitelist, obvious to a timing feature.")

    with t3:
        z = e["id_by_class"]
        labels = ["other" if i == -1 else f"0x{i:03X}" for i in z["ids"]]
        h = go.Figure(go.Heatmap(z=z["z"], x=labels, y=z["classes"], colorscale=[[0, "#0F172A"], [1, "#22D3EE"]], colorbar=dict(title="share")))
        ui.show(ui.style(h, 360, False).update_layout(title="CAN-ID mix per class (top IDs)"))
        ui.callout("DoS lives on ID <span class='mono'>0x000</span> (not in the normal set); Fuzzy spreads over many IDs (high ID entropy); Spoof classes hide on "
                   "<span class='mono'>0x316</span> / <span class='mono'>0x43F</span>, which are legitimate - so ID alone cannot catch them.")

    with t4:
        c1, c2 = st.columns(2)
        corr = e["corr"]
        hm = go.Figure(go.Heatmap(z=corr["z"], x=corr["cols"], y=corr["cols"], colorscale="RdBu", zmid=0, zmin=-1, zmax=1))
        hm.update_yaxes(autorange="reversed")
        with c1:
            ui.show(ui.style(hm, 470, False).update_layout(title="Feature correlation"))
        pc = m["features"]["pca_scatter"]
        sc = go.Figure()
        for c in CLASSES:
            idx = [i for i, l in enumerate(pc["label"]) if l == c]
            sc.add_trace(go.Scattergl(x=[pc["x"][i] for i in idx], y=[pc["y"][i] for i in idx], mode="markers", name=c, marker=dict(color=COLORS[c], size=5, opacity=.7)))
        with c2:
            ui.show(ui.style(sc, 470).update_layout(title="PCA projection (2 components)", xaxis_title="PC1", yaxis_title="PC2"))
        ui.callout("Highly correlated groups (payload mean/std, rolling-window stats) justify the feature-selection and PCA study on the next page. "
                   "Classes overlap in 2-D, so a non-linear model is needed.")

    ui.section("Preprocessing pipeline")
    ui.table(pd.DataFrame([
        ("1", "Schema validation", "Required columns checked; clear error otherwise"), ("2", "Time ordering", "Frames sorted by timestamp"),
        ("3", "Time-ordered split", "First 70% = train, last 30% = test (no shuffling across time)"),
        ("4", "Baseline fitting", "ID whitelist + expected period per ID learned from TRAIN-normal frames only"),
        ("5", "Online features", "Every feature uses only past frames (no look-ahead)"),
        ("6", "Scaling", "StandardScaler inside pipelines for LR / KNN / SVM; trees are scale-invariant")], columns=["#", "Step", "Detail"]))


def render_models(m: dict) -> None:
    f = m["features"]
    ui.hero("Features & Models", "Feature engineering, selection and dimensionality reduction, then six supervised models and an unsupervised detector.")
    ui.section("Engineered features", f"{len(FEATURES)} features in four groups - all computable in real time")
    desc = {"Raw frame": "ID, DLC and the 8 payload bytes as received.",
            "Payload statistics": "Mean, std, entropy and zero-byte count of the payload, plus change vs the previous same-ID frame and the closest repeat among the last 4 - random payloads have high entropy, DoS payloads are all zero, forged payloads are frozen duplicates.",
            "Timing": "Gap to previous frame, gap to previous frame of the same ID, ratio to that ID's expected period, rolling mean gap.",
            "ID behaviour": "Is the ID in the learned whitelist, distance to nearest known ID, plus rolling-window unique-ID count, ID entropy, top-ID share and same-ID fraction."}
    cols = st.columns(4)
    for col, (g, fs) in zip(cols, FEATURE_GROUPS.items()):
        col.markdown(ui.card(f"{g} ({len(fs)})", f"<p>{desc[g]}</p>"), unsafe_allow_html=True)

    ui.section("Feature selection", "Mutual information (model-free) vs Random-Forest importance (model-based)")
    imp = pd.DataFrame({"feature": list(f["rf_importance"]), "RF importance": list(f["rf_importance"].values()), "Mutual information": list(f["mi"].values())})
    imp["RF importance"] /= imp["RF importance"].max(); imp["Mutual information"] /= imp["Mutual information"].max()
    imp = imp.sort_values("RF importance", ascending=False).head(15).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=imp.feature, x=imp["RF importance"], orientation="h", name="RF importance", marker_color="#22D3EE")
    fig.add_bar(y=imp.feature, x=imp["Mutual information"], orientation="h", name="Mutual information", marker_color="#8B5CF6")
    ui.show(ui.style(fig, 470).update_layout(barmode="group", title="Top 15 features (normalised to 1)"))

    c1, c2 = st.columns(2)
    pv = np.cumsum(f["pca_var"])
    pf = go.Figure(go.Scatter(x=list(range(1, len(pv) + 1)), y=pv, mode="lines+markers", line=dict(color="#22D3EE", width=3), name="cumulative variance"))
    pf.add_hline(y=.95, line_dash="dash", line_color="#FBBF24", annotation_text="95%")
    pf.update_layout(xaxis_title="components", yaxis_title="explained variance")
    with c1:
        ui.show(ui.style(pf, 340, False).update_layout(title=f"PCA: {f['pca_n95']} components keep 95% variance"))
    dr = pd.DataFrame(f["dimred"])
    bf = go.Figure(go.Bar(x=dr.strategy, y=dr.f1_macro, marker_color=["#22D3EE", "#8B5CF6", "#A78BFA", "#F472B6"], text=[f"{v:.3f}" for v in dr.f1_macro], textposition="outside"))
    bf.update_yaxes(range=[max(0, dr.f1_macro.min() - .08), min(1.02, dr.f1_macro.max() + .04)], title="macro F1 (test)")
    with c2:
        ui.show(ui.style(bf, 340, False).update_layout(title="Does dimensionality reduction help?"))
    dd = dr.rename(columns={"strategy": "Strategy", "n_features": "Features", "f1_macro": "Macro F1", "fit_s": "Fit time (s)"}).round(3)
    ui.table(dd)
    ui.callout("Same Random Forest, four input spaces. Interpretable selected features keep nearly all accuracy with half the inputs; PCA compresses but loses the "
               "per-feature meaning engineers need for a TARA justification - so the final model uses selected, named features.")

    ui.callout("<b>Feature ablation (found through error analysis).</b> Spoof_Gear was being missed because each forged burst carries a different frozen payload. "
               "Adding <span class='mono'>payload_ham_min</span> (near-duplicate of a recent same-ID frame) lifted macro F1 on 8 unseen simulator seeds from <b>0.841</b> (24 base features) to "
               "<b>0.972</b>, and to <b>0.987</b> (std 0.022 vs 0.071) with the payload-change features added. Measured during development on synthetic data - reproduce with "
               "<span class='mono'>python scripts/feature_ablation.py</span>.", "#8B5CF6")

    ui.section("Model comparison", "Six supervised models + tuned winner. CV = group-based (whole bursts held out); Test = future, unseen time period")
    cmp = pd.DataFrame(m["comparison"])
    fig = go.Figure()
    fig.add_bar(x=cmp.model, y=cmp.cv_f1_mean, name="Group-CV macro F1", marker_color="#8B5CF6")
    fig.add_bar(x=cmp.model, y=cmp.f1_macro, name="Test macro F1 (future data)", marker_color="#22D3EE")
    fig.update_yaxes(range=[max(0, min(cmp.f1_macro.min(), cmp.cv_f1_mean.min()) - .05), 1.01])
    ui.show(ui.style(fig, 400).update_layout(barmode="group", title="Validation vs future-data performance"))
    tbl = cmp[["model", "accuracy", "f1_macro", "attack_recall", "false_alarm_rate", "cv_f1_mean", "cv_f1_std", "fit_s", "predict_s", "train_rows"]].copy()
    tbl.columns = ["Model", "Accuracy", "Macro F1", "Attack recall", "False alarms", "CV F1", "CV std", "Fit s", "Predict s", "Train rows"]
    ui.table(tbl.round(4))
    why = {"Logistic Regression": "Linear baseline; fast, interpretable coefficients.", "K-Nearest Neighbours": "Instance-based; shows local structure, costly at inference.",
           "Decision Tree": "Rule-like, fully explainable; prone to overfit bursts.", "Random Forest": "Bagged trees; robust, gives importances, fast CPU inference.",
           "Gradient Boosting (Hist)": "Boosted trees; strong accuracy on tabular data.", "SVM (RBF)": "Maximum-margin non-linear boundary; trained on a subsample (cost grows ~n^2)."}
    with st.expander("Why these models? (approach selection)"):
        for k, v in why.items():
            st.markdown(f"**{k}** - {v}")
        st.markdown("**Isolation Forest** - unsupervised; trained on normal traffic only, so it can flag attacks it has never seen (see Evaluation).")

    ui.section("Hyper-parameter optimisation", "Randomised search, group-CV, scored on macro F1 - the final model is chosen by CV score, never by the test set")
    c1, c2 = st.columns(2)
    for col, name in zip((c1, c2), m["tuning"]):
        t = m["tuning"][name]
        trials = pd.DataFrame(t["trials"]).sort_values("cv_f1", ascending=False).round(4)
        trials = trials.astype({c: str for c in trials.columns if c != "cv_f1"})  # mixed str/float params break Arrow
        col.markdown(f"**{name}** - best CV F1 **{t['best_cv_f1']:.4f}**")
        col.json(t["best_params"], expanded=False)
        col.dataframe(trials.head(6), hide_index=True)
    ui.callout(f"Selected: <b>{m['meta']['best_model']}</b> with <span class='mono'>{m['meta']['best_params']}</span>.")


def render_eval(m: dict) -> None:
    fin = m["final"]
    ui.hero("Evaluation", "Performance on a held-out future time window, per-class behaviour, explainability and the zero-day experiment.")
    s = fin["scores"]
    ui.kpis([("Accuracy", f"{s['accuracy']:.4f}", "all classes", "#22D3EE"), ("Macro F1", f"{s['f1_macro']:.4f}", "equal weight per class", "#8B5CF6"),
             ("Attack recall", f"{s['attack_recall']:.2%}", "any attack vs normal", "#34D399"), ("False-alarm rate", f"{s['false_alarm_rate']:.3%}", "normal flagged as attack", "#FBBF24")])
    c1, c2 = st.columns([1.1, 1])
    cm = np.array(fin["cm"], dtype=float)
    norm = st.radio("Confusion matrix", ["Row-normalised (recall)", "Counts"], horizontal=True, label_visibility="collapsed")
    z = cm / cm.sum(1, keepdims=True).clip(min=1) if norm.startswith("Row") else cm
    txt = [[f"{v:.1%}" if norm.startswith("Row") else f"{int(v):,}" for v in r] for r in z]
    hm = go.Figure(go.Heatmap(z=z, x=fin["classes"], y=fin["classes"], text=txt, texttemplate="%{text}", colorscale=[[0, "#0F172A"], [1, "#22D3EE"]], showscale=False))
    hm.update_yaxes(autorange="reversed", title="true"); hm.update_xaxes(title="predicted")
    with c1:
        ui.show(ui.style(hm, 400, False).update_layout(title="Confusion matrix"))
    rep = pd.DataFrame({c: fin["report"][c] for c in fin["classes"]}).T[["precision", "recall", "f1-score", "support"]].round(4)
    with c2:
        st.markdown("**Per-class report**")
        st.dataframe(rep)
        ui.callout("Read the <b>recall</b> column first: a missed attack is the costly error in a safety-relevant system. Small classes (Spoof) are where models differ.")

    roc = go.Figure()
    for c, r in fin["roc"].items():
        roc.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=f"{c} (AUC {r['auc']:.3f})", line=dict(color=COLORS[c], width=2.5)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#475569", dash="dash"), showlegend=False))
    roc.update_layout(xaxis_title="false-positive rate", yaxis_title="true-positive rate")
    p = m["features"]["perm_importance"]
    pi = pd.DataFrame({"f": list(p), "v": list(p.values()), "e": list(m["features"]["perm_importance_std"].values())}).sort_values("v").tail(12)
    pf = go.Figure(go.Bar(y=pi.f, x=pi.v, error_x=dict(type="data", array=pi.e), orientation="h", marker_color="#8B5CF6"))
    pf.update_xaxes(title="drop in macro F1 when shuffled")
    a, b = st.columns(2)
    with a:
        ui.show(ui.style(roc, 400).update_layout(title="ROC curves (one-vs-rest)"))
    with b:
        ui.show(ui.style(pf, 400, False).update_layout(title="Permutation importance (explainability)"))

    ui.section("Zero-day experiment", "Leave-one-attack-out: what happens when an attack type was never seen in training?")
    zd = pd.DataFrame(m["zero_day"]["rows"])
    fz = go.Figure()
    fz.add_bar(x=zd.unseen_attack, y=zd.supervised_rf_detect, name="Supervised Random Forest", marker_color="#22D3EE")
    fz.add_bar(x=zd.unseen_attack, y=zd.isolation_forest_detect, name="Isolation Forest (normal-only)", marker_color="#F472B6")
    fz.update_yaxes(range=[0, 1.05], title="share of unseen-attack frames flagged")
    ui.show(ui.style(fz, 380).update_layout(barmode="group", title="Detection of attacks absent from training"))
    ui.callout(f"The supervised model is excellent on known attacks but can miss families it never saw. The Isolation Forest (false-alarm rate on normal traffic: "
               f"<b>{m['zero_day']['iso_false_alarm']:.1%}</b>) needs no attack labels. Production design: supervised classifier for known attacks <b>plus</b> an anomaly detector as a safety net.", "#F472B6")

    ui.section("How the model was validated")
    ui.table(pd.DataFrame([
        ("Time-ordered hold-out", "Train on the first 70% of the capture, test on the last 30% - the realistic 'deploy on tomorrow's traffic' setting."),
        ("Group K-fold CV", "Time chunks are groups, so a whole attack burst is never split across train and validation (plain shuffled CV scored ~0.99 for every model - leakage)."),
        ("Selection by CV, not test", "Model and hyper-parameters chosen on validation folds; the test set is touched once."),
        ("Baseline fitted on train only", "ID whitelist and expected periods never see test traffic."),
        ("Metrics", "Macro F1, per-class recall, attack recall, false-alarm rate, ROC-AUC, latency.")], columns=["Safeguard", "What we did"]))
