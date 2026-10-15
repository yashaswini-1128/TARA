"""Training + evaluation pipeline.

Steps: time-ordered split -> baseline (train only) -> features -> EDA stats ->
feature-selection / PCA study -> 6 supervised models + Isolation Forest ->
cross-validation -> hyper-parameter tuning -> final model -> evaluation ->
explainability -> leave-one-attack-out (zero-day) experiment -> artifacts.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (ARTIFACT_DIR, BASELINE_PATH, CLASSES, EDA_PATH, METRICS_PATH, MODEL_PATH,
                     MODEL_VERSION, SEED)
from .features import FEATURES, build_features, fit_baseline


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labs = [c for c in CLASSES if c in set(y_true)]  # macro metrics over classes present in the evaluation set
    atk_true, atk_pred = y_true != "Normal", y_pred != "Normal"
    kw = dict(labels=labs, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, **kw),
        "recall_macro": recall_score(y_true, y_pred, **kw),
        "f1_macro": f1_score(y_true, y_pred, **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "attack_recall": float((atk_pred & atk_true).sum() / max(atk_true.sum(), 1)),
        "false_alarm_rate": float((atk_pred & ~atk_true).sum() / max((~atk_true).sum(), 1)),
    }


def _stratified_subsample(X, y, n, seed=SEED):
    if len(X) <= n:
        return X, y
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(y), group_keys=False).apply(
        lambda s: s.sample(min(len(s), max(int(n * len(s) / len(y)), 300)), random_state=seed))
    idx = np.sort(idx.to_numpy())
    return X.iloc[idx], np.asarray(y)[idx]


def _eda_summary(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict:
    span = float(df.timestamp.max() - df.timestamp.min())
    top_ids = df.can_id.value_counts().head(14).index.tolist()
    id_by_class = (pd.crosstab(df.label, df.can_id.where(df.can_id.isin(top_ids), -1), normalize="index")
                   .reindex(CLASSES).fillna(0))
    desc = X.assign(label=y.values).groupby("label")[["dt_same_id", "payload_entropy", "win_unique_ids"]].median()
    return {
        "n_frames": len(df), "duration_s": span, "frames_per_s": len(df) / span,
        "unique_ids": int(df.can_id.nunique()), "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        "imbalance_ratio": float(df.label.value_counts().max() / df.label.value_counts().min()),
        "id_by_class": {"ids": [int(i) for i in id_by_class.columns], "classes": list(id_by_class.index),
                        "z": id_by_class.to_numpy().round(4).tolist()},
        "class_medians": desc.reindex(CLASSES).round(6).to_dict(),
        "corr": {"cols": FEATURES, "z": X[FEATURES].corr().fillna(0).round(3).to_numpy().tolist()},
    }


def _drift_reference(X_norm: pd.DataFrame, feats: list[str]) -> dict:
    ref = {}
    for f in feats:
        edges = np.unique(np.quantile(X_norm[f], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        counts, _ = np.histogram(X_norm[f], bins=np.r_[-np.inf, edges[1:-1], np.inf])
        ref[f] = {"edges": edges[1:-1].tolist(), "props": (counts / counts.sum()).tolist()}
    return ref


def run_training(df: pd.DataFrame, fast: bool = False, log=print) -> dict:
    t_start = time.time()
    df = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * 0.7)  # time-ordered split: train on the past, test on the future
    tr, te = df.iloc[:cut], df.iloc[cut:]
    baseline = fit_baseline(tr[tr.label == "Normal"])
    log(f"[1/8] split: train={len(tr):,} test={len(te):,}; baseline IDs={len(baseline['known_ids'])}")

    X_all = build_features(df, baseline)
    Xtr, Xte = X_all.iloc[:cut], X_all.iloc[cut:]
    ytr, yte = tr.label.to_numpy(), te.label.to_numpy()
    log("[2/8] features built")

    eda = _eda_summary(df, X_all, df.label)
    eda_sample, _ = _stratified_subsample(X_all.assign(label=df.label.values), df.label.values, 14000)
    eda_sample.to_csv(EDA_PATH, index=False)

    # ---- feature selection & dimensionality reduction study --------------------------------
    sub_X, sub_y = _stratified_subsample(Xtr, ytr, 30000)
    mi = mutual_info_classif(sub_X, sub_y, random_state=SEED)
    rf_probe = RandomForestClassifier(80, n_jobs=-1, class_weight="balanced", random_state=SEED).fit(sub_X, sub_y)
    imp = rf_probe.feature_importances_
    top_k = [FEATURES[i] for i in np.argsort(mi)[::-1][:12]]
    top_rf = [FEATURES[i] for i in np.argsort(imp)[::-1][:12]]
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(random_state=SEED).fit(scaler.transform(sub_X))
    n95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    pca_k = PCA(n_components=n95, random_state=SEED).fit(scaler.transform(Xtr))
    dimred = []

    def probe(name, Xa, Xb, n_feat):
        t0 = time.time()
        m = RandomForestClassifier(80, n_jobs=-1, class_weight="balanced", random_state=SEED).fit(Xa, ytr)
        dimred.append({"strategy": name, "n_features": n_feat,
                       "f1_macro": f1_score(yte, m.predict(Xb), average="macro", labels=[c for c in CLASSES if c in set(yte)], zero_division=0), "fit_s": time.time() - t0})

    probe("All features", Xtr, Xte, len(FEATURES))
    probe("Top-12 by mutual information", Xtr[top_k], Xte[top_k], 12)
    probe("Top-12 by RF importance", Xtr[top_rf], Xte[top_rf], 12)
    probe(f"PCA ({n95} comps, 95% variance)", pca_k.transform(scaler.transform(Xtr)),
          pca_k.transform(scaler.transform(Xte)), n95)
    pz = pca.transform(scaler.transform(sub_X))[:, :2]
    log("[3/8] feature-selection study done")

    # ---- model zoo -------------------------------------------------------------------------
    zoo = {
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced")), 40000),
        "K-Nearest Neighbours": (make_pipeline(StandardScaler(), KNeighborsClassifier(5, n_jobs=-1)), 25000),
        "Decision Tree": (DecisionTreeClassifier(max_depth=14, class_weight="balanced", random_state=SEED), None),
        "Random Forest": (RandomForestClassifier(100, max_depth=16, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED), None),
        "Gradient Boosting (Hist)": (HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=SEED), None),
        "SVM (RBF)": (make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced", random_state=SEED)), 9000),
    }
    # leakage-aware validation: each time chunk is one group, so a whole attack burst never straddles train/validation
    gtr = pd.Series(np.minimum((tr.timestamp.to_numpy() / (tr.timestamp.max() + 1e-9) * 10).astype(int), 9), index=Xtr.index)
    cv_X, cv_y = _stratified_subsample(Xtr, ytr, 12000 if fast else 20000)
    g_cv = gtr.loc[cv_X.index].to_numpy()
    skf = GroupKFold(5)
    comparison = []
    for name, (model, cap) in zoo.items():
        Xf, yf = (_stratified_subsample(Xtr, ytr, cap) if cap else (Xtr, ytr))
        t0 = time.time(); model.fit(Xf, yf); fit_s = time.time() - t0
        t0 = time.time(); pred = model.predict(Xte); pred_s = time.time() - t0
        cv = cross_val_score(model, cv_X, cv_y, cv=skf, groups=g_cv, scoring="f1_macro", n_jobs=1)
        comparison.append({"model": name, **_scores(yte, pred), "cv_f1_mean": cv.mean(), "cv_f1_std": cv.std(),
                           "fit_s": fit_s, "predict_s": pred_s, "train_rows": len(Xf)})
        log(f"      {name:26s} F1={comparison[-1]['f1_macro']:.4f}  cv={cv.mean():.4f}")
    log("[4/8] model comparison done")

    # ---- hyper-parameter tuning --------------------------------------------------------------
    tune_X, tune_y = _stratified_subsample(Xtr, ytr, 15000 if fast else 30000)
    spaces = {
        "Random Forest": (RandomForestClassifier(n_jobs=-1, random_state=SEED), {
            "n_estimators": [60, 100, 150], "max_depth": [10, 14, 18], "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.5], "class_weight": ["balanced", "balanced_subsample"]}),
        "Gradient Boosting (Hist)": (HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), {
            "learning_rate": [0.05, 0.1, 0.2], "max_iter": [100, 200], "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0]}),
    }
    tuned, tuning_log = {}, {}
    for name, (est, grid) in spaces.items():
        rs = RandomizedSearchCV(est, grid, n_iter=6 if fast else 8, cv=GroupKFold(3), scoring="f1_macro", random_state=SEED, n_jobs=1)
        rs.fit(tune_X, tune_y, groups=gtr.loc[tune_X.index].to_numpy())
        tuned[name] = (rs.best_estimator_, rs.best_score_)
        res = pd.DataFrame(rs.cv_results_)
        tuning_log[name] = {"best_params": rs.best_params_, "best_cv_f1": rs.best_score_,
                            "trials": [{**p, "cv_f1": s} for p, s in zip(res.params, res.mean_test_score)]}
    best_name = max(tuned, key=lambda k: tuned[k][1])  # chosen by CV score, never by the test set
    final = tuned[best_name][0].fit(Xtr, ytr)
    log(f"[5/8] tuning done -> {best_name} {tuning_log[best_name]['best_params']}")

    # ---- final evaluation ---------------------------------------------------------------------
    proba, classes = final.predict_proba(Xte), list(final.classes_)
    pred = final.predict(Xte)
    final_scores = _scores(yte, pred)
    comparison.append({"model": f"{best_name} (tuned)", **final_scores, "cv_f1_mean": tuned[best_name][1], "cv_f1_std": 0.0,
                       "fit_s": 0.0, "predict_s": 0.0, "train_rows": len(Xtr)})
    cm = confusion_matrix(yte, pred, labels=CLASSES)
    rep = classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0)
    roc = {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(yte == c, proba[:, i])
        sel = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
        roc[c] = {"fpr": fpr[sel], "tpr": tpr[sel], "auc": auc(fpr, tpr)}
    t0 = time.time(); final.predict(Xte.iloc[:5000]); lat_ms = (time.time() - t0) / 5000 * 1000

    ps = _stratified_subsample(Xte, yte, 6000)
    perm = permutation_importance(final, ps[0], ps[1], n_repeats=3, random_state=SEED, scoring="f1_macro", n_jobs=1)
    log("[6/8] evaluation done")

    # ---- leave-one-attack-out (zero-day) experiment ------------------------------------------------
    iso_scaler = StandardScaler().fit(Xtr[ytr == "Normal"])
    iso = IsolationForest(n_estimators=150, contamination=0.01, random_state=SEED, n_jobs=-1).fit(iso_scaler.transform(Xtr[ytr == "Normal"]))
    iso_flag = iso.predict(iso_scaler.transform(Xte)) == -1
    iso_fpr = float(iso_flag[yte == "Normal"].mean())
    zero_day = []
    for atk in CLASSES[1:]:
        keep = ytr != atk
        m = RandomForestClassifier(60, max_depth=14, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED).fit(Xtr[keep], ytr[keep])
        mask = yte == atk
        sup = float((m.predict(Xte[mask]) != "Normal").mean()) if mask.any() else float("nan")
        zero_day.append({"unseen_attack": atk, "supervised_rf_detect": sup, "isolation_forest_detect": float(iso_flag[mask].mean()) if mask.any() else float("nan")})
    log("[7/8] zero-day experiment done")

    # ---- artifacts ------------------------------------------------------------------------------------
    baseline["drift_ref"] = _drift_reference(Xtr[ytr == "Normal"], ["dt_same_id", "dt_ratio", "win_unique_ids", "win_id_entropy", "win_top_share", "payload_entropy", "win_mean_dt"])
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "classes": classes, "version": MODEL_VERSION}, MODEL_PATH, compress=3)
    BASELINE_PATH.write_text(json.dumps(baseline))
    metrics = {
        "meta": {"version": MODEL_VERSION, "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "seed": SEED, "sklearn": sklearn.__version__, "python": platform.python_version(),
                 "best_model": best_name, "best_params": tuning_log[best_name]["best_params"],
                 "train_rows": len(Xtr), "test_rows": len(Xte), "train_s": time.time() - t_start,
                 "latency_ms_per_frame": lat_ms, "n_features": len(FEATURES)},
        "split": {"train": pd.Series(ytr).value_counts().to_dict(), "test": pd.Series(yte).value_counts().to_dict()},
        "eda": eda,
        "features": {"mi": dict(zip(FEATURES, mi)), "rf_importance": dict(zip(FEATURES, imp)),
                     "perm_importance": dict(zip(FEATURES, perm.importances_mean)),
                     "perm_importance_std": dict(zip(FEATURES, perm.importances_std)),
                     "pca_var": pca.explained_variance_ratio_, "pca_n95": n95,
                     "pca_scatter": {"x": pz[:2500, 0], "y": pz[:2500, 1], "label": sub_y[:2500]},
                     "dimred": dimred, "top_mi": top_k, "top_rf": top_rf},
        "comparison": comparison, "tuning": tuning_log,
        "final": {"scores": final_scores, "cm": cm, "classes": CLASSES, "report": rep, "roc": roc},
        "zero_day": {"rows": zero_day, "iso_false_alarm": iso_fpr},
    }
    METRICS_PATH.write_text(json.dumps(_clean(metrics)))
    log(f"[8/8] artifacts saved ({time.time() - t_start:.0f}s total)  final F1={final_scores['f1_macro']:.4f}")
    return metrics
