"""Inference, alert debouncing and drift monitoring."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import BASELINE_PATH, METRICS_PATH, MODEL_PATH, RAW_COLS
from .features import build_features, load_baseline

REQUIRED = [c for c in RAW_COLS if c != "label"]


class Detector:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.model, self.features, self.classes = bundle["model"], bundle["features"], bundle["classes"]
        self.version = bundle["version"]
        self.baseline = load_baseline(BASELINE_PATH)

    def predict(self, frames: pd.DataFrame) -> pd.DataFrame:
        miss = set(REQUIRED) - set(frames.columns)
        if miss:
            raise ValueError(f"Missing columns: {sorted(miss)}")
        frames = frames.sort_values("timestamp").reset_index(drop=True)
        X = build_features(frames, self.baseline)
        proba = self.model.predict_proba(X[self.features])
        out = frames.copy()
        out["pred"] = np.array(self.classes)[proba.argmax(1)]
        out["confidence"] = proba.max(1)
        out["p_attack"] = 1 - proba[:, self.classes.index("Normal")]
        for i, c in enumerate(self.classes):
            out[f"p_{c}"] = proba[:, i]
        out.attrs["features"] = X
        return out


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()


def artifacts_ready() -> bool:
    return MODEL_PATH.exists() and BASELINE_PATH.exists() and METRICS_PATH.exists()


def alerts(pred: pd.DataFrame, gap: float = 0.05, min_frames: int = 5) -> list[dict]:
    """Debounce frame-level predictions into alerts: >= min_frames flagged frames of one class within `gap` seconds."""
    a = pred[pred.pred != "Normal"]
    out = []
    for cls, g in a.groupby("pred"):
        t = g.timestamp.to_numpy()
        if len(t) == 0:
            continue
        cuts = np.where(np.diff(t) > gap)[0] + 1
        for seg in np.split(np.arange(len(t)), cuts):
            if len(seg) >= min_frames:
                out.append({"class": cls, "start": float(t[seg[0]]), "end": float(t[seg[-1]]), "frames": int(len(seg)),
                            "mean_confidence": float(g.confidence.to_numpy()[seg].mean())})
    return sorted(out, key=lambda r: r["start"])


def psi(actual: pd.Series, ref: dict) -> float:
    """Population Stability Index of a live feature vs the training-normal reference."""
    edges = np.r_[-np.inf, ref["edges"], np.inf]
    cnt, _ = np.histogram(actual, bins=edges)
    p = np.clip(cnt / max(cnt.sum(), 1), 1e-4, None)
    q = np.clip(np.array(ref["props"]), 1e-4, None)
    return float(((p - q) * np.log(p / q)).sum())


def drift_report(X: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    rows = []
    for f, ref in baseline.get("drift_ref", {}).items():
        v = psi(X[f], ref)
        rows.append({"feature": f, "psi": v, "status": "stable" if v < 0.1 else "watch" if v < 0.25 else "shift"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
