"""Feature engineering for CAN frames.

Design rule: every feature is computable *online* from past frames only
(no look-ahead), and anything learned from data (ID whitelist, expected
period per ID) is fitted on the TRAINING split only -> no leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PAYLOAD_COLS, WINDOW

FEATURES = [
    "can_id", "dlc", *PAYLOAD_COLS,
    "payload_mean", "payload_std", "payload_entropy", "payload_zeros", "payload_delta", "bytes_changed", "payload_ham_min",
    "dt_prev", "dt_same_id", "dt_ratio", "id_known", "id_gap",
    "win_unique_ids", "win_id_entropy", "win_top_share", "win_same_id_frac", "win_mean_dt",
]
FEATURE_GROUPS = {
    "Raw frame": ["can_id", "dlc", *PAYLOAD_COLS],
    "Payload statistics": ["payload_mean", "payload_std", "payload_entropy", "payload_zeros", "payload_delta", "bytes_changed", "payload_ham_min"],
    "Timing": ["dt_prev", "dt_same_id", "dt_ratio", "win_mean_dt"],
    "ID behaviour": ["id_known", "id_gap", "win_unique_ids", "win_id_entropy", "win_top_share", "win_same_id_frac"],
}


def fit_baseline(train_normal: pd.DataFrame) -> dict:
    """Learn the 'normal' ID whitelist and the expected period of each ID."""
    g = train_normal.sort_values("timestamp").groupby("can_id")["timestamp"]
    period = g.apply(lambda s: float(np.median(np.diff(s.to_numpy())))).dropna()
    return {
        "known_ids": sorted(int(i) for i in period.index),
        "period": {str(int(k)): float(v) for k, v in period.items()},
        "median_period": float(period.median()),
    }


def save_baseline(b: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(b))


def load_baseline(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _entropy_rows(a: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a))
    for i in range(len(a)):
        _, c = np.unique(a[i], return_counts=True)
        p = c / c.sum()
        out[i] = -(p * np.log2(p)).sum()
    return out


def _rolling_id_stats(ids: np.ndarray, w: int) -> tuple[np.ndarray, ...]:
    """Unique-ID count, ID entropy, top-ID share and same-ID fraction over the last w frames."""
    n = len(ids)
    uniq = np.zeros(n); ent = np.zeros(n); top = np.zeros(n); same = np.zeros(n)
    counts: dict[int, int] = {}
    for i in range(n):
        counts[ids[i]] = counts.get(ids[i], 0) + 1
        if i >= w:
            old = ids[i - w]
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        size = min(i + 1, w)
        c = np.fromiter(counts.values(), dtype=float)
        p = c / size
        uniq[i] = len(c); ent[i] = -(p * np.log2(p)).sum(); top[i] = c.max() / size
        same[i] = counts[ids[i]] / size
    return uniq, ent, top, same


def build_features(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Turn raw frames (time-ordered) into the model feature matrix."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    X = pd.DataFrame(index=df.index)
    for c in ["can_id", "dlc", *PAYLOAD_COLS]:
        X[c] = df[c].astype(float)
    pay = df[PAYLOAD_COLS].to_numpy(dtype=float)
    X["payload_mean"] = pay.mean(1)
    X["payload_std"] = pay.std(1)
    X["payload_entropy"] = _entropy_rows(pay)
    X["payload_zeros"] = (pay == 0).sum(1)

    # how much did the payload change since the previous frame with the SAME id? (frozen forged payloads look different)
    diff = df.groupby("can_id")[PAYLOAD_COLS].diff().fillna(0.0)
    X["payload_delta"] = diff.abs().mean(axis=1)
    X["bytes_changed"] = (diff != 0).sum(axis=1).astype(float)

    # near-duplicate payloads: fewest differing bytes vs any of the last 4 frames with the same ID.
    # Forged frames repeat a frozen payload (value-agnostic, so it generalises to unseen bursts).
    grp, pay_np, ham = df.groupby("can_id")[PAYLOAD_COLS], df[PAYLOAD_COLS].to_numpy(), np.full(len(df), 8.0)
    for k in range(1, 5):
        ham = np.minimum(ham, (pay_np != grp.shift(k).to_numpy()).sum(1))
    X["payload_ham_min"] = ham

    ts = df["timestamp"]
    X["dt_prev"] = ts.diff().fillna(0.0).clip(lower=0)
    X["dt_same_id"] = ts.groupby(df["can_id"]).diff()
    med = baseline["median_period"]
    expected = df["can_id"].map(lambda i: baseline["period"].get(str(int(i)), med)).astype(float)
    X["dt_same_id"] = X["dt_same_id"].fillna(expected).clip(lower=0)
    X["dt_ratio"] = X["dt_same_id"] / expected.clip(lower=1e-6)
    known = set(baseline["known_ids"])
    X["id_known"] = df["can_id"].isin(known).astype(float)
    kn = np.array(sorted(known))
    X["id_gap"] = np.abs(df["can_id"].to_numpy()[:, None] - kn[None, :]).min(1) if len(kn) else 0.0

    u, e, t, s = _rolling_id_stats(df["can_id"].to_numpy(), WINDOW)
    X["win_unique_ids"], X["win_id_entropy"], X["win_top_share"], X["win_same_id_frac"] = u, e, t, s
    X["win_mean_dt"] = X["dt_prev"].rolling(WINDOW, min_periods=1).mean()
    return X[FEATURES]
