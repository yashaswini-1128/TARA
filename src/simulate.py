"""Synthetic CAN-bus traffic generator.

Mirrors the schema of the public HCRL *Car-Hacking* dataset
(timestamp, CAN ID, DLC, 8 data bytes, label) so real captures can be
dropped in later without touching the rest of the pipeline.

Attack models
  DoS        - highest-priority ID 0x000 flooded every ~0.3 ms
  Fuzzy      - random IDs / random payloads injected (some IDs overlap legit ones)
  Spoof_RPM  - forged engine-speed frames on legit ID 0x316
  Spoof_Gear - forged gear-position frames on legit ID 0x43F
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ATTACKS, RAW_COLS

# (can_id, period_seconds) for the legitimate periodic messages
LEGIT_IDS = [
    (0x0A0, 0.010), (0x0B0, 0.010), (0x130, 0.010), (0x140, 0.010), (0x153, 0.010),
    (0x164, 0.020), (0x18F, 0.020), (0x1F1, 0.020), (0x220, 0.020), (0x260, 0.020),
    (0x2A0, 0.020), (0x2B0, 0.020), (0x316, 0.010), (0x329, 0.020), (0x350, 0.050),
    (0x370, 0.050), (0x43F, 0.010), (0x440, 0.050), (0x4B0, 0.100), (0x4F0, 0.100),
    (0x545, 0.100), (0x5A0, 0.100), (0x5F0, 0.200), (0x690, 0.200),
]


def _frame(t, ids, data) -> pd.DataFrame:
    df = pd.DataFrame(np.asarray(data).astype(int), columns=[f"d{i}" for i in range(8)])
    df.insert(0, "dlc", 8)
    df.insert(0, "can_id", np.asarray(ids).astype(int))
    df.insert(0, "timestamp", np.asarray(t, dtype=float))
    return df


def _legit_stream(duration: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cid, period in LEGIT_IDS:
        n = int(duration / period)
        t = np.arange(n) * period + rng.uniform(0, period) + rng.normal(0, period * 0.02, n)
        base = rng.integers(0, 256, 8)
        walk = np.cumsum(rng.normal(0, 1.2, (n, 8)), axis=0)
        data = (base + walk) % 256
        const_mask = rng.random(8) < 0.4          # some bytes never change
        data[:, const_mask] = base[const_mask]
        data[:, 6] = np.arange(n) % 16             # rolling counter byte
        rows.append(_frame(t, np.full(n, cid), data))
    df = pd.concat(rows, ignore_index=True)
    df["label"] = "Normal"
    return df


def _bursts(duration: float, rng: np.random.Generator, n_bursts: int, dur=(1.0, 2.5)):
    """One burst per equal time segment, so every attack appears in both the train and test periods."""
    edges = np.linspace(1.0, max(duration - 4.0, 2.0), n_bursts + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s0 = rng.uniform(a, max(a, b - dur[1]))
        out.append((s0, s0 + rng.uniform(*dur)))
    return out


def _attack_frames(kind: str, t0: float, t1: float, rng: np.random.Generator, intensity: float) -> pd.DataFrame:
    """intensity in (0,1]: lower = stealthier (sparser injections)."""
    if kind == "DoS":
        step = 0.0003 / intensity
        t = np.arange(t0, t1, step)
        ids, data = np.zeros(len(t)), np.zeros((len(t), 8))
    elif kind == "Fuzzy":
        step = 0.0006 / intensity
        t = np.arange(t0, t1, step)
        n = len(t)
        legit = np.array([c for c, _ in LEGIT_IDS])
        ids = np.where(rng.random(n) < 0.3, rng.choice(legit, n), rng.integers(0, 0x800, n))
        data = rng.integers(0, 256, (n, 8))
    else:  # Spoof_RPM / Spoof_Gear: forged frames on a legit ID with a frozen payload
        step = 0.003 / intensity
        t = np.arange(t0, t1, step)
        t = t + rng.normal(0, step * 0.1, len(t))
        n = len(t)
        ids = np.full(n, 0x316 if kind == "Spoof_RPM" else 0x43F)
        base = rng.integers(180, 256, 8) if kind == "Spoof_RPM" else rng.integers(0, 256, 8)
        data = np.tile(base, (n, 1))
        data[:, 6] = rng.integers(0, 16, n)
    df = _frame(t, ids, data)
    df["label"] = kind
    return df


def generate(duration: float = 90.0, seed: int = 42, attacks: list[str] | None = None,
             bursts_per_attack: int = 4, intensity: float = 1.0) -> pd.DataFrame:
    """Generate a time-ordered CAN capture with labelled attack bursts."""
    rng = np.random.default_rng(seed)
    attacks = ATTACKS if attacks is None else attacks
    parts = [_legit_stream(duration, rng)]
    for kind in attacks:
        for s, e in _bursts(duration, rng, bursts_per_attack):
            inten = float(np.clip(intensity * rng.uniform(0.35, 1.0), 0.1, 1.0))  # loud + stealthy bursts
            parts.append(_attack_frames(kind, s, e, rng, inten))
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    return df[RAW_COLS]
