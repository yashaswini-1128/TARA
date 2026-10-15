"""Load CAN captures: synthetic CSV or the real HCRL Car-Hacking CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, PAYLOAD_COLS, RAW_COLS


def load_frames(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_hcrl_csv(path: Path | str, attack_label: str, max_rows: int | None = None) -> pd.DataFrame:
    """Parse one HCRL Car-Hacking file, e.g. DoS_dataset.csv.

    Line format: timestamp, CAN ID (hex), DLC, DATA[0..DLC-1] (hex), flag (R/T)
    Flag T = injected frame -> ``attack_label``; R = regular -> "Normal".
    """
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            p = line.strip().split(",")
            if len(p) < 4:
                continue
            dlc = int(p[2])
            data = [int(x, 16) for x in p[3:3 + dlc]] + [0] * (8 - dlc)
            rows.append([float(p[0]), int(p[1], 16), dlc, *data[:8], attack_label if p[-1] == "T" else "Normal"])
    return pd.DataFrame(rows, columns=["timestamp", "can_id", "dlc", *PAYLOAD_COLS, "label"])


def merge_captures(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate separate captures into one continuous timeline."""
    out, offset = [], 0.0
    for f in frames:
        f = f.copy()
        f["timestamp"] = f["timestamp"] - f["timestamp"].min() + offset
        offset = f["timestamp"].max() + 0.01
        out.append(f)
    return pd.concat(out, ignore_index=True)
