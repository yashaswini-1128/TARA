"""Feature-set ablation across simulator seeds:  python scripts/feature_ablation.py [--seeds 8]"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

from src.config import CLASSES
from src.features import FEATURES, build_features, fit_baseline
from src.simulate import generate

DELTA = ["payload_delta", "bytes_changed"]
BASE = [f for f in FEATURES if f not in DELTA + ["payload_ham_min"]]
SETS = {"A: base": BASE, "B: +delta": BASE + DELTA, "C: +ham_min": BASE + ["payload_ham_min"], "D: all": FEATURES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    n = ap.parse_args().seeds
    rows = []
    for seed in range(101, 101 + n):
        df = generate(60, seed)
        cut = int(len(df) * 0.7)
        tr, te = df.iloc[:cut], df.iloc[cut:]
        X = build_features(df, fit_baseline(tr[tr.label == "Normal"]))
        labs = [c for c in CLASSES if c in set(te.label)]
        for name, cols in SETS.items():
            m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.15, class_weight="balanced", random_state=0).fit(X.iloc[:cut][cols], tr.label)
            p = m.predict(X.iloc[cut:][cols])
            rec = dict(zip(labs, recall_score(te.label, p, labels=labs, average=None, zero_division=0)))
            rows.append({"set": name, "macro_f1": f1_score(te.label, p, labels=labs, average="macro"), "gear_recall": rec["Spoof_Gear"]})
    print(pd.DataFrame(rows).groupby("set").agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
