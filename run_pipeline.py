"""One command to generate data and train:  python run_pipeline.py [--fast] [--real-dir PATH]"""
import argparse
from pathlib import Path

from src.config import DATA_DIR, DATA_PATH
from src.data_io import load_frames, load_hcrl_csv, merge_captures
from src.simulate import generate
from src.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=90.0, help="seconds of synthetic traffic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="smaller CV/tuning budgets")
    ap.add_argument("--real-dir", type=Path, help="folder with HCRL DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv, RPM_dataset.csv")
    ap.add_argument("--max-rows", type=int, default=300000, help="rows to read per real file")
    a = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if a.real_dir:
        files = {"DoS_dataset.csv": "DoS", "Fuzzy_dataset.csv": "Fuzzy", "gear_dataset.csv": "Spoof_Gear", "RPM_dataset.csv": "Spoof_RPM"}
        df = merge_captures([load_hcrl_csv(a.real_dir / f, lab, a.max_rows) for f, lab in files.items()])
        print(f"Loaded real HCRL data: {len(df):,} frames")
    else:
        df = generate(a.duration, a.seed)
        print(f"Generated synthetic traffic: {len(df):,} frames")
    df.to_csv(DATA_PATH, index=False)
    run_training(load_frames(DATA_PATH), fast=a.fast)


if __name__ == "__main__":
    main()
