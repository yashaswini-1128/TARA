"""Central configuration: paths, class names, colours, seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_PATH = DATA_DIR / "can_frames.csv.gz"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
BASELINE_PATH = ARTIFACT_DIR / "baseline.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
EDA_PATH = ARTIFACT_DIR / "eda_sample.csv.gz"

SEED = 42
MODEL_VERSION = "1.0.0"
WINDOW = 32  # rolling window (frames) used by the context features

CLASSES = ["Normal", "DoS", "Fuzzy", "Spoof_RPM", "Spoof_Gear"]
ATTACKS = CLASSES[1:]
COLORS = {
    "Normal": "#34D399", "DoS": "#F87171", "Fuzzy": "#FBBF24",
    "Spoof_RPM": "#A78BFA", "Spoof_Gear": "#F472B6",
}
PAYLOAD_COLS = [f"d{i}" for i in range(8)]
RAW_COLS = ["timestamp", "can_id", "dlc"] + PAYLOAD_COLS + ["label"]
