"""Fast unit tests:  pytest -q   (API tests need artifacts from run_pipeline.py)"""
import numpy as np
import pytest

from src import tara
from src.config import ATTACKS, CLASSES
from src.detect import artifacts_ready
from src.features import FEATURES, build_features, fit_baseline
from src.simulate import generate


@pytest.fixture(scope="module")
def frames():
    return generate(20, seed=3, bursts_per_attack=2)


def test_generator_has_all_classes(frames):
    assert set(frames.label) == set(CLASSES)
    assert frames.timestamp.is_monotonic_increasing


def test_features_shape_and_no_nans(frames):
    b = fit_baseline(frames[frames.label == "Normal"])
    X = build_features(frames, b)
    assert list(X.columns) == FEATURES and len(X) == len(frames)
    assert not X.isna().any().any()


def test_dos_is_obviously_fast(frames):
    b = fit_baseline(frames[frames.label == "Normal"])
    X = build_features(frames, b).assign(label=frames.label.values)
    assert X[X.label == "DoS"].dt_ratio.median() < 0.1 * X[X.label == "Normal"].dt_ratio.median()


def test_forged_payloads_are_near_duplicates(frames):
    b = fit_baseline(frames[frames.label == "Normal"])
    X = build_features(frames, b).assign(label=frames.label.values)
    assert X[X.label == "Spoof_Gear"].payload_ham_min.median() < X[X.label == "Normal"].payload_ham_min.median()


def test_risk_matrix_is_monotonic():
    for imp, row in tara.RISK_MATRIX.items():
        assert row == sorted(row), imp
    for j in range(4):
        col = [tara.RISK_MATRIX[i][j] for i in tara.IMPACT_LEVELS]
        assert col == sorted(col)


def test_mitigations_never_increase_risk():
    for a in ATTACKS:
        full = tara.assess(a)
        assert full["residual"]["risk"] <= full["risk"]
        assert tara.assess(a, [])["residual"]["risk"] == full["risk"]


def test_unknown_attack_returns_none():
    assert tara.assess("Normal") is None


@pytest.mark.skipif(not artifacts_ready(), reason="run run_pipeline.py first")
def test_api_roundtrip(frames):
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["model_ready"] is True
    assert c.get("/tara/DoS").status_code == 200 and c.get("/tara/Nope").status_code == 404
    rows = [{"timestamp": float(r.timestamp), "can_id": int(r.can_id), "dlc": 8, "data": [int(getattr(r, f"d{i}")) for i in range(8)]} for r in frames.head(800).itertuples()]
    out = c.post("/predict", json={"frames": rows}).json()
    assert out["n_frames"] == 800 and "alerts" in out
