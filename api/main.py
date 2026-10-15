"""FastAPI service:  uvicorn api.main:app --reload   (docs at /docs)"""
from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import tara
from src.config import METRICS_PATH
from src.detect import alerts, artifacts_ready, drift_report, get_detector

app = FastAPI(title="CANGuard TARA API", version="1.0.0",
              description="CAN-bus intrusion detection with an ISO/SAE 21434-style risk assessment.")


class Frame(BaseModel):
    timestamp: float
    can_id: int = Field(ge=0, le=0x7FF)
    dlc: int = Field(8, ge=0, le=8)
    data: list[int] = Field(min_length=8, max_length=8)


class PredictRequest(BaseModel):
    frames: list[Frame] = Field(min_length=1, max_length=200_000)


def _need_model():
    if not artifacts_ready():
        raise HTTPException(503, "Model artifacts missing - run `python run_pipeline.py` first.")
    return get_detector()


def _run(df: pd.DataFrame) -> dict:
    det = _need_model()
    try:
        pred = det.predict(df)
    except ValueError as e:
        raise HTTPException(422, str(e))
    al = alerts(pred)
    for a in al:
        a["tara"] = {k: v for k, v in (tara.assess(a["class"]) or {}).items() if k in ("id", "name", "impact", "feasibility", "risk", "risk_label", "treatment")}
    return {"n_frames": len(pred), "counts": pred.pred.value_counts().to_dict(), "alerts": al,
            "drift": drift_report(pred.attrs["features"], det.baseline).to_dict("records"),
            "predictions": pred[["timestamp", "can_id", "pred", "confidence"]].round(4).to_dict("records")[:5000]}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": artifacts_ready()}


@app.get("/model/info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(503, "No metrics - run the pipeline first.")
    m = json.loads(METRICS_PATH.read_text())
    return {"meta": m["meta"], "final_scores": m["final"]["scores"], "comparison": m["comparison"]}


@app.post("/predict")
def predict(req: PredictRequest):
    rows = [{"timestamp": f.timestamp, "can_id": f.can_id, "dlc": f.dlc, **{f"d{i}": b for i, b in enumerate(f.data)}} for f in req.frames]
    return _run(pd.DataFrame(rows))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    return _run(df)


@app.get("/tara")
def tara_all():
    return tara.all_assessments()


@app.get("/tara/{attack}")
def tara_one(attack: str, mitigations: str | None = None):
    sel = mitigations.split(",") if mitigations else None
    res = tara.assess(attack, sel)
    if res is None:
        raise HTTPException(404, f"Unknown attack class '{attack}'. Valid: {list(tara.THREATS)}")
    return res
