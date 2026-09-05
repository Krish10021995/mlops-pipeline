from __future__ import annotations

import argparse
import csv
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from mlops import config
from mlops.features.build import FEATURE_COLS, build_features
from mlops.models.predict import load_model

MODEL = None
MODEL_META = {"run_id": "unknown", "source": "not-loaded"}

PREDICT_REQUESTS = Counter("predict_requests_total", "Total /v1/predict calls")
PREDICT_ERRORS = Counter("predict_errors_total", "Failed /v1/predict calls")
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction latency",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
)
PREDICTION_GAUGE = Gauge("prediction_last_value", "Most recent predicted fare amount")

LIVE_LOG = Path(os.environ.get("LIVE_LOG_PATH", "data/predictions/live.csv"))


def _ensure_model() -> None:
    global MODEL, MODEL_META
    if MODEL is None:
        MODEL = load_model()
        MODEL_META = {
            "run_id": os.environ.get("MODEL_RUN_ID", "unknown"),
            "source": os.environ.get("MODEL_URI", "MODEL_PATH/registry"),
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    _ensure_model()
    yield


app = FastAPI(title="taxi-fare-api", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    pickup_datetime: datetime
    dropoff_datetime: datetime
    trip_distance: float = Field(gt=0, le=100)
    passenger_count: int = Field(ge=1, le=9)


class PredictResponse(BaseModel):
    prediction: float
    model_run_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "run_id": MODEL_META["run_id"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    PREDICT_REQUESTS.inc()
    start = perf_counter()
    try:
        rows = pd.DataFrame(
            [
                {
                    "pickup_datetime": req.pickup_datetime,
                    "dropoff_datetime": req.dropoff_datetime,
                    "trip_distance": req.trip_distance,
                    "passenger_count": req.passenger_count,
                }
            ]
        )
        features = build_features(rows)[FEATURE_COLS]
        value = float(MODEL.predict(features)[0])
        PREDICTION_GAUGE.set(value)
        _append_live_log(value)
        return PredictResponse(prediction=round(value, 2), model_run_id=MODEL_META["run_id"])
    except Exception as exc:
        PREDICT_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        PREDICT_LATENCY.observe(perf_counter() - start)


def _append_live_log(value: float) -> None:
    if LIVE_LOG.suffix != ".csv":
        return
    LIVE_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LIVE_LOG.exists()
    with LIVE_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["created_at", "prediction"])
        writer.writerow([datetime.utcnow().isoformat(), round(value, 2)])


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the taxi fare prediction API")
    default_host = os.environ.get("API_HOST") or config.get("serving", "host", default="0.0.0.0")
    default_port = int(os.environ.get("API_PORT") or config.get("serving", "port", default=8000))
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args(argv)
    uvicorn.run(app, host=args.host, port=args.port)
