from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor
from starlette.testclient import TestClient

import mlops.serving.api as api
from mlops.data.generate import generate
from mlops.features.build import FEATURE_COLS, build_features, clean_features


def _tiny_model():
    df = clean_features(build_features(generate(600, seed=5)))
    model = HistGradientBoostingRegressor(max_iter=30, random_state=0)
    return model.fit(df[FEATURE_COLS], df["fare_amount"])


def _install_model(tmp_path):
    api.MODEL = _tiny_model()
    api.MODEL_META = {"run_id": "test", "source": "test"}
    api.LIVE_LOG = tmp_path / "live.csv"


def test_predict_roundtrip(tmp_path):
    _install_model(tmp_path)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/predict",
            json={
                "pickup_datetime": "2024-11-10T09:30:00",
                "dropoff_datetime": "2024-11-10T09:48:00",
                "trip_distance": 6.2,
                "passenger_count": 2,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] > 0
    assert body["model_run_id"] == "test"
    assert api.LIVE_LOG.exists()


def test_predict_rejects_invalid_payload(tmp_path):
    _install_model(tmp_path)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/predict",
            json={
                "pickup_datetime": "not-a-date",
                "dropoff_datetime": "2024-11-10T09:48:00",
                "trip_distance": -1,
                "passenger_count": 0,
            },
        )
    assert response.status_code == 422


def test_health(tmp_path):
    _install_model(tmp_path)
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metrics_endpoint(tmp_path):
    _install_model(tmp_path)
    with TestClient(api.app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "predict_requests_total" in response.text
