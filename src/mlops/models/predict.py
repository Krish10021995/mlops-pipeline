from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from mlops import config
from mlops.features.build import FEATURE_COLS


def load_model(uri: str | None = None):
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return joblib.load(env_path)["model"]

    uri = uri or os.environ.get("MODEL_URI")
    if uri:
        return mlflow.sklearn.load_model(uri)

    client = mlflow.tracking.MlflowClient()
    name = os.environ.get("MODEL_NAME") or config.get("model", "name", default="taxi-fare-model")
    alias = os.environ.get("MODEL_ALIAS") or config.get(
        "model", "registry_alias", default="champion"
    )
    model_version = client.get_model_version_by_alias(name, alias)
    if model_version is None:
        raise RuntimeError(f"No alias {alias!r} registered for model {name!r}")
    return mlflow.sklearn.load_model(model_version.source)


def predict(features: pd.DataFrame, model) -> pd.Series:
    X = features[FEATURE_COLS].reindex(columns=FEATURE_COLS)
    return pd.Series(model.predict(X), index=features.index)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run batch prediction with the served model")
    parser.add_argument(
        "--input",
        default=config.get("features", "output_path", default="data/features/features.parquet"),
    )
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input).head(args.rows)
    model = load_model()
    df["prediction"] = predict(df, model)
    columns = ["pickup_datetime", "trip_distance", "fare_amount", "prediction"]
    print(df[columns].to_string(index=False))
