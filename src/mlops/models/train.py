from __future__ import annotations

import argparse

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops import config
from mlops.features.build import FEATURE_COLS
from mlops.utils import default_tracking_uri, ensure_dir

MODEL_FACTORY = {
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "RandomForestRegressor": RandomForestRegressor,
}

ALLOWED_PARAMS = {"max_iter", "learning_rate", "max_depth", "random_state", "min_samples_leaf"}


def split_by_time(
    df: pd.DataFrame, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("pickup_datetime").reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = df.iloc[n - n_test :]
    val = df.iloc[n - n_test - n_val : n - n_test]
    train = df.iloc[: n - n_test - n_val]
    return train, val, test


def make_model(model_type: str, params: dict | None = None):
    model_type = model_type or config.get("model", "type", default="HistGradientBoostingRegressor")
    kwargs = {k: v for k, v in (params or {}).items() if k in ALLOWED_PARAMS}
    return MODEL_FACTORY[model_type](early_stopping=True, validation_fraction=0.1, **kwargs)


def train_pipeline(
    df: pd.DataFrame,
    *,
    model_path: str,
    params: dict | None = None,
    register: bool = False,
    model_name: str | None = None,
    experiment: str | None = None,
) -> dict:
    model_name = model_name or config.get("model", "name", default="taxi-fare-model")
    model_type = config.get("model", "type", default="HistGradientBoostingRegressor")

    train, val, test = split_by_time(df)
    X, y = train[FEATURE_COLS], train["fare_amount"]
    Xv, yv = val[FEATURE_COLS], val["fare_amount"]

    model = make_model(model_type, params)
    mlflow.set_tracking_uri(default_tracking_uri())
    experiment = experiment or config.get("tracking", "experiment_name", default="taxi-fare")
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        model.fit(X, y)
        val_pred = model.predict(Xv)
        metrics = {
            "val_rmse": float(np.sqrt(mean_squared_error(yv, val_pred))),
            "val_mae": float(mean_absolute_error(yv, val_pred)),
            "val_r2": float(r2_score(yv, val_pred)),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        model_info = mlflow.sklearn.log_model(model, name="model")

        run_id = run.info.run_id
        if register:
            registered = mlflow.register_model(model_info.model_uri, model_name)
            alias = config.get("model", "registry_alias", default="champion")
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, alias, registered.version)

    payload = {"model": model, "features": FEATURE_COLS, "run_id": run_id, "metrics": metrics}
    ensure_dir(model_path)
    joblib.dump(payload, model_path)
    metrics.update({"run_id": run_id, "model_path": str(model_path)})
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and register the taxi fare model")
    parser.add_argument(
        "--input",
        default=config.get("features", "output_path", default="data/features/features.parquet"),
    )
    parser.add_argument(
        "--model-path",
        default=config.get("model", "artifact_path", default="artifacts/taxi-fare-model.joblib"),
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args(argv)

    params = dict(config.get("model", "params", default={}))
    if args.max_iter is not None:
        params["max_iter"] = args.max_iter

    df = pd.read_parquet(args.input)
    metrics = train_pipeline(df, model_path=args.model_path, params=params, register=args.register)
    print("Training complete:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
