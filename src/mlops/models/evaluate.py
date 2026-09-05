from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlops import config
from mlops.features.build import FEATURE_COLS
from mlops.models.train import split_by_time
from mlops.utils import ensure_dir


def evaluate_model(payload: dict, df: pd.DataFrame) -> dict:
    _, _, test = split_by_time(df)
    X_test, y_test = test[FEATURE_COLS], test["fare_amount"]

    pred = payload["model"].predict(X_test)

    train_median = float(
        df.sort_values("pickup_datetime").iloc[: int(len(df) * 0.7)]["fare_amount"].median()
    )
    baseline_pred = np.full_like(pred, train_median)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))

    return {
        "test_rows": len(test),
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "baseline_rmse": baseline_rmse,
        "improvement_vs_baseline": 1.0 - rmse / baseline_rmse if baseline_rmse > 0 else None,
        "run_id": payload.get("run_id", "unknown"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained model against a baseline")
    parser.add_argument(
        "--model-path",
        default=config.get("model", "artifact_path", default="artifacts/taxi-fare-model.joblib"),
    )
    parser.add_argument(
        "--input",
        default=config.get("features", "output_path", default="data/features/features.parquet"),
    )
    parser.add_argument(
        "--output", default="artifacts/eval_metrics.json"
    )
    parser.add_argument("--gate", action="store_true", help="Fail if model does not beat baseline")
    args = parser.parse_args(argv)

    payload = joblib.load(args.model_path)
    df = pd.read_parquet(args.input)
    metrics = evaluate_model(payload, df)

    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))

    if args.gate and metrics.get("improvement_vs_baseline") is not None:
        if metrics["improvement_vs_baseline"] <= 0:
            print("GATE: model does NOT beat the baseline - blocking promotion.")
            raise SystemExit(1)
        print("GATE: model beats baseline - promotion allowed.")
