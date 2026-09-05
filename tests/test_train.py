from __future__ import annotations

import joblib

from mlops.data.generate import generate
from mlops.features.build import FEATURE_COLS, build_features, clean_features
from mlops.models.train import split_by_time, train_pipeline


def _small_dataset(n: int = 400):
    return clean_features(build_features(generate(n, seed=11)))


def test_split_by_time_is_chronological():
    df = _small_dataset()
    train, val, test = split_by_time(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["pickup_datetime"].max() <= val["pickup_datetime"].min()
    assert val["pickup_datetime"].max() <= test["pickup_datetime"].min()


def test_train_pipeline(tmp_path):
    df = _small_dataset()
    out = tmp_path / "model.joblib"
    metrics = train_pipeline(
        df,
        model_path=str(out),
        params={"max_iter": 30, "learning_rate": 0.1},
        register=False,
        experiment="tests",
    )
    assert {"val_rmse", "val_mae", "val_r2"} <= set(metrics)
    assert metrics["val_rmse"] < 20

    payload = joblib.load(out)
    assert payload["model"] is not None
    predictions = payload["model"].predict(df[FEATURE_COLS].head(3))
    assert predictions.shape == (3,)
