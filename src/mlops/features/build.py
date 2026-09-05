from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mlops import config
from mlops.utils import ensure_dir

FEATURE_COLS = [
    "trip_distance",
    "passenger_count",
    "hour",
    "dow",
    "is_weekend",
    "is_rush",
    "duration_min",
    "speed_mph",
    "log_distance",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    pickup = pd.to_datetime(df["pickup_datetime"])
    dropoff = pd.to_datetime(df["dropoff_datetime"])
    df["hour"] = pickup.dt.hour.astype(int)
    df["dow"] = pickup.dt.dayofweek.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_rush"] = (
        ((df["hour"] >= 7) & (df["hour"] < 10)) | ((df["hour"] >= 16) & (df["hour"] < 20))
    ).astype(int)
    df["duration_min"] = ((dropoff - pickup).dt.total_seconds() / 60.0).clip(lower=1)
    df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60.0)
    df["log_distance"] = np.log1p(df["trip_distance"])
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=FEATURE_COLS + ["fare_amount"])
    df = df[df["duration_min"] >= 1]
    df = df[df["speed_mph"].between(0, 80)]
    df = df.drop_duplicates(subset=["pickup_datetime"])
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build features from raw trips")
    parser.add_argument(
        "--input",
        default=config.get("data", "raw_path", default="data/raw/trips.parquet"),
    )
    parser.add_argument(
        "--output",
        default=config.get("features", "output_path", default="data/features/features.parquet"),
    )
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.input)
    df = clean_features(build_features(raw))
    ensure_dir(args.output)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df):,} clean feature rows to {args.output}")
