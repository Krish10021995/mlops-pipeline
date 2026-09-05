from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mlops import config
from mlops.utils import ensure_dir


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-11-01 00:00:00")
    pickup = pd.Series(start + pd.to_timedelta(rng.uniform(0, 30 * 24, size=n_rows), unit="h"))
    distance_mi = rng.lognormal(mean=np.log(2.6), sigma=0.7, size=n_rows).clip(0.2, 60)
    passengers = rng.integers(1, 6, size=n_rows)
    speed = rng.lognormal(mean=np.log(12.0), sigma=0.35, size=n_rows).clip(4, 35)
    duration_min = (distance_mi / speed * 60.0 + rng.normal(0, 3, n_rows)).clip(min=1)

    hour = pickup.dt.hour.to_numpy()
    dow = pickup.dt.dayofweek.to_numpy()

    fare = 3.0 + 2.5 * distance_mi + 0.35 * duration_min
    fare = fare * rng.uniform(1.0, 1.5, n_rows)
    fare = fare + np.where((hour >= 16) & (hour < 20), 1.5, 0.0)
    fare = fare + np.where(dow >= 5, 1.0, 0.0)
    fare = fare.clip(min=2.5) + rng.normal(0, 1.2, n_rows)

    df = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(pickup),
            "dropoff_datetime": pd.to_datetime(pickup) + pd.to_timedelta(duration_min, unit="m"),
            "trip_distance": distance_mi.round(3),
            "passenger_count": passengers,
            "rate_code": rng.integers(1, 4, n_rows),
            "vendor_id": rng.integers(1, 3, n_rows),
            "fare_amount": fare.round(2),
        }
    )
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NYC taxi trip data")
    parser.add_argument(
        "--rows",
        type=int,
        default=config.get("data", "n_rows", default=50_000),
    )
    parser.add_argument("--seed", type=int, default=config.get("data", "seed", default=42))
    parser.add_argument(
        "--output",
        default=config.get("data", "raw_path", default="data/raw/trips.parquet"),
    )
    args = parser.parse_args(argv)

    df = generate(args.rows, args.seed)
    ensure_dir(args.output)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")
