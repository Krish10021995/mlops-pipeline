from __future__ import annotations

import pandas as pd
from mlops.data.generate import generate


def test_generate_schema():
    df = generate(500, seed=7)
    assert set(df.columns) >= {
        "pickup_datetime",
        "dropoff_datetime",
        "trip_distance",
        "passenger_count",
        "fare_amount",
    }
    assert len(df) == 500


def test_fare_positive_and_plausible():
    df = generate(500, seed=7)
    assert (df["fare_amount"] > 2.5).all()
    assert df["fare_amount"].median() < 100
    assert df["passenger_count"].between(1, 5).all()
    assert (df["trip_distance"] >= 0.2).all()


def test_dropoff_after_pickup():
    df = generate(200, seed=1)
    pickup = pd.to_datetime(df["pickup_datetime"])
    dropoff = pd.to_datetime(df["dropoff_datetime"])
    assert (dropoff > pickup).all()


def test_deterministic_with_seed():
    a = generate(300, seed=42)
    b = generate(300, seed=42)
    assert a.equals(b)
