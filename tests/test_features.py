from __future__ import annotations

import pandas as pd
from mlops.data.generate import generate

from mlops.features.build import FEATURE_COLS, build_features, clean_features


def test_build_features_adds_expected_columns():
    raw = generate(300, seed=3)
    features = clean_features(build_features(raw))
    assert set(FEATURE_COLS) <= set(features.columns)
    assert not features[FEATURE_COLS].isna().any(axis=None)


def test_time_features_are_sane():
    raw = generate(300, seed=3)
    features = build_features(raw)
    assert features["hour"].between(0, 23).all()
    assert features["dow"].between(0, 6).all()
    assert features["is_weekend"].isin([0, 1]).all()
    assert features["is_rush"].isin([0, 1]).all()


def test_clean_features_removes_duplicates():
    raw = generate(300, seed=3)
    part = clean_features(build_features(raw))
    duplicated = pd.concat([part, part], ignore_index=True)
    cleaned = clean_features(duplicated)
    assert len(cleaned) == len(duplicated) / 2
