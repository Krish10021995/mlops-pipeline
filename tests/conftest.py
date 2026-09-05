from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_mlflow():
    tmp = tempfile.mkdtemp(prefix="mlflow-tests-")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp}/mlruns.db"
    yield
