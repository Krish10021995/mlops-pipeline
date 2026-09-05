from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mlops import config
from mlops.utils import ensure_dir


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10, eps: float = 1e-4) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    lower = min(float(expected.min()), float(actual.min()))
    upper = max(float(expected.max()), float(actual.max()))
    if upper - lower < eps:
        return 0.0

    edges = np.linspace(lower, upper, buckets + 1)
    exp, _ = np.histogram(expected, bins=edges)
    act, _ = np.histogram(actual, bins=edges)
    exp = exp / exp.sum()
    act = act / act.sum()
    exp = np.clip(exp, eps, None)
    act = np.clip(act, eps, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect drift between training and live data")
    parser.add_argument(
        "--reference",
        default=config.get("drift", "reference_path", default="data/features/features.parquet"),
    )
    parser.add_argument(
        "--live",
        default=config.get("drift", "live_path", default="data/predictions/live.csv"),
    )
    parser.add_argument(
        "--threshold", type=float, default=config.get("drift", "threshold", default=0.2)
    )
    parser.add_argument("--output", default="artifacts/drift_report.json")
    args = parser.parse_args(argv)

    live_path = Path(args.live)
    if not live_path.exists():
        print("No live predictions yet - nothing to compare.")
        return

    reference = pd.read_parquet(args.reference)["fare_amount"].to_numpy()
    live = pd.read_csv(live_path)["prediction"].to_numpy()

    score = psi(reference, live)
    report = {
        "metric": "predicted fare vs trained fare",
        "psi": score,
        "threshold": args.threshold,
        "drift_detected": score > args.threshold,
        "samples": int(len(live)),
    }
    ensure_dir(args.output)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(1 if report["drift_detected"] else 0)
