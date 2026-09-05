from __future__ import annotations

import numpy as np

from mlops.serving.drift import psi


def test_psi_identical_distributions_is_zero():
    values = np.random.default_rng(1).normal(size=2000)
    assert psi(values, values) == 0.0


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(10, 2, 2000)
    shifted = base + 6
    assert psi(base, shifted) > 0.5


def test_psi_with_empty_input_is_nan():
    assert np.isnan(psi(np.array([]), np.array([1.0, 2.0])))


def test_psi_small_inputs_finite():
    assert np.isfinite(psi(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5])))
