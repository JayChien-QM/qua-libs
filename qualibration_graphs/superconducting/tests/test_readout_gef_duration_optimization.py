"""Tests for the G-E-F readout duration optimization (node 14c).

* ``select_duration`` -- the shortest duration inside the tolerance band of the best fidelity
  wins, a qubit whose best fidelity is below the floor fails, and a fidelity still rising at the
  end of the axis is flagged.
* ``fit_raw_data`` -- on synthetic g/e/f shots whose blobs separate as the integration grows,
  the classification reproduces the expected fidelity trend and picks the saturation knee.
* the parameter validator -- the 4 ns chunk grid that ``demod.accumulated`` requires.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.readout_gef_duration_optimization.analysis import (  # noqa: E402
    fit_raw_data,
    select_duration,
)
from calibration_utils.readout_gef_duration_optimization.parameters import NodeSpecificParameters  # noqa: E402

DURATIONS = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])


def _fidelity(values) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=float), dims=["duration"], coords={"duration": DURATIONS})


# ------------------------------------------------------------------- select_duration


def test_select_duration_takes_the_shortest_point_inside_the_tolerance_band():
    choice = select_duration(_fidelity([0.6, 0.90, 0.948, 0.95, 0.94]), 0.005, 0.5)
    assert choice.success
    assert choice.duration == 600.0
    assert choice.best_duration == 800.0
    assert choice.best_fidelity == pytest.approx(0.95)


def test_select_duration_with_zero_tolerance_is_the_argmax():
    choice = select_duration(_fidelity([0.6, 0.90, 0.948, 0.95, 0.94]), 0.0, 0.5)
    assert choice.duration == 800.0


def test_select_duration_ignores_failed_slices():
    choice = select_duration(_fidelity([np.nan, 0.90, np.nan, 0.95, 0.94]), 0.02, 0.5)
    assert choice.success
    assert choice.duration == 800.0


def test_select_duration_fails_below_the_fidelity_floor():
    choice = select_duration(_fidelity([0.35, 0.40, 0.42, 0.45, 0.44]), 0.005, 0.5)
    assert not choice.success
    assert "min_assignment_fidelity" in choice.note


def test_select_duration_fails_when_nothing_was_fitted():
    choice = select_duration(_fidelity([np.nan] * 5), 0.005, 0.5)
    assert not choice.success


def test_select_duration_flags_a_fidelity_still_rising_at_the_end():
    choice = select_duration(_fidelity([0.6, 0.7, 0.8, 0.9, 0.95]), 0.005, 0.5)
    assert choice.success
    assert choice.duration == 1000.0
    assert "still rising" in choice.note


# ------------------------------------------------------------------- fit_raw_data


def _synthetic_dataset(n_runs: int = 1500, seed: int = 0) -> xr.Dataset:
    """g/e/f blobs on a triangle whose noise shrinks as 1/sqrt(duration), in Volts."""
    rng = np.random.default_rng(seed)
    centers = {"g": (0.0, 0.0), "e": (1.0, 0.0), "f": (0.5, 0.9)}
    data = {}
    sigma = 0.6 * np.sqrt(DURATIONS[0] / DURATIONS)  # shape (duration,)
    for state, (i0, q0) in centers.items():
        data[f"I{state}"] = i0 + rng.normal(size=(1, n_runs, len(DURATIONS))) * sigma
        data[f"Q{state}"] = q0 + rng.normal(size=(1, n_runs, len(DURATIONS))) * sigma
    coords = {"qubit": ["q1"], "n_runs": np.arange(1, n_runs + 1), "duration": DURATIONS}
    return xr.Dataset({k: (("qubit", "n_runs", "duration"), v) for k, v in data.items()}, coords=coords)


def test_fit_raw_data_tracks_the_fidelity_and_picks_the_knee():
    node = SimpleNamespace(parameters=SimpleNamespace(fidelity_tolerance=0.01, min_assignment_fidelity=0.5))
    ds_fit, ds_iq_blobs, results = fit_raw_data(_synthetic_dataset(), node)

    fidelity = ds_fit.assignment_fidelity.sel(qubit="q1").values
    assert np.all(np.isfinite(fidelity))
    # The blobs separate as the integration grows.
    assert fidelity[-1] > fidelity[0] + 0.1
    assert fidelity[-1] > 0.9

    result = results["q1"]
    assert result.success
    assert result.optimal_fidelity >= result.best_fidelity - 0.01
    assert result.optimal_duration <= result.best_duration
    assert np.allclose(np.sum(result.confusion_matrix, axis=1), 1.0)
    assert ds_iq_blobs is not None
    assert "confusion_matrix" in ds_iq_blobs


# ------------------------------------------------------------------- parameters


def test_parameters_reject_a_chunk_off_the_4ns_grid():
    with pytest.raises(ValueError, match="multiple of 4"):
        NodeSpecificParameters(max_duration_in_ns=3000, num_durations=7)


def test_parameters_accept_the_default_grid():
    params = NodeSpecificParameters()
    assert params.max_duration_in_ns / params.num_durations % 4 == 0
