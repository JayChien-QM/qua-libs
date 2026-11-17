"""Analysis helpers for the G-E-F readout duration optimization (node 14c).

Every shot is acquired with accumulated demodulation, so the dataset holds the |g>, |e> and |f>
IQ shots at every integration duration up to the sweep maximum. At each duration the three blobs
are classified exactly as node 15 does it: the centers come from ``iq_blobs_ef``'s Gaussian fit
and every shot goes to the center with the smallest ``|dI| + |dQ|``, which is the rule
``readout_state_gef`` applies on the OPX. The three-state assignment fidelity is the mean of the
diagonal of the resulting confusion matrix.

The chosen duration is the shortest one whose fidelity is within ``fidelity_tolerance`` of the
best on the axis: past the saturation knee a longer readout costs time in every downstream
experiment for no fidelity gain, while T1 decay during the readout eventually lowers it.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from calibration_utils.iq_blobs_ef import fit_raw_data as fit_gef_blobs
from calibration_utils.readout_duration_power_optimization.analysis import to_volts_per_duration

IQ_KEYS = ["Ig", "Qg", "Ie", "Qe", "If", "Qf"]
STATES = ["g", "e", "f"]


@dataclass
class FitParameters:
    """Fitted G-E-F readout duration for a single qubit."""

    optimal_duration: float
    optimal_fidelity: float
    best_duration: float
    best_fidelity: float
    confusion_matrix: list
    success: bool
    note: str = ""


@dataclass
class DurationChoice:
    """The duration a qubit's G-E-F readout should run at."""

    duration: float
    fidelity: float
    best_duration: float
    best_fidelity: float
    success: bool
    note: str


def process_raw_dataset(ds: xr.Dataset, node: Optional[QualibrationNode] = None) -> xr.Dataset:
    """Convert the accumulated demodulation results to Volts, one divisor per duration slice."""
    del node
    return to_volts_per_duration(ds, IQ_KEYS)


def select_duration(
    fidelity: xr.DataArray, fidelity_tolerance: float, min_assignment_fidelity: float
) -> DurationChoice:
    """Pick the shortest duration whose fidelity is within ``fidelity_tolerance`` of the best.

    Args:
        fidelity: Three-state assignment fidelity along ``duration`` (NaN where the fit failed).
        fidelity_tolerance: Absolute fidelity the readout may give up to be shorter.
        min_assignment_fidelity: Below this best fidelity the qubit fails.
    """
    durations = np.asarray(fidelity.duration.values, dtype=float)
    values = np.asarray(fidelity.values, dtype=float)
    nan_choice = dict(duration=float("nan"), fidelity=float("nan"))
    if not np.isfinite(values).any():
        return DurationChoice(**nan_choice, best_duration=float("nan"), best_fidelity=float("nan"),
                              success=False, note="the g/e/f blobs could not be fitted at any duration")

    best_index = int(np.nanargmax(values))
    best = DurationChoice(
        duration=float("nan"),
        fidelity=float("nan"),
        best_duration=float(durations[best_index]),
        best_fidelity=float(values[best_index]),
        success=False,
        note="",
    )
    if best.best_fidelity < min_assignment_fidelity:
        best.note = (
            f"the best three-state assignment fidelity {best.best_fidelity:.3f} is below "
            f"min_assignment_fidelity={min_assignment_fidelity}"
        )
        return best

    # Durations are ascending, so the first one inside the tolerance band is the shortest.
    within = np.flatnonzero(np.nan_to_num(values, nan=-np.inf) >= best.best_fidelity - fidelity_tolerance)
    index = int(within[0])
    best.duration = float(durations[index])
    best.fidelity = float(values[index])
    best.success = True
    if index == len(durations) - 1:
        best.note = "the fidelity is still rising at max_duration_in_ns; consider a longer sweep"
    return best


def _fit_slice(ds: xr.Dataset) -> Optional[xr.Dataset]:
    """Node 15's g/e/f classification on a single (qubit, duration) slice, or None on failure."""
    try:
        ds_fit, _ = fit_gef_blobs(ds, None)
    except (RuntimeError, ValueError) as error:  # curve_fit failing on unresolved blobs
        logging.getLogger(__name__).debug("g/e/f fit failed: %s", error)
        return None
    return ds_fit


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, xr.Dataset, Dict[str, FitParameters]]:
    """Classify the g/e/f shots at every duration and pick each qubit's readout duration.

    Returns:
        ``(ds_fit, ds_iq_blobs, fit_results)``. ``ds_fit`` carries the fidelity and confusion
        matrix along the duration axis plus the chosen duration; ``ds_iq_blobs`` is node 15's
        fit at the chosen duration, used for the IQ-blob and confusion-matrix figures.
    """
    qubit_names = [str(q) for q in ds.qubit.values]
    durations = ds.duration.values
    fidelity = np.full((len(qubit_names), len(durations)), np.nan)
    confusion = np.full((len(qubit_names), len(durations), 3, 3), np.nan)
    for i, q in enumerate(qubit_names):
        for j, duration in enumerate(durations):
            ds_slice = ds[IQ_KEYS].sel(qubit=[q], duration=duration).drop_vars("duration")
            fit = _fit_slice(ds_slice)
            if fit is None:
                continue
            matrix = fit.confusion_matrix.sel(qubit=q).values
            confusion[i, j] = matrix
            fidelity[i, j] = float(np.mean(np.diag(matrix)))

    ds_fit = ds.assign(
        assignment_fidelity=(("qubit", "duration"), fidelity),
        confusion_matrix=(("qubit", "duration", "prepared", "measured"), confusion),
    ).assign_coords(prepared=STATES, measured=STATES)
    ds_fit.assignment_fidelity.attrs = {"long_name": "three-state assignment fidelity"}

    choices = {
        q: select_duration(
            ds_fit.assignment_fidelity.sel(qubit=q),
            node.parameters.fidelity_tolerance,
            node.parameters.min_assignment_fidelity,
        )
        for q in qubit_names
    }
    for name in ("duration", "fidelity", "best_duration", "best_fidelity"):
        ds_fit[f"optimal_{name}" if name in ("duration", "fidelity") else name] = xr.DataArray(
            [getattr(choices[q], name) for q in qubit_names], coords={"qubit": ds_fit.qubit.data}
        )
    ds_fit["success"] = xr.DataArray([choices[q].success for q in qubit_names], coords={"qubit": ds_fit.qubit.data})

    # IQ blobs at the chosen duration. A failed qubit still occupies its slot, at the longest
    # duration, and is discarded by its success flag.
    blob_slices = []
    for q in qubit_names:
        duration = choices[q].duration if choices[q].success else durations[-1]
        blob_slices.append(ds[IQ_KEYS].sel(qubit=[q], duration=duration).drop_vars("duration"))
    ds_iq_blobs = _fit_slice(xr.concat(blob_slices, dim="qubit"))

    fit_results = {}
    for i, q in enumerate(qubit_names):
        choice = choices[q]
        j = int(np.flatnonzero(durations == choice.duration)[0]) if choice.success else -1
        fit_results[q] = FitParameters(
            optimal_duration=choice.duration,
            optimal_fidelity=choice.fidelity,
            best_duration=choice.best_duration,
            best_fidelity=choice.best_fidelity,
            confusion_matrix=confusion[i, j].tolist(),
            success=choice.success and ds_iq_blobs is not None,
            note=choice.note,
        )
    return ds_fit, ds_iq_blobs, fit_results


def log_fitted_results(fit_results: Dict, log_callable=None):
    """Log the chosen G-E-F readout duration and its fidelity for every qubit."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, result in fit_results.items():
        if result["success"]:
            s = (
                f"Results for qubit {q}:  SUCCESS!\n"
                f"\tG-E-F readout duration: {result['optimal_duration']:.0f} ns "
                f"(F = {100 * result['optimal_fidelity']:.2f} %)\n"
                f"\tBest on the axis: {result['best_duration']:.0f} ns "
                f"(F = {100 * result['best_fidelity']:.2f} %)\n"
            )
        else:
            s = f"Results for qubit {q}:  FAIL!\n"
        if result["note"]:
            s += f"\t{result['note']}\n"
        log_callable(s)
