"""Plotting utilities for the G-E-F readout duration optimization (node 14c).

``plot_fidelity_vs_duration`` shows the three-state assignment fidelity and the per-state
assignment probabilities against integration duration, with the tolerance band and the chosen
duration marked. The IQ blobs and the confusion matrix at the chosen duration are plotted by the
``calibration_utils.iq_blobs_ef`` helpers, which the node calls directly.
"""

from typing import List

import numpy as np
import xarray as xr
from matplotlib.figure import Figure
from qualibration_libs.plotting import QubitGrid, grid_iter
from quam_builder.architecture.superconducting.qubit import AnyTransmon


def plot_fidelity_vs_duration(
    ds: xr.Dataset, qubits: List[AnyTransmon], fits: xr.Dataset, fidelity_tolerance: float
) -> Figure:
    """Grid of per-qubit assignment fidelity against integration duration."""
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        fit = fits.sel(qubit=qubit["qubit"])
        duration = fit.duration.values
        for state in fit.prepared.values:
            p = fit.confusion_matrix.sel(prepared=state, measured=state)
            ax.plot(duration, 100 * p, ".-", lw=0.8, alpha=0.7, label=f"P({state}|{state})")
        ax.plot(duration, 100 * fit.assignment_fidelity, "o-", color="k", label="mean")
        best = float(fit.best_fidelity)
        if np.isfinite(best):
            ax.axhspan(100 * (best - fidelity_tolerance), 100 * best, color="grey", alpha=0.2, label="tolerance")
        if bool(fit.success):
            chosen = float(fit.optimal_duration)
            ax.axvline(chosen, color="C3", ls="--", label="chosen")
            title = f"{qubit['qubit']}  ({chosen:.0f} ns, F={100 * float(fit.optimal_fidelity):.2f}%)"
        else:
            title = f"{qubit['qubit']}  (failed)"
        ax.set_xlabel("Integration duration [ns]")
        ax.set_ylabel("Assignment probability [%]")
        ax.set_title(title, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    grid.fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    grid.fig.suptitle("G-E-F readout: assignment fidelity vs integration duration")
    grid.fig.set_size_inches(15, 9)
    grid.fig.tight_layout()
    return grid.fig
