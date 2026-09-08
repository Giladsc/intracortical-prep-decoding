"""Turning raw trial arrays into labels and a clean trial set.

Each function here is *first derived inline* in the notebook noted in its docstring,
so nothing appears as a black box before it has been explained. The notebooks import
these afterwards to avoid re-typing them.

Nothing in this module standardises (z-scores) the neural features. Standardisation
belongs strictly inside the cross-validation loop, and it is written out there in the
notebooks so that the leakage discipline is visible where it is being taught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Direction labels  (derived inline in notebook 01)
# ---------------------------------------------------------------------------
def direction_angles(target: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Movement direction per trial, in radians on [0, 2*pi).

    theta = atan2(y_target - y_start, x_target - x_start)
    """
    delta = target - start
    return np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * np.pi)


def direction_labels(target: np.ndarray, start: np.ndarray, n_directions: int = 8,
                     tolerance_deg: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Integer direction class (0..n-1) and the class-centre angles in radians.

    Validates that the observed angles really do form `n_directions` evenly spaced
    clusters. Raises rather than silently rounding arbitrary angles into bins - if
    the task is not a clean radial-n task we want to know, not to get plausible
    looking labels.
    """
    theta = direction_angles(target, start)
    finite = np.isfinite(theta)
    if not finite.any():
        raise ValueError("No finite movement directions could be derived from "
                         "target/start positions.")

    spacing = 2 * np.pi / n_directions
    labels = np.full(theta.shape, -1, dtype=int)
    labels[finite] = np.rint(theta[finite] / spacing).astype(int) % n_directions

    # How far is each trial from the ideal grid angle it was assigned to?
    ideal = labels[finite] * spacing
    residual = np.abs(np.angle(np.exp(1j * (theta[finite] - ideal))))
    worst_deg = float(np.degrees(residual.max()))
    if worst_deg > tolerance_deg:
        raise ValueError(
            f"Movement directions do not form {n_directions} evenly spaced clusters: "
            f"the worst trial sits {worst_deg:.2f} deg from its nearest grid angle "
            f"(tolerance {tolerance_deg} deg). Check target_data_all / start_data_all."
        )

    observed = np.unique(labels[finite])
    if observed.size != n_directions:
        raise ValueError(
            f"Expected {n_directions} distinct directions but found {observed.size} "
            f"(classes present: {observed.tolist()})."
        )

    centres = np.arange(n_directions) * spacing
    return labels, centres


# ---------------------------------------------------------------------------
# Trial selection  (derived inline in notebook 03)
# ---------------------------------------------------------------------------
@dataclass
class SelectionReport:
    """Why each trial was dropped. Printed by every notebook before decoding."""
    n_total: int
    n_kept: int
    dropped_nonfinite_features: int
    dropped_nonfinite_delay: int
    dropped_short_delay: int
    dropped_no_go: int
    min_delay_s: float
    go_trials_only: bool

    def __str__(self) -> str:
        go_note = "" if self.go_trials_only else "  (go-only filter disabled)"
        return (
            "Trial selection\n"
            f"  starting trials              : {self.n_total}\n"
            f"  dropped, non-finite features : {self.dropped_nonfinite_features}\n"
            f"  dropped, non-finite delay    : {self.dropped_nonfinite_delay}\n"
            f"  dropped, delay < {self.min_delay_s:g} s        : {self.dropped_short_delay}\n"
            f"  dropped, no-go trials        : {self.dropped_no_go}{go_note}\n"
            f"  --> kept                     : {self.n_kept}"
        )


def select_trials(features: np.ndarray, delay: np.ndarray, no_go: np.ndarray,
                  min_delay_s: float, go_trials_only: bool
                  ) -> tuple[np.ndarray, SelectionReport]:
    """Boolean keep-mask over trials, plus a report of what was removed and why.

    The counts are reported as *sequential* exclusions (each stage counts only
    trials still alive at that point), so they sum to the number dropped.
    """
    n_total = features.shape[0]
    alive = np.ones(n_total, dtype=bool)

    bad_features = ~np.isfinite(features).all(axis=tuple(range(1, features.ndim)))
    n_bad_features = int((bad_features & alive).sum())
    alive &= ~bad_features

    bad_delay = ~np.isfinite(delay)
    n_bad_delay = int((bad_delay & alive).sum())
    alive &= ~bad_delay

    short_delay = np.zeros(n_total, dtype=bool)
    finite_delay = np.isfinite(delay)
    short_delay[finite_delay] = delay[finite_delay] < min_delay_s
    n_short = int((short_delay & alive).sum())
    alive &= ~short_delay

    n_no_go = 0
    if go_trials_only:
        # A non-finite no_go flag means we cannot confirm the trial was a go trial,
        # so it is excluded too.
        is_no_go = ~(no_go == 0)
        n_no_go = int((is_no_go & alive).sum())
        alive &= ~is_no_go

    report = SelectionReport(
        n_total=n_total,
        n_kept=int(alive.sum()),
        dropped_nonfinite_features=n_bad_features,
        dropped_nonfinite_delay=n_bad_delay,
        dropped_short_delay=n_short,
        dropped_no_go=n_no_go,
        min_delay_s=min_delay_s,
        go_trials_only=go_trials_only,
    )
    if report.n_kept == 0:
        raise ValueError("Trial selection removed every trial.\n" + str(report))
    return alive, report


def check_class_balance(labels: np.ndarray, n_directions: int,
                        min_per_direction: int) -> np.ndarray:
    """Trial count per direction; raises if any direction is too sparse to decode."""
    counts = np.bincount(labels, minlength=n_directions)
    thin = np.where(counts < min_per_direction)[0]
    if thin.size:
        raise ValueError(
            f"Direction(s) {thin.tolist()} have fewer than {min_per_direction} trials "
            f"(counts: {counts.tolist()}). Decoding would not be meaningful."
        )
    return counts


# ---------------------------------------------------------------------------
# Feature handling  (derived inline in notebook 02)
# ---------------------------------------------------------------------------
def to_hz(spike_counts: np.ndarray, bin_s: float) -> np.ndarray:
    """Smoothed spike counts per bin -> an equivalent firing rate in Hz."""
    return spike_counts / bin_s


def pick_feature(trial_data, feature_type: str) -> np.ndarray:
    """The (trials, bins, features) array named by `feature_type`.

    "spikes"      threshold crossings only (the prototype default: one feature per
                  channel, and each one is interpretable as a firing rate)
    "sbp"         spike-band power only
    "spikes+sbp"  both, concatenated along the feature axis
    """
    if feature_type == "spikes":
        return trial_data.spikes
    if feature_type == "sbp":
        return trial_data.sbp
    if feature_type == "spikes+sbp":
        return np.concatenate([trial_data.spikes, trial_data.sbp], axis=2)
    raise ValueError(f"Unknown FEATURE_TYPE {feature_type!r}; expected "
                     f"'spikes', 'sbp' or 'spikes+sbp'.")


def analysis_time_index(t_ms: np.ndarray, step_ms: float | None) -> np.ndarray:
    """Indices of the bins to analyse. `None` keeps every provided bin.

    Sub-sampling only thins the analysis grid; it never modifies or re-bins the data.
    """
    if step_ms is None:
        return np.arange(t_ms.size)
    native_step = float(np.round(np.diff(t_ms)[0], 6))
    if step_ms < native_step:
        raise ValueError(f"Requested analysis step {step_ms} ms is finer than the "
                         f"data's own {native_step} ms bins.")
    stride = int(round(step_ms / native_step))
    return np.arange(0, t_ms.size, stride)
