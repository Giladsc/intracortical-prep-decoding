"""Loading the Dryad `fig1-4` HDF5 files into a plain Python object.

This is plumbing, not science: it opens the file, checks that the arrays are shaped
the way the dataset README promises, and hands back a `TrialData` container. The
interesting derivations (direction labels, trial selection) live in
`src/preprocessing.py` and are worked through inline in the notebooks.

Dataset: Rigotti-Thompson et al. (2026), Dryad doi:10.5061/dryad.brv15dvr9
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# Fields the fig1-4 files are documented to contain. Optional ones are absent for
# some participants (`emg_data_all` is T16 only) or some figures.
_REQUIRED_FIELDS = (
    "spike_data_all",
    "sbp_data_all",
    "target_data_all",
    "start_data_all",
    "delay_data_all",
    "no_go_bool_all",
    "block_id_data_all",
    "keep_chans_all",
    "t_data",
)
_OPTIONAL_FIELDS = (
    "cursor_data_all",
    "trial_duration_data_all",
    "emg_data_all",
)


@dataclass
class TrialData:
    """One trialised alignment epoch for one participant.

    Array conventions (see the Dryad README):
        spikes  (trials, bins, chans)  smoothed threshold-crossing counts per bin
        sbp     (trials, bins, chans)  smoothed spike-band power, arbitrary units
        target  (trials, 2)            target position [X, Y] in px
        start   (trials, 2)            start position [X, Y] in px
        delay   (trials,)              instructed delay duration in s
        no_go   (trials,)              0 = go trial, 1 = no-go (catch) trial
        block   (trials,)              recording block id
        t_ms    (bins,)                time relative to the alignment event, in ms
    """

    participant: str
    alignment: str
    source_path: Path

    spikes: np.ndarray
    sbp: np.ndarray
    target: np.ndarray
    start: np.ndarray
    delay: np.ndarray
    no_go: np.ndarray
    block: np.ndarray
    keep_chans: np.ndarray
    t_ms: np.ndarray
    bin_s: float

    optional: dict[str, np.ndarray] = field(default_factory=dict)

    # -- convenient shapes -------------------------------------------------
    @property
    def n_trials(self) -> int:
        return self.spikes.shape[0]

    @property
    def n_bins(self) -> int:
        return self.spikes.shape[1]

    @property
    def n_channels(self) -> int:
        return self.spikes.shape[2]

    @property
    def cursor(self) -> np.ndarray | None:
        return self.optional.get("cursor_data_all")

    def __repr__(self) -> str:  # keeps notebook output readable
        return (
            f"TrialData({self.participant}/{self.alignment}: "
            f"{self.n_trials} trials x {self.n_bins} bins x {self.n_channels} chans, "
            f"bin={self.bin_s * 1000:.0f} ms, "
            f"t={self.t_ms[0]:.0f}..{self.t_ms[-1]:.0f} ms)"
        )


def find_data_file(participant: str, alignment: str, figure: str,
                   candidate_dirs) -> Path | None:
    """Find `{figure}_{participant}_{alignment}.h5`, or None if it is not there.

    The non-raising half of `resolve_data_file`. Use this to ask whether a file
    exists - not every participant has every figure (only T11 and T16 have a
    fig5 continuous block) - and let the caller decide what to do about it.
    """
    filename = f"{figure}_{participant}_{alignment}.h5"
    for directory in candidate_dirs:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_data_file(participant: str, alignment: str, figure: str,
                      candidate_dirs) -> Path:
    """Find `{figure}_{participant}_{alignment}.h5` in the candidate directories."""
    candidate_dirs = list(candidate_dirs)
    path = find_data_file(participant, alignment, figure, candidate_dirs)
    if path is not None:
        return path
    filename = f"{figure}_{participant}_{alignment}.h5"
    searched = [str(Path(d)) for d in candidate_dirs]
    raise FileNotFoundError(
        f"Could not find {filename}. Searched:\n  "
        + "\n  ".join(searched)
        + "\nDownload it from doi:10.5061/dryad.brv15dvr9 and place it in the data/ "
          "directory, or set INTRACORTICAL_DATA_DIR."
    )


def _derive_bin_size(t_ms: np.ndarray, path: Path, strict: bool = True) -> float:
    """Bin width in seconds, derived rather than assumed to be 20 ms.

    `strict` (the trialised files, which the analysis runs on) requires a single
    uniform step: a varying bin width would silently corrupt every rate and every
    time axis downstream. The continuous recordings are only ever looked at, and
    some of them carry a clock that jitters by a millisecond, so they pass
    `strict=False` and take the median step - reported, not hidden.
    """
    if t_ms.ndim != 1 or t_ms.size < 2:
        raise ValueError(f"{path.name}: t_data must be a 1-D array of >= 2 bins, "
                         f"got shape {t_ms.shape}.")
    steps = np.diff(t_ms)
    unique_steps = np.unique(np.round(steps, 6))
    if unique_steps.size == 1:
        return float(unique_steps[0]) / 1000.0
    if strict:
        raise ValueError(
            f"{path.name}: t_data is not uniformly spaced; found bin widths "
            f"{unique_steps} ms. The analysis assumes a constant bin width."
        )
    median_ms = float(np.median(steps))
    warnings.warn(
        f"{path.name}: clock steps vary ({unique_steps.min():.0f}-"
        f"{unique_steps.max():.0f} ms); using the median, {median_ms:.0f} ms. "
        f"This stream is for viewing only, and its time axis is plotted from the "
        f"recorded clock, so the jitter is visible rather than smoothed over.",
        stacklevel=2,
    )
    return median_ms / 1000.0


def load_trial_data(participant: str, alignment: str, figure: str,
                    candidate_dirs) -> TrialData:
    """Load one participant/alignment file and validate its basic structure."""
    path = resolve_data_file(participant, alignment, figure, candidate_dirs)

    with h5py.File(path, "r") as f:
        present = set(f.keys())
        missing = [k for k in _REQUIRED_FIELDS if k not in present]
        if missing:
            raise ValueError(
                f"{path.name}: missing expected field(s) {missing}. "
                f"File contains: {sorted(present)}"
            )
        arrays = {k: np.asarray(f[k][()]) for k in _REQUIRED_FIELDS}
        optional = {k: np.asarray(f[k][()]) for k in _OPTIONAL_FIELDS if k in present}

    t_ms = arrays["t_data"].ravel().astype(float)
    bin_s = _derive_bin_size(t_ms, path)

    spikes = arrays["spike_data_all"]
    if spikes.ndim != 3:
        raise ValueError(f"{path.name}: expected spike_data_all to be "
                         f"(trials, bins, chans), got shape {spikes.shape}.")

    n_trials, n_bins, n_chans = spikes.shape
    if n_bins != t_ms.size:
        raise ValueError(f"{path.name}: spike_data_all has {n_bins} bins but t_data "
                         f"has {t_ms.size} entries.")

    # Every per-trial field must agree on the trial count.
    for name in ("sbp_data_all", "target_data_all", "start_data_all",
                 "delay_data_all", "no_go_bool_all", "block_id_data_all"):
        if arrays[name].shape[0] != n_trials:
            raise ValueError(
                f"{path.name}: {name} has {arrays[name].shape[0]} trials but "
                f"spike_data_all has {n_trials}."
            )
    if arrays["keep_chans_all"].size != n_chans:
        raise ValueError(
            f"{path.name}: keep_chans_all has {arrays['keep_chans_all'].size} entries "
            f"but spike_data_all has {n_chans} channels."
        )

    return TrialData(
        participant=participant,
        alignment=alignment,
        source_path=path,
        spikes=spikes.astype(float),
        sbp=arrays["sbp_data_all"].astype(float),
        target=arrays["target_data_all"].astype(float),
        start=arrays["start_data_all"].astype(float),
        delay=arrays["delay_data_all"].astype(float).ravel(),
        no_go=arrays["no_go_bool_all"].astype(float).ravel(),
        block=arrays["block_id_data_all"].astype(float).ravel(),
        keep_chans=arrays["keep_chans_all"].ravel(),
        t_ms=t_ms,
        bin_s=bin_s,
        optional=optional,
    )


def describe_h5(path: Path) -> list[dict]:
    """Every dataset in the file with its shape and dtype (used by notebook 01)."""
    rows: list[dict] = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rows.append({"key": name, "shape": str(obj.shape),
                             "dtype": str(obj.dtype)})
        f.visititems(visit)
    return rows


# ---------------------------------------------------------------------------
# Continuous (non-trialised) recordings - the `fig5_{P}_continuous.h5` files
# ---------------------------------------------------------------------------
# These are the only genuinely continuous streams in the dataset, which is why
# notebook 00 uses one for its raw-signal browser. They come from a DIFFERENT
# task (closed-loop preparatory BCI control) and are preprocessed differently
# (causal exponential smoothing, z-scored, no >2 Hz channel cut), so they are
# for looking at, never for the decoding results.

_CONT_DATA_FIELDS = ("clock_time", "neural_data_norm", "cursor_data",
                     "dec_move_state", "control_samples", "emg_data")
_CONT_TRIAL_FIELDS = ("start_time", "go_cue_time", "end_time", "delay_data",
                      "no_go_bool", "start_data", "target_data",
                      "dec_movement_onset_time")


@dataclass
class ContinuousData:
    """One unbroken recording block, with trial events on the same clock.

        clock_ms        (bins,)            timestamp per bin, from recording start
        neural          (bins, features)   z-scored spike + spike-band-power features
        keep_chans      (n_keep,)          indices into `neural` used by the decoder
        cursor          (bins, 2)          cursor position [X, Y] in px
        emg             (bins, 2)          surface EMG (T16 only)
        dec_move_state  (bins,)            online movement-onset probability, 0-1
        trials          dict               per-trial event times, same clock as clock_ms
    """

    participant: str
    source_path: Path

    clock_ms: np.ndarray
    neural: np.ndarray
    keep_chans: np.ndarray
    bin_s: float
    trials: dict[str, np.ndarray]
    optional: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return self.neural.shape[0]

    @property
    def n_features(self) -> int:
        return self.neural.shape[1]

    @property
    def duration_s(self) -> float:
        return (self.clock_ms[-1] - self.clock_ms[0]) / 1000.0

    @property
    def cursor(self) -> np.ndarray | None:
        return self.optional.get("cursor_data")

    @property
    def emg(self) -> np.ndarray | None:
        return self.optional.get("emg_data")

    @property
    def dec_move_state(self) -> np.ndarray | None:
        v = self.optional.get("dec_move_state")
        return None if v is None else v.ravel()

    def dead_features(self, sd_threshold: float = 0.01) -> np.ndarray:
        """Indices of features that never vary - flat lines in the browser."""
        return np.where(self.neural.std(axis=0) < sd_threshold)[0]

    def __repr__(self) -> str:
        return (f"ContinuousData({self.participant}: {self.n_bins} bins x "
                f"{self.n_features} features, bin={self.bin_s * 1000:.0f} ms, "
                f"{self.duration_s:.1f} s, {self.trials['start_time'].size} trials)")


def load_continuous_data(participant: str, candidate_dirs,
                         figure: str = "fig5") -> ContinuousData:
    """Load a continuous example block (`{figure}_{participant}_continuous.h5`)."""
    path = resolve_data_file(participant, "continuous", figure, candidate_dirs)

    with h5py.File(path, "r") as f:
        for group in ("data", "trial_info"):
            if group not in f:
                raise ValueError(f"{path.name}: missing the '{group}/' group. "
                                 f"File contains: {sorted(f.keys())}")
        if "keep_chans" not in f:
            raise ValueError(f"{path.name}: missing top-level 'keep_chans'.")

        data_grp, trial_grp = f["data"], f["trial_info"]
        missing = [k for k in ("clock_time", "neural_data_norm")
                   if k not in data_grp]
        if missing:
            raise ValueError(f"{path.name}: data/ is missing {missing}. "
                             f"Contains: {sorted(data_grp.keys())}")

        clock_ms = np.asarray(data_grp["clock_time"][()]).ravel().astype(float)
        neural = np.asarray(data_grp["neural_data_norm"][()]).astype(float)
        keep_chans = np.asarray(f["keep_chans"][()]).ravel()
        optional = {k: np.asarray(data_grp[k][()])
                    for k in _CONT_DATA_FIELDS
                    if k in data_grp and k not in ("clock_time", "neural_data_norm")}
        trials = {k: np.asarray(trial_grp[k][()]).astype(float)
                  for k in _CONT_TRIAL_FIELDS if k in trial_grp}

    bin_s = _derive_bin_size(clock_ms, path, strict=False)

    if neural.ndim != 2:
        raise ValueError(f"{path.name}: expected neural_data_norm to be "
                         f"(bins, features), got shape {neural.shape}.")
    if neural.shape[0] != clock_ms.size:
        raise ValueError(f"{path.name}: neural_data_norm has {neural.shape[0]} bins "
                         f"but clock_time has {clock_ms.size} entries.")
    if keep_chans.size and keep_chans.max() >= neural.shape[1]:
        raise ValueError(f"{path.name}: keep_chans references feature "
                         f"{keep_chans.max()} but neural_data_norm has only "
                         f"{neural.shape[1]} features.")
    for name, arr in optional.items():
        if arr.shape[0] != clock_ms.size:
            raise ValueError(f"{path.name}: data/{name} has {arr.shape[0]} bins but "
                             f"clock_time has {clock_ms.size}.")
    if "start_time" not in trials:
        raise ValueError(f"{path.name}: trial_info/ has no 'start_time'.")

    return ContinuousData(
        participant=participant,
        source_path=path,
        clock_ms=clock_ms,
        neural=neural,
        keep_chans=keep_chans,
        bin_s=bin_s,
        trials=trials,
        optional=optional,
    )
