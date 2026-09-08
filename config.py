"""Central configuration for the preparatory-decoding prototype.

Every knob that changes what the analysis does lives here, so the notebooks stay
about the science. Switching participant is a one-line edit below.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Data selection
# --------------------------------------------------------------------------
PARTICIPANT = "T16"      # "T5" | "T11" | "T16"
FIGURE = "fig1"          # radial-8 attempted wrist movement task
ALIGNMENT = "delay"      # "delay" (t=0 at target onset) | "move" (t=0 at movement start)

# --------------------------------------------------------------------------
# Trial selection
# --------------------------------------------------------------------------
MIN_DELAY_S = 0.8        # require an instructed delay at least this long
GO_TRIALS_ONLY = True    # exclude no-go / catch trials from the primary analysis
MIN_TRIALS_PER_DIRECTION = 10   # refuse to decode from a badly unbalanced set

# --------------------------------------------------------------------------
# Neural feature
# --------------------------------------------------------------------------
FEATURE_TYPE = "spikes"  # "spikes" | "sbp" | "spikes+sbp"

# --------------------------------------------------------------------------
# Task structure
# --------------------------------------------------------------------------
N_DIRECTIONS = 8
CHANCE_LEVEL = 1.0 / N_DIRECTIONS   # 0.125

# The strictly-preparatory window, in ms relative to target onset.
# The earliest go cue in T16 is at 1000 ms; we stop 200 ms short of it because the
# features were smoothed acausally with a Gaussian of sigma = 100 ms, so bins close
# to the go cue already contain post-go activity. See README "Preprocessing caveat".
PREP_WINDOW_MS = (-1000.0, 800.0)

# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
N_SPLITS = 5             # auto-capped to the number of recording blocks
RANDOM_STATE = 42
LOGREG_C = 1.0           # inverse L2 regularisation strength

# Analysis grid. None => use every provided time bin (20 ms for T16).
# Set to e.g. 40.0 to decode on a coarser grid without touching the data.
DECODE_STEP_MS = None
TG_STEP_MS = None        # grid for the temporal-generalisation matrix

# --------------------------------------------------------------------------
# Permutation testing (secondary; off by default)
# --------------------------------------------------------------------------
RUN_PERMUTATIONS = False
N_PERMUTATIONS = 200

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def candidate_data_dirs() -> list[Path]:
    """Directories searched for the .h5 files, in priority order."""
    dirs = []
    env = os.environ.get("INTRACORTICAL_DATA_DIR")
    if env:
        dirs.append(Path(env).expanduser().resolve())
    dirs.append(PROJECT_ROOT / "data")
    dirs.append(PROJECT_ROOT.parent)   # the Dryad download sits here
    return dirs


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
