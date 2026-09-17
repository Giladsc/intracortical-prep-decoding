# Data directory

The `.h5` files are **not** included in this repository — they are ~58 MB each, and the
full dataset is 1.54 GB.

## Where the code looks

`config.py` searches these locations in order and uses the first one that contains the file:

1. `$INTRACORTICAL_DATA_DIR` — set this environment variable to point anywhere
2. `./data/` — this directory
3. the parent directory of the project (where the Dryad download may already sit)

So you can either drop the files here, or leave them wherever you downloaded them and set
the environment variable.

## What to download

From Dryad, [doi:10.5061/dryad.brv15dvr9](https://doi.org/10.5061/dryad.brv15dvr9):

| file | size | needed for |
|---|---|---|
| `fig1_T16_delay.h5` | ~58 MB | **notebooks 00–04** — the main required file |
| `fig5_T16_continuous.h5` | ~19 MB | **notebook 00** — the only genuinely continuous recording, used for the raw-signal browser |
| `fig1_T11_delay.h5` | ~38 MB | optional: repeat the analysis on participant T11 |
| `fig1_T5_delay.h5` | ~40 MB | optional: participant T5 |
| `fig1_T16_move.h5` | ~68 MB | optional: the later preparation → execution extension |

`fig5_T16_continuous.h5` is a different task (closed-loop BCI cursor control) with different
preprocessing (causal exponential smoothing, z-scored). Notebook 00 uses it **only for
looking at the signal** — no number from it enters any result. Skip it if you only want the
epoch views; notebook 00 will tell you the file is missing and the rest still runs.

**A continuous block exists for T11 and T16 only.** T5 has the fig1 task and nothing else,
so under `PARTICIPANT = "T5"` notebook 00 skips its sections 1–6 and goes straight to the
fig1 epoch views. `config.available_participants("continuous", "fig5")` reports this from
what is actually on disk.

Do not download the whole 1.54 GB archive unless you want the other tasks (finger
movements, curved reaches, the Pac-Man task, closed-loop control).

## A note on participant differences

The notebooks are written to adapt, but the participants genuinely differ, and the checks in
notebook 01 exist because of it:

| | trials | channels | blocks | delays |
|---|---|---|---|---|
| **T16** | 385 | 61 | 6 | 1.0, 1.5 s |
| **T11** | 266 | 58 | 4 | 1.0, 1.5 s |
| **T5** | 342 | 48 | 3 | 0.2, 0.6, 1.0, 1.4 s, plus NaN entries |

T5 is the interesting case: its delays vary widely, so the `MIN_DELAY_S = 0.8` filter does
real work there (it is a no-op for T16 and T11), and some trials carry non-finite `delay`
and `no_go` values that the trial-selection step reports and removes.

The number of cross-validation folds is capped at the number of recording blocks, so T11
runs with 4 folds and T5 with 3, each with a printed notice.
