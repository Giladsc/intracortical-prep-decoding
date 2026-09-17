# Time-resolved decoding of intended movement direction from human motor cortex

An independent tutorial reimplementation of the Figure 1 analysis from Rigotti-Thompson et
al. 2026, built on the authors' public BrainGate2 dataset. Not affiliated with the
BrainGate2 clinical trial, its investigators, or the authors' own repository
([snel-repo/human-preparatory-activity-paper](https://github.com/snel-repo/human-preparatory-activity-paper)).

A small, deliberately transparent research prototype built as five tutorial notebooks.
Each notebook runs an analysis *and* explains the neuroscience behind it, the reasoning
behind each methodological choice, and how to read the result.

## 1. The scientific question

Human motor cortex is active while a person *plans* a movement, before they make it. This
project asks whether that preparatory activity actually specifies **which** movement is
intended, and whether the code it uses holds still:

| | question | evidence | notebook |
|---|---|---|---|
| **Q1** | Does motor cortex contain information about intended movement direction *before* movement begins? | time-resolved 8-direction decoding | 03 |
| **Q2** | *When* during the instructed delay is that information expressed? | decoding accuracy over time | 03 |
| **Q3** | Is the preparatory directional code *stable* or *dynamically changing*? | train-time × test-time generalisation | 04 |

## 2. Dataset

Rigotti-Thompson, M., Nason-Tomaszewski, S. R., Bechefsky, P. H., Acosta, A., Hahn, N. V.,
Avansino, D. T., Richards, B. A., Nicolas, C., Ali, Y. H., Henderson, J. M., Hochberg,
L. R., AuYong, N., & Pandarinath, C. (2026). *Preparatory encoding of intended movement in
human motor cortex and implications for brain-computer interfaces.*

- **Data:** Dryad, [doi:10.5061/dryad.brv15dvr9](https://doi.org/10.5061/dryad.brv15dvr9)
- **Authors' code:** [snel-repo/human-preparatory-activity-paper](https://github.com/snel-repo/human-preparatory-activity-paper)

Recordings come from intracortical microelectrode arrays in the motor cortex of three
participants (T5, T11, T16) in the BrainGate2 clinical trial
([NCT00912041](https://clinicaltrials.gov/study/NCT00912041)).

### Files you need

```
fig1_T16_delay.h5          (~58 MB)   notebooks 00-04
fig5_T16_continuous.h5     (~19 MB)   notebook 00 only, optional
```

`fig1` is the open-loop instructed-delay **radial-8 attempted wrist movement** task, and
`delay` means trials are aligned to the start of the delay period. You do **not** need the
full 1.54 GB dataset. `fig1_T11_delay.h5` and `fig1_T5_delay.h5` let you repeat everything
for the other participants; `fig1_T16_move.h5` is for a later extension.

`fig5_T16_continuous.h5` is used **only by notebook 00**, and only to look at: it is the one
genuinely continuous recording in the dataset (239 s of unbroken bins with real trigger
times), which is what makes an honest `raw.plot` possible. It comes from a different task
and is preprocessed differently, so no number from it enters any result. Notebook 00 still
runs without it — the fig1 epoch views are independent.

## 3. Layout

```
intracortical-prep-decoding/
├── README.md               this file
├── requirements.txt
├── config.py               every setting lives here
├── data/                   put the .h5 files here (or point INTRACORTICAL_DATA_DIR at them)
├── notebooks/
│   ├── 00_signal_browser.ipynb
│   ├── 01_inspect_dataset.ipynb
│   ├── 02_neural_qc_and_psth.ipynb
│   ├── 03_time_resolved_decoding.ipynb
│   └── 04_temporal_generalization.ipynb
├── src/
│   ├── data_loading.py     HDF5 -> TrialData, with structural checks
│   ├── preprocessing.py    direction labels, trial selection, feature handling
│   ├── decoding.py         pipeline, CV scheme, circular error metric
│   └── plotting.py         figure styling and shared annotations
└── results/
    ├── figures/
    └── tables/
```

`config.py` looks for the data in `$INTRACORTICAL_DATA_DIR`, then `./data/`, then the parent
directory — so if the Dryad download already sits one level up, everything runs untouched.

## 4. Install

```powershell
cd intracortical-prep-decoding
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name intracortical-prep `
    --display-name "Python (intracortical-prep)"
```

Then open the notebooks in VS Code (or `jupyter lab`) and select the
**Python (intracortical-prep)** kernel.

> **A note if you have scikit-learn installed globally:** versions before 1.5 were built
> against numpy 1.x and crash on import under numpy 2. The venv above avoids the problem
> entirely; `requirements.txt` pins `scikit-learn>=1.5`.

Dependencies are only numpy, scipy, pandas, h5py, matplotlib, scikit-learn, and Jupyter.
**No MATLAB is required**, unlike the authors' reproduction repository.

## 5. How to run

Run the notebooks in order, top to bottom. Each is self-contained — it loads the data
itself, seeds NumPy at 42, and writes its outputs to `results/`.

| notebook | what it does | runtime |
|---|---|---|
| **00 — signal browser** | the MNE-style views: `raw.plot` over a genuinely continuous recording with real triggers, `epochs.plot` for single fig1 trials, and `epochs.plot_image` per channel — with ipywidgets sliders | seconds |
| **01 — inspect dataset** | prints every HDF5 key, verifies shapes/units/labels, derives the 8 directions, recovers the go-cue timing from cursor motion | seconds |
| **02 — neural QC and PSTHs** | firing rates in Hz, population overview, per-channel direction tuning (ANOVA), direction-conditioned PSTHs, polar tuning curves | seconds |
| **03 — time-resolved decoding** | trial selection, block-grouped CV, one classifier per time bin, accuracy + balanced accuracy + circular error, confusion matrix, optional permutation test | ~1 min |
| **04 — temporal generalisation** | train-time × test-time matrix using the *same* folds as 03, stability index, lag profile | ~1 min |

Notebook 03 saves its cross-validation folds; notebook 04 loads them and verifies a
checksum of the trial set, so the two analyses provably use identical splits.

**To switch participant:** change one line in `config.py` (`PARTICIPANT = "T11"`) and
re-run. Fold counts adapt automatically to the number of recording blocks.

Notebook 00 can also switch on its own, without touching `config.py`: set
`BROWSE["participant"]` in its first cell (`None` hands control back to `config.py`).
That override is local to that notebook's session — notebooks 01–04 always follow
`config.py`, so the analysis cannot silently run on a participant you only meant to
look at. Note that T5 has no continuous recording, so notebook 00 skips its sections
1–6 there and goes straight to the fig1 epoch views.

## 6. What the HDF5 variables mean

| field | shape | meaning |
|---|---|---|
| `spike_data_all` | (trials, bins, chans) | smoothed threshold-crossing counts per bin. Divide by the bin width for Hz. |
| `sbp_data_all` | (trials, bins, chans) | smoothed spike-band power, arbitrary units |
| `keep_chans_all` | (chans,) | electrode indices retained (mean firing rate > 2 Hz) |
| `target_data_all` | (trials, 2) | target position `[X, Y]` in px |
| `start_data_all` | (trials, 2) | start position `[X, Y]` in px |
| `cursor_data_all` | (trials, bins, 2) | cursor position through the trial |
| `delay_data_all` | (trials,) | instructed delay duration, seconds |
| `no_go_bool_all` | (trials,) | 0 = go trial, 1 = no-go (catch) trial |
| `block_id_data_all` | (trials,) | recording block — the cross-validation grouping variable |
| `emg_data_all` | (trials, bins, 2) | surface EMG (T16 only) |
| `t_data` | (bins,) | time relative to the alignment event, ms |

**Direction is not stored** — it is derived from the geometry:

$$\theta = \operatorname{atan2}(y_{target} - y_{start},\ x_{target} - x_{start}) \bmod 2\pi$$

Notebook 01 derives this and *verifies* the angles form eight evenly spaced clusters rather
than assuming it.

## 7. ⚠️ The main preprocessing caveat

**The `fig1` features arrive already smoothed, acausally, with a Gaussian kernel of
σ = 100 ms.** "Acausal" means the kernel is centred, so the value at time *t* averages
activity both before *and after* *t*.

Consequences, carried through every notebook:

1. **No further smoothing is applied.** Adding a second large kernel would compound one
   already there.
2. **No onset-latency claim is made.** If accuracy starts rising at t = 80 ms, that does not
   mean cortex responded at 80 ms — activity from ~180 ms has leaked backwards into that
   bin. The apparent onset is necessarily earlier than the true one.
3. **Adjacent time bins are not independent.** Bins 20 ms apart share most of their
   smoothing window, which is why the decoding curve looks smooth and why per-time-point
   significance testing needs a max-statistic correction.

Our claims therefore concern the **presence and temporal structure** of preparatory
information, never a precise neural onset time.

A second, task-specific caveat: T16's instructed delays are 1.0 s and 1.5 s, so **the go cue
does not fall at a single time**. After the earliest go cue the trials are a mixture of
preparing and moving. We decode the whole epoch but restrict the *preparatory* claim to
`t ∈ [−1000, +800] ms` — stopping 200 ms short of the earliest go cue — and shade the mixed
region on every figure.

## 8. Cross-validation strategy

**StratifiedGroupKFold**, with recording block as the group and direction as the
stratification label.

Trials were recorded in blocks minutes apart, between which the neural signal drifts. Under
a random trial split, a classifier can partly identify the *block* from its drift signature
and exploit any block/direction imbalance — inflating accuracy without learning anything
about intended movement. Grouping by block closes that door, and the surviving claim is
stronger: the directional code generalises to recording blocks the decoder never saw.

The number of folds is capped at the number of blocks (5 for T16's 6 blocks, 4 for T11's 4).
If grouped folds cannot be built with all eight directions present on both sides,
`make_cv` falls back to `StratifiedKFold` with a loud warning, and the scheme name is
written into the output table so a within-block result can never be misread as a
cross-block one.

**The splits are generated once and reused at every time point** and by notebook 04, so
differences across the time course reflect the data and never a reshuffle.

### Deviation from the authors' analysis — flagged, not hidden

The original Figure 1 analysis uses **leave-one-out cross-validation** and an **SVM** on
**spike + spike-band-power** features. We deliberately use **block-grouped CV** with
**spike-only L2 logistic regression**. This is a stricter generalisation test and a simpler,
more interpretable model, chosen so every stage is understandable. Where our accuracies are
lower than the paper's, this is a likely reason — not a discrepancy in the data.

## 9. Outputs

```
results/figures/
├── continuous_channel_sd.png      which features carry signal at all (nb 00)
├── raw_browser_continuous.png     raw.plot-style browser with triggers   (nb 00)
├── raw_browser_single_trial.png   one trial: go cue -> EMG -> cursor     (nb 00)
├── epoch_browser_trial.png        epochs.plot-style single fig1 trial    (nb 00)
├── epochs_image_channel.png       erpimage, most-tuned channel           (nb 00)
├── epochs_image_untuned.png       the same plot with no tuning to see    (nb 00)
├── population_overview.png        mean firing rate across channels and trials
├── tuning_f_distribution.png      per-channel direction tuning, preparation vs baseline
├── example_psths.png              firing rate by direction, most-tuned channels
├── tuning_polar.png               polar tuning curves
├── preferred_directions.png       preferred direction of every channel
├── time_resolved_decoding.png     accuracy and angular error over time    (main result)
├── confusion_preparatory.png      which directions get confused with which
├── temporal_generalization.png    train-time x test-time heatmap          (main result)
└── generalization_by_lag.png      accuracy vs train/test lag

results/tables/
├── dataset_summary.csv                    one row per dataset property
├── channel_tuning.csv                     per-channel F, preferred direction, modulation
├── time_resolved_decoding.csv             long format: t_ms x fold x metrics
├── oof_predictions.npz                    out-of-fold predictions per trial per time
├── cv_splits.npz                          the folds, shared with notebook 04
├── temporal_generalization.npy            (n_times, n_times) accuracy matrix
├── temporal_generalization_folds.npy      (n_folds, n_times, n_times)
└── temporal_generalization_times.npy      the time axis, ms
```

## 10. Headline findings (T16)

| | result |
|---|---|
| baseline, before target (−1000 to −200 ms) | accuracy **0.135** (chance 0.125), angular error **90.0°** (chance 90.0°) |
| preparatory (300 to 800 ms) | accuracy **0.294** — 2.4× chance — angular error **59.4°** |
| peak within the delay | **0.361** at t = +460 ms |
| errors during preparation | 34% land on an *adjacent* direction, only 6.6% on the opposite one |
| stability index, informative delay | **0.94** — a decoder transfers across the delay almost as well as it works at its own training time |
| preparation → movement transfer | **0.309**, slightly above the preparatory diagonal itself |

**Q1: yes** — direction is decodable well above chance during the delay, before any movement,
and the pre-target baseline sits at chance as it must.
**Q2:** information rises after target onset, peaks early (~460 ms), then settles to a lower
sustained level across the delay. Timing is interpreted conservatively.
**Q3: largely stable** — the generalisation matrix shows a broad square block rather than a
narrow diagonal band, so one population pattern is established and then held, rather than
continuously changing.

## 11. Scope

Deliberately excluded from this prototype: deep learning, latent-state models, dPCA,
manifold alignment, spike sorting, cross-participant and cross-effector transfer. The point
is to understand the data before adding complexity.

Natural next steps, in order: the no-go trial control (set `GO_TRIALS_ONLY = False`);
preparation → execution generalisation using `fig1_T16_move.h5` with held-out trials; then
the other participants.
