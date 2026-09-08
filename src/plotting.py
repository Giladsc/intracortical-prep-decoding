"""Figure styling and the recurring annotations.

Plumbing only - imported so the notebook cells stay about the analysis rather than
about matplotlib. The one design decision worth knowing:

DIRECTION COLOURS. Movement direction is a *circular* variable, so the 8 classes are
coloured by a cyclic scale in which hue tracks angle and opposite directions sit
opposite on the wheel. The steps alternate dark/light so that lightness carries the
signal too, which is what keeps the scale readable for colour-vision-deficient
readers: the minimum perceptual separation between neighbouring directions is
dE ~= 21 under normal, deuteranope and protanope simulation alike (the usual target
is 8), and every step clears 3:1 contrast against white. Direction is additionally
encoded by position in the polar plots and by direct labels on the PSTHs, so it is
never carried by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# Validated 8-step cyclic direction scale, indexed by direction class 0..7
# (0 = 0 deg = rightward, increasing counter-clockwise in 45 deg steps).
DIRECTION_COLORS = [
    "#0b3d91",  # 0    right
    "#6c95cc",  # 45   up-right
    "#146b52",  # 90   up
    "#669f7c",  # 135  up-left
    "#8a4b00",  # 180  left
    "#be883f",  # 225  down-left
    "#8e1f5a",  # 270  down
    "#c77c9d",  # 315  down-right
]

# Compass glyphs, a second (non-colour) channel for direction identity.
DIRECTION_ARROWS = ["→", "↗", "↑", "↖",
                    "←", "↙", "↓", "↘"]

# Neutral ink: text never wears the series colour.
INK = "#1c1c1c"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"
SURFACE = "#ffffff"
ACCENT = "#0b3d91"      # single-series lines
CHANCE_COLOR = "#8a8a8a"


def use_house_style() -> None:
    """Recessive grid and axes, readable type, no chartjunk. Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 150,
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#b4b4b4",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
    })


def direction_legend_labels(centres_rad: np.ndarray) -> list[str]:
    """'0 deg ->', '45 deg /' ... - angle plus a compass glyph."""
    return [f"{np.degrees(a):.0f}° {DIRECTION_ARROWS[i]}"
            for i, a in enumerate(centres_rad)]


# ---------------------------------------------------------------------------
# Event annotations shared by every time-axis figure
# ---------------------------------------------------------------------------
def annotate_events(ax, go_cue_times_ms=(), prep_window_ms=None,
                    label_events=True, post_go_shade=True) -> None:
    """Target onset, go cue(s), and the post-go region on a time axis.

    `prep_window_ms` marks the window over which a *preparatory* claim is defensible;
    everything after the earliest go cue mixes preparation with execution and is
    shaded to say so.

    In a grid of small multiples set `label_events=False` on all but one panel: the
    rules still appear everywhere, but the text is written once instead of six times.
    """
    ax.axvline(0.0, color=INK, lw=1.2, ls="-", zorder=2)
    if label_events:
        ax.annotate("target on", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points",
                    fontsize=8, color=INK_MUTED, ha="left", va="top")

    xmax = ax.get_xlim()[1]
    for i, t_go in enumerate(sorted(go_cue_times_ms)):
        ax.axvline(t_go, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        if label_events:
            ax.annotate(f"go cue ({t_go / 1000:g}s delay)",
                        xy=(t_go, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -12 - 12 * i), textcoords="offset points",
                        fontsize=8, color=INK_MUTED, ha="left", va="top")

    if post_go_shade and len(go_cue_times_ms):
        first_go = min(go_cue_times_ms)
        ax.axvspan(first_go, xmax, color="#f2f2f2", zorder=0, lw=0)

    if prep_window_ms is not None:
        ax.axvspan(prep_window_ms[0], prep_window_ms[1],
                   color="#eef3fb", zorder=0, lw=0)


def add_chance_line(ax, chance: float, label: str = "chance (1/8)") -> None:
    ax.axhline(chance, color=CHANCE_COLOR, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.annotate(label, xy=(1.0, chance), xycoords=("axes fraction", "data"),
                xytext=(-4, 4), textcoords="offset points",
                fontsize=8, color=INK_MUTED, ha="right", va="bottom")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_population_overview(t_ms, mean_hz, sem_hz, go_cue_times_ms=(),
                             prep_window_ms=None, title="", ax=None):
    """Mean firing rate across channels and trials against time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.4))
    ax.set_xlim(t_ms[0], t_ms[-1])
    annotate_events(ax, go_cue_times_ms, prep_window_ms)
    ax.fill_between(t_ms, mean_hz - sem_hz, mean_hz + sem_hz,
                    color=ACCENT, alpha=0.18, lw=0, zorder=3)
    ax.plot(t_ms, mean_hz, color=ACCENT, zorder=4)
    ax.set_xlabel("time from target onset (ms)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(title)
    return ax


def plot_direction_psths(t_ms, psth_by_dir, channel_ids, centres_rad,
                         sem_by_dir=None, go_cue_times_ms=(), prep_window_ms=None,
                         suptitle="", ncols=3):
    """One panel per channel; one line per direction.

    psth_by_dir : (n_channels, n_directions, n_bins) mean firing rate in Hz
    """
    n_ch = len(channel_ids)
    ncols = min(ncols, n_ch)
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.9 * nrows),
                             squeeze=False, sharex=True)
    labels = direction_legend_labels(centres_rad)

    for k, ch in enumerate(channel_ids):
        ax = axes[k // ncols][k % ncols]
        ax.set_xlim(t_ms[0], t_ms[-1])
        annotate_events(ax, go_cue_times_ms, prep_window_ms,
                        label_events=(k == 0))
        for d in range(psth_by_dir.shape[1]):
            if sem_by_dir is not None:
                ax.fill_between(t_ms,
                                psth_by_dir[k, d] - sem_by_dir[k, d],
                                psth_by_dir[k, d] + sem_by_dir[k, d],
                                color=DIRECTION_COLORS[d], alpha=0.13, lw=0, zorder=3)
            ax.plot(t_ms, psth_by_dir[k, d], color=DIRECTION_COLORS[d],
                    lw=1.8, zorder=4, label=labels[d] if k == 0 else None)
        ax.set_title(f"channel {ch}", fontsize=10)
        if k % ncols == 0:
            ax.set_ylabel("firing rate (Hz)")
        if k // ncols == nrows - 1:
            ax.set_xlabel("time from target onset (ms)")

    for k in range(n_ch, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.1, handlelength=1.4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97 if suptitle else 1))
    return fig


def plot_polar_tuning(mean_by_dir, channel_ids, centres_rad, suptitle="", ncols=4):
    """Mean preparatory activity as a function of direction, on polar axes.

    mean_by_dir : (n_channels, n_directions) mean firing rate in Hz
    """
    n_ch = len(channel_ids)
    ncols = min(ncols, n_ch)
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows),
                             squeeze=False, subplot_kw={"projection": "polar"})
    closed_angles = np.append(centres_rad, centres_rad[0])

    for k, ch in enumerate(channel_ids):
        ax = axes[k // ncols][k % ncols]
        vals = mean_by_dir[k]
        ax.plot(closed_angles, np.append(vals, vals[0]),
                color=INK_MUTED, lw=1.4, zorder=3)
        for d, a in enumerate(centres_rad):
            ax.plot([a], [vals[d]], marker="o", markersize=8,
                    color=DIRECTION_COLORS[d], markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=4)
        ax.set_title(f"channel {ch}", fontsize=10, pad=12)
        ax.set_xticks(centres_rad)
        ax.set_xticklabels([f"{int(np.degrees(a))}°" for a in centres_rad],
                           fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_rlabel_position(112.5)   # between spokes, so labels miss the markers
        ax.grid(color=GRID, lw=0.7)

    for k in range(n_ch, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_time_resolved(t_ms, acc_by_fold, chance, go_cue_times_ms=(),
                       prep_window_ms=None, angular_error=None,
                       chance_angle=None, title=""):
    """Decoding accuracy against time, with across-fold spread.

    acc_by_fold : (n_folds, n_times)
    """
    if angular_error is None:
        fig, ax_acc = plt.subplots(figsize=(9, 4.2))
        axes = [ax_acc]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
        ax_acc = axes[0]

    mean = acc_by_fold.mean(axis=0)
    sem = acc_by_fold.std(axis=0, ddof=1) / np.sqrt(acc_by_fold.shape[0])

    ax_acc.set_xlim(t_ms[0], t_ms[-1])
    annotate_events(ax_acc, go_cue_times_ms, prep_window_ms)
    add_chance_line(ax_acc, chance)
    ax_acc.fill_between(t_ms, mean - sem, mean + sem, color=ACCENT, alpha=0.20,
                        lw=0, zorder=3)
    ax_acc.plot(t_ms, mean, color=ACCENT, zorder=4)
    ax_acc.set_ylabel("decoding accuracy")
    ax_acc.set_ylim(0, max(1.0, float((mean + sem).max()) * 1.08))
    ax_acc.set_title(title)

    if angular_error is not None:
        ax_err = axes[1]
        err_mean = angular_error.mean(axis=0)
        err_sem = angular_error.std(axis=0, ddof=1) / np.sqrt(angular_error.shape[0])
        ax_err.set_xlim(t_ms[0], t_ms[-1])
        annotate_events(ax_err, go_cue_times_ms, prep_window_ms, label_events=False)
        if chance_angle is not None:
            ax_err.axhline(chance_angle, color=CHANCE_COLOR, lw=1.2, ls=(0, (5, 4)),
                           zorder=2)
            ax_err.annotate(f"chance ({chance_angle:.0f}°)",
                            xy=(1.0, chance_angle),
                            xycoords=("axes fraction", "data"),
                            xytext=(-4, 4), textcoords="offset points",
                            fontsize=8, color=INK_MUTED, ha="right", va="bottom")
        ax_err.fill_between(t_ms, err_mean - err_sem, err_mean + err_sem,
                            color="#8a4b00", alpha=0.20, lw=0, zorder=3)
        ax_err.plot(t_ms, err_mean, color="#8a4b00", zorder=4)
        ax_err.set_ylabel("mean |angular error| (°)")
        ax_err.invert_yaxis()   # better performance upward, as in the panel above

    axes[-1].set_xlabel("time from target onset (ms)")
    fig.tight_layout()
    return fig


def plot_temporal_generalization(matrix, t_ms, chance, go_cue_times_ms=(),
                                 prep_window_ms=None, title=""):
    """Train-time x test-time accuracy heatmap.

    Colour diverges around chance: grey means 'no directional information', so the
    eye reads the extent of the informative region directly rather than having to
    compare against a legend.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    vmax = float(np.nanmax(matrix))
    norm = TwoSlopeNorm(vmin=min(float(np.nanmin(matrix)), chance - 1e-6),
                        vcenter=chance, vmax=max(vmax, chance + 1e-6))
    extent = (t_ms[0], t_ms[-1], t_ms[0], t_ms[-1])
    im = ax.imshow(matrix, origin="lower", aspect="equal", extent=extent,
                   cmap="RdBu_r", norm=norm, interpolation="nearest")

    ax.plot([t_ms[0], t_ms[-1]], [t_ms[0], t_ms[-1]],
            color=INK, lw=0.8, ls=(0, (3, 3)), zorder=3)
    for t_event in [0.0, *go_cue_times_ms]:
        style = dict(color=INK_MUTED, lw=0.8,
                     ls="-" if t_event == 0 else (0, (4, 3)), zorder=3)
        ax.axvline(t_event, **style)
        ax.axhline(t_event, **style)

    ax.set_xlabel("test time (ms from target onset)")
    ax.set_ylabel("train time (ms from target onset)")
    ax.set_title(title)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("decoding accuracy", labelpad=10)
    cbar.ax.axhline(chance, color=INK, lw=1.0)
    # "chance" sits inside the bar, so it cannot collide with the axis label outside it.
    cbar.ax.annotate("chance", xy=(0.5, chance), xycoords=("axes fraction", "data"),
                     xytext=(0, 3), textcoords="offset points",
                     fontsize=7.5, color=INK, ha="center", va="bottom")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Signal browsers (notebook 00) - the MNE raw.plot / epochs.plot equivalents
# ---------------------------------------------------------------------------
# Stacked traces are drawn in neutral ink, never coloured by channel identity:
# with tens of overlapping series a categorical palette carries no information
# and only adds visual noise. Colour is reserved for the event markers, whose
# four steps clear a minimum all-pairs dE of 41 under normal, deuteranope and
# protanope simulation and 4.3:1 contrast on white - and each is additionally
# given its own line style, so identity never rests on hue alone.
EVENT_STYLES = {
    "trial start": {"color": "#1c1c1c", "ls": "-"},
    "target on":   {"color": "#1c1c1c", "ls": "-"},
    "go cue":      {"color": "#c2410c", "ls": (0, (5, 2))},
    "trial end":   {"color": "#1d4ed8", "ls": (0, (1, 2))},
    "move onset":  {"color": "#7a7a7a", "ls": (0, (3, 2, 1, 2))},
}

TRACE_COLOR = "#3d3d3d"


def _draw_events(ax, events, xlim, label=True):
    """Vertical rules for event times. `events` maps a name to an array of times."""
    seen = []
    for name, times in (events or {}).items():
        style = EVENT_STYLES.get(name, {"color": INK_MUTED, "ls": "-"})
        times = np.asarray(times, dtype=float)
        times = times[np.isfinite(times) & (times >= xlim[0]) & (times <= xlim[1])]
        for k, t in enumerate(times):
            ax.axvline(t, color=style["color"], ls=style["ls"], lw=1.1, zorder=5,
                       label=name if (label and k == 0 and name not in seen) else None)
        if times.size:
            seen.append(name)
    return seen


def plot_signal_browser(t_s, signals, labels, events=None, aux=None,
                        start_s=None, duration_s=10.0, n_channels=20,
                        scale=None, units="a.u.", title="",
                        xlabel="time (s)", figsize=None):
    """MNE raw.plot-style stacked browser over a continuous signal.

    t_s      (n_samples,)             time axis in seconds
    signals  (n_samples, n_channels)  one column per channel
    labels   list of channel names, same length as signals' columns
    events   {name: array of times} - drawn as vertical rules
    aux      {name: (n_samples,) array} - auxiliary rows in a panel below
    scale    vertical spacing between channels, in signal units. None => a
             robust automatic value (4x the median channel SD), the way MNE
             picks a default and then lets you change it.
    """
    t_s = np.asarray(t_s, dtype=float)
    signals = np.asarray(signals, dtype=float)

    start_s = float(t_s[0]) if start_s is None else float(start_s)
    stop_s = start_s + float(duration_s)
    win = (t_s >= start_s) & (t_s <= stop_s)
    if not win.any():
        raise ValueError(f"No samples between {start_s:.2f} and {stop_s:.2f}; "
                         f"the data span {t_s[0]:.2f} to {t_s[-1]:.2f}.")

    n_show = int(min(n_channels, signals.shape[1]))
    seg = signals[win][:, :n_show]
    t_win = t_s[win]

    if scale is None:
        # Median peak-to-peak rather than SD: it is what actually determines
        # whether neighbouring traces collide, so the default separates them.
        ptp = np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)
        ptp = ptp[np.isfinite(ptp) & (ptp > 0)]
        scale = 1.1 * float(np.median(ptp)) if ptp.size else 1.0
    scale = float(scale) or 1.0

    n_aux = len(aux) if aux else 0
    if figsize is None:
        figsize = (12, max(4.0, 0.32 * n_show + 0.7 * n_aux + 1.6))
    if n_aux:
        fig, (ax, ax_aux) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [max(3, 0.32 * n_show), 0.55 * n_aux]})
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_aux = None

    # Channels stacked top-to-bottom, as MNE draws them.
    offsets = np.arange(n_show)[::-1] * scale
    for i in range(n_show):
        ax.plot(t_win, seg[:, i] + offsets[i], color=TRACE_COLOR, lw=0.7, zorder=4)

    ax.set_xlim(start_s, stop_s)
    ax.set_ylim(-scale, offsets[0] + scale)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels[:n_show], fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.set_ylabel("channel")
    # Extra pad leaves room for the event legend, which sits above the axes.
    ax.set_title(title, pad=26 if events else 6)

    seen = _draw_events(ax, events, (start_s, stop_s))

    # Amplitude scale bar, so the vertical axis stays quantitative. Drawn just
    # outside the right edge so it never sits on top of a trace.
    bar_x = stop_s + 0.012 * (stop_s - start_s)
    ax.plot([bar_x, bar_x], [0, scale], color=INK, lw=2.2, zorder=6,
            solid_capstyle="butt", clip_on=False)
    ax.annotate(f"{scale:.2g} {units}", xy=(bar_x, scale / 2),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=INK, ha="left", va="center",
                annotation_clip=False)

    if seen:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(seen),
                  fontsize=8, handlelength=1.8)

    if ax_aux is not None:
        aux_offsets = np.arange(n_aux)[::-1].astype(float)
        for k, (name, series) in enumerate(aux.items()):
            s = np.asarray(series, dtype=float)[win]
            rng = np.nanmax(np.abs(s)) or 1.0
            ax_aux.plot(t_win, 0.42 * s / rng + aux_offsets[k],
                        color=ACCENT, lw=1.1, zorder=4)
            ax_aux.annotate(name, xy=(start_s, aux_offsets[k]),
                            xytext=(4, 6), textcoords="offset points",
                            fontsize=7.5, color=INK_MUTED, va="bottom")
        ax_aux.set_ylim(-0.6, n_aux - 0.4)
        ax_aux.set_yticks([])
        ax_aux.grid(False)
        ax_aux.set_ylabel("behaviour")
        _draw_events(ax_aux, events, (start_s, stop_s), label=False)
        ax_aux.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)

    fig.tight_layout()
    return fig


def plot_epoch_browser(t_ms, epoch, labels, events=None, aux=None,
                       n_channels=20, scale=None, units="Hz", title=""):
    """MNE epochs.plot-style view of ONE trial: channels stacked over time.

    epoch  (n_bins, n_channels) for a single trial, in Hz.
    Time is in ms relative to the alignment event, so event times are in ms too.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    return plot_signal_browser(
        t_ms, epoch, labels,
        events=events, aux=aux,
        start_s=float(t_ms[0]), duration_s=float(t_ms[-1] - t_ms[0]),
        n_channels=n_channels, scale=scale, units=units, title=title,
        xlabel="time from target onset (ms)",
    )


def plot_epochs_image(t_ms, trials, sort_labels, centres_rad,
                      go_cue_times_ms=(), prep_window_ms=None, title="",
                      cmap="Blues"):
    """MNE epochs.plot_image-style erpimage for one channel.

    trials       (n_trials, n_bins) firing rate in Hz for a single channel
    sort_labels  (n_trials,) direction class per trial; trials are grouped by it

    A firing rate is a magnitude, so the heatmap uses a single-hue sequential
    ramp - never a rainbow, and never diverging around an arbitrary midpoint.
    """
    trials = np.asarray(trials, dtype=float)
    sort_labels = np.asarray(sort_labels)
    order = np.argsort(sort_labels, kind="stable")
    sorted_trials = trials[order]
    sorted_labels = sort_labels[order]
    n_dir = len(centres_rad)

    # A dedicated colourbar column, so the image and the mean panel below it keep
    # exactly the same width - otherwise the shared time axis reads as misaligned.
    fig = plt.figure(figsize=(9.0, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[40, 1.2], height_ratios=[3, 1.5])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[1, 0], sharex=ax_img)
    cax = fig.add_subplot(gs[0, 1])

    # Robust colour limits: a handful of high-rate bins would otherwise wash the
    # whole image out to near-white.
    vmin, vmax = np.nanpercentile(sorted_trials, [1, 99])
    extent = (t_ms[0], t_ms[-1], sorted_trials.shape[0], 0)
    im = ax_img.imshow(sorted_trials, aspect="auto", origin="upper", extent=extent,
                       cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax_img.set_ylabel("trial (grouped by direction)")
    ax_img.set_title(title)
    ax_img.grid(False)

    # Direction sidebar: which block of rows belongs to which direction.
    boundaries = np.searchsorted(sorted_labels, np.arange(n_dir + 1))
    span = t_ms[-1] - t_ms[0]
    bar_w = 0.012 * span
    for d in range(n_dir):
        lo, hi = boundaries[d], boundaries[d + 1]
        if hi <= lo:
            continue
        ax_img.add_patch(Rectangle((t_ms[0] - bar_w * 1.4, lo), bar_w, hi - lo,
                                   color=DIRECTION_COLORS[d], clip_on=False, lw=0))
        ax_img.annotate(DIRECTION_ARROWS[d],
                        xy=(t_ms[0] - bar_w * 2.4, (lo + hi) / 2),
                        fontsize=8, color=INK, ha="right", va="center",
                        annotation_clip=False)
    ax_img.set_xlim(t_ms[0], t_ms[-1])

    for t_event in [0.0, *go_cue_times_ms]:
        ax_img.axvline(t_event, color=INK, lw=1.0,
                       ls="-" if t_event == 0 else (0, (4, 3)), zorder=5)

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label("firing rate (Hz)")
    ax_img.tick_params(axis="x", labelbottom=False)

    # Per-direction means underneath, on the shared time axis.
    ax_mean.set_xlim(t_ms[0], t_ms[-1])
    annotate_events(ax_mean, go_cue_times_ms, prep_window_ms, label_events=False)
    for d in range(n_dir):
        m = sort_labels == d
        if m.any():
            ax_mean.plot(t_ms, trials[m].mean(axis=0), color=DIRECTION_COLORS[d],
                         lw=1.6, zorder=4)
    ax_mean.set_xlabel("time from target onset (ms)")
    ax_mean.set_ylabel("mean rate (Hz)")
    return fig
