"""
Visualize recorded hand data and the resulting spikes side by side.

Usage:
    uv run src/visualize_data.py data/parsed_data.csv
    uv run src/visualize_data.py data/parsed_data.csv data/parsed_data_spikes.csv

If only the source CSV is given, the spikes CSV is inferred as
<stem>_spikes.csv next to it.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.widgets import CheckButtons

from encoding import N_CHANNELS, N_FINGERS, rotation_deg as compute_rotation_deg

CHANNEL_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Rotation"]
GESTURE_NAMES = {
    0: "HC — Hand closed",
    1: "HO — Hand open",
    2: "Thumb",
    3: "Index",
    4: "Pinky",
    5: "Rock",
    6: "Thumb & Pinky",
}
DIST_COLS     = [f"digit_{i}_distance" for i in range(N_FINGERS)]
COLORS        = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_CHANNELS))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: visualize_data.py <source.csv> [spikes.csv]")
        sys.exit(1)

    src_path = Path(sys.argv[1])
    spk_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                else src_path.with_stem(src_path.stem + "_spikes"))

    if not src_path.exists():
        print(f"not found: {src_path}"); sys.exit(1)
    if not spk_path.exists():
        print(f"spikes file not found: {spk_path}\nRun process_data.py first.")
        sys.exit(1)

    src_rows = load_csv(src_path)
    spk_rows = load_csv(spk_path)

    sim_time = np.array([float(r["sim_time_ms"]) for r in spk_rows])
    labels   = np.array([int(r["label"])         for r in spk_rows])

    # ── input features ────────────────────────────────────────────────────────
    distances = np.array([[float(r[c]) for c in DIST_COLS] for r in src_rows])
    normal_y  = np.array([float(r["palm_normal_y"]) for r in src_rows])
    rotations = np.array([compute_rotation_deg(ny) for ny in normal_y])

    dist_norm = (distances - distances.min(axis=0)) / np.ptp(distances, axis=0).clip(1e-6)
    rot_norm  = (rotations - rotations.min()) / max(np.ptp(rotations), 1e-6)
    inputs    = np.hstack([dist_norm, rot_norm[:, None]])   # (T, 6)

    # ── spike matrix ──────────────────────────────────────────────────────────
    spikes = np.array([[int(r[f"spike_ch{i}"]) for i in range(N_CHANNELS)]
                       for r in spk_rows])   # (T, 6)

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#12121f")
    fig.suptitle(f"{src_path.name}  —  input vs spikes", color="white", fontsize=13)

    # leave left margin for checkboxes
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        hspace=0.45,
        top=0.91, bottom=0.07, left=0.13, right=0.97,
        height_ratios=[2, 2, 0.6],
    )

    ax_in  = fig.add_subplot(gs[0])
    ax_spk = fig.add_subplot(gs[1])
    ax_lbl = fig.add_subplot(gs[2])

    for ax in (ax_in, ax_spk, ax_lbl):
        ax.set_facecolor("#0a0a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlim(sim_time[0], sim_time[-1])

    # ── input panel ───────────────────────────────────────────────────────────
    ax_in.set_title("Input features (normalised)", color="white", fontsize=10, pad=4)
    ax_in.set_ylabel("value", color="white", fontsize=8)
    ax_in.set_ylim(-0.05, 1.15)
    ax_in.set_xlabel("sim time (ms)", color="white", fontsize=8)

    lines = [
        ax_in.plot(sim_time, inputs[:, i], color=COLORS[i],
                   linewidth=1.2, label=CHANNEL_NAMES[i])[0]
        for i in range(N_CHANNELS)
    ]

    # ── spike raster ──────────────────────────────────────────────────────────
    ax_spk.set_title("Spike raster", color="white", fontsize=10, pad=4)
    ax_spk.set_ylabel("channel", color="white", fontsize=8)
    ax_spk.set_xlabel("sim time (ms)", color="white", fontsize=8)
    ax_spk.set_ylim(-0.5, N_CHANNELS - 0.5)
    ax_spk.set_yticks(range(N_CHANNELS))
    ax_spk.set_yticklabels(CHANNEL_NAMES, color="white", fontsize=7)
    for i in range(N_CHANNELS):
        ax_spk.axhline(i, color="#222", linewidth=0.5, zorder=0)

    scatters = [
        ax_spk.scatter(
            sim_time[spikes[:, i] == 1],
            np.full((spikes[:, i] == 1).sum(), i),
            color=COLORS[i], marker="|", s=80, linewidths=1.5, zorder=3,
        )
        for i in range(N_CHANNELS)
    ]

    # ── label band ────────────────────────────────────────────────────────────
    ax_lbl.set_title("Label", color="white", fontsize=10, pad=4)
    ax_lbl.set_yticks([])
    ax_lbl.set_xlabel("sim time (ms)", color="white", fontsize=8)
    unique_labels = sorted(set(labels.tolist()))
    lc_map = {lbl: plt.colormaps["tab10"](k / max(len(unique_labels), 1))
              for k, lbl in enumerate(unique_labels)}

    prev = 0
    for j in range(1, len(sim_time)):
        if labels[j] != labels[prev] or j == len(sim_time) - 1:
            ax_lbl.axvspan(sim_time[prev], sim_time[j],
                           color=lc_map[labels[prev]], alpha=0.6)
            prev = j
    for lbl in unique_labels:
        name = GESTURE_NAMES.get(lbl, f"label {lbl}")
        ax_lbl.plot([], [], color=lc_map[lbl], linewidth=6, label=name)
    ax_lbl.legend(loc="upper right", fontsize=7, framealpha=0.3,
                  labelcolor="white", facecolor="#1e1e32")

    # ── checkboxes ────────────────────────────────────────────────────────────
    chk_ax = fig.add_axes((0.01, 0.25, 0.11, 0.55), facecolor="#1e1e32")
    chk_ax.set_title("Channels", color="white", fontsize=9, pad=6)

    checks = CheckButtons(
        chk_ax,
        labels=CHANNEL_NAMES,
        actives=[True] * N_CHANNELS,
        label_props={"color": [(*COLORS[i][:3], 1.0) for i in range(N_CHANNELS)],
                     "fontsize": [11] * N_CHANNELS},
        frame_props={"edgecolor": [(*COLORS[i][:3], 1.0) for i in range(N_CHANNELS)],
                     "sizes": [120] * N_CHANNELS},
        check_props={"facecolor": [(*COLORS[i][:3], 1.0) for i in range(N_CHANNELS)],
                     "sizes": [80] * N_CHANNELS},
    )

    def on_toggle(label: str | None) -> None:
        if label is None:
            return
        i = CHANNEL_NAMES.index(label)
        visible = lines[i].get_visible()
        lines[i].set_visible(not visible)
        scatters[i].set_visible(not visible)
        # grey out the raster row label
        ax_spk.get_yticklabels()[i].set_alpha(0.3 if visible else 1.0)
        fig.canvas.draw_idle()

    checks.on_clicked(on_toggle)

    plt.show()


if __name__ == "__main__":
    main()
