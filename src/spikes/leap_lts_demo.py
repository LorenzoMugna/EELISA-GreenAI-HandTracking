"""
Leap Motion → LTS neuron spike raster  (GUI entry point).

Two encoding modes (toggle with the radio buttons):
  Tonic  — current ∝ fingertip distance / palm rotation angle
  Phasic — current ∝ |Δdistance| or |Δangle| per frame (change-selective)

A two-step GUI calibration phase (tonic mode only) auto-detects the
extended and fist poses before the raster starts.

Run with:
    uv run src/spikes/leap_lts_demo.py
The Ultraleap service must be running.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import leap
from matplotlib.widgets import RadioButtons

from leap_lts_core import (
    WINDOW_MS, N_CHANNELS, N_FINGERS, CHANNEL_NAMES,
    MODE_TONIC, MODE_PHASIC,
    CAL_EXTEND, CAL_DONE,
    spike_history, bar_values, sim_time_ref, enc_mode,
    cal_phase, cal_progress, cal_max, cal_min,
    stop_event,
    FingertipListener, start_background_threads, configure_udp,
)

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_CHANNELS))

# ── themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": dict(
        fig_bg      = "#12121f",
        ax_bg       = "#0a0a1a",
        text        = "white",
        text_sub    = "#aaaacc",
        spine       = "#333",
        grid        = "#222",
        cal_bg      = "#12121fee",
        cal_prog_bg = "#2a2a3e",
        radio_bg    = "#1e1e32",
    ),
    "light": dict(
        fig_bg      = "#f4f4f8",
        ax_bg       = "#ffffff",
        text        = "#111111",
        text_sub    = "#555577",
        spine       = "#bbbbbb",
        grid        = "#e0e0e0",
        cal_bg      = "#f4f4f8ee",
        cal_prog_bg = "#d0d0e8",
        radio_bg    = "#e4e4f0",
    ),
}


# ── GUI ───────────────────────────────────────────────────────────────────────
def run_gui(theme: str = "dark") -> None:
    t = THEMES[theme]
    fig = plt.figure(figsize=(12, 7), facecolor=t["fig_bg"])

    # ── raster ────────────────────────────────────────────────────────────────
    raster_ax = fig.add_axes((0.08, 0.38, 0.78, 0.54))
    raster_ax.set_facecolor(t["ax_bg"])
    raster_ax.set_xlim(0, WINDOW_MS)
    raster_ax.set_ylim(-0.5, N_CHANNELS - 0.5)
    raster_ax.set_xlabel("time  (ms)", color=t["text"], labelpad=6)
    raster_ax.tick_params(colors=t["text"])
    raster_ax.set_yticks(range(N_CHANNELS))
    raster_ax.set_yticklabels(CHANNEL_NAMES, color=t["text"])
    for spine in raster_ax.spines.values():
        spine.set_edgecolor(t["spine"])
    for i in range(N_CHANNELS):
        raster_ax.axhline(i, color=t["grid"], linewidth=0.5, zorder=0)

    scatters = [
        raster_ax.scatter([], [], s=60, color=COLORS[i], marker="|", linewidths=2, zorder=3)
        for i in range(N_CHANNELS)
    ]

    # ── input bar chart ───────────────────────────────────────────────────────
    bar_ax = fig.add_axes((0.08, 0.07, 0.78, 0.24))
    bar_ax.set_facecolor(t["ax_bg"])
    bar_ax.set_xlim(-0.5, N_CHANNELS - 0.5)
    bar_ax.set_ylim(0, 200)
    bar_ax.set_xticks(range(N_CHANNELS))
    bar_ax.set_xticklabels(CHANNEL_NAMES, color=t["text"], fontsize=8)
    bar_ax.set_ylabel("input value", color=t["text"], labelpad=6)
    bar_ax.tick_params(colors=t["text"])
    for spine in bar_ax.spines.values():
        spine.set_edgecolor(t["spine"])

    bars = bar_ax.bar(range(N_CHANNELS), [0.0] * N_CHANNELS,
                      color=COLORS, width=0.6, zorder=3)
    range_lines = [
        (bar_ax.plot([i - 0.3, i + 0.3], [0, 0],
                     color=COLORS[i], lw=1.2, ls="--", alpha=0.0)[0],
         bar_ax.plot([i - 0.3, i + 0.3], [0, 0],
                     color=COLORS[i], lw=1.2, ls="--", alpha=0.0)[0])
        for i in range(N_FINGERS)
    ]
    rot_idx = N_FINGERS
    bar_ax.plot([rot_idx - 0.3, rot_idx + 0.3], [0,   0],
                color=COLORS[rot_idx], lw=1.2, ls="--", alpha=0.5)
    bar_ax.plot([rot_idx - 0.3, rot_idx + 0.3], [180, 180],
                color=COLORS[rot_idx], lw=1.2, ls="--", alpha=0.5)

    # ── encoding mode radio buttons ───────────────────────────────────────────
    radio_ax = fig.add_axes((0.87, 0.12, 0.12, 0.18), facecolor=t["radio_bg"])
    radio_ax.set_title("Encoding", color=t["text"], fontsize=9, pad=4)
    radio = RadioButtons(
        radio_ax,
        labels=[MODE_TONIC, MODE_PHASIC],
        active=0,
        label_props={"color": [t["text"], t["text"]], "fontsize": [10, 10]},
        radio_props={"facecolor": ["#7c4dff", "#ff4d7c"]},
    )

    bar_ylabel = bar_ax.yaxis.get_label()

    def on_mode_change(label: str | None) -> None:
        if label is None:
            return
        enc_mode[0] = label
        show_cal_lines = label == MODE_TONIC
        if show_cal_lines:
            bar_ylabel.set_text("input value")
            bar_ax.set_ylim(0, max(max(cal_max) * 1.15, 200))
        else:
            bar_ylabel.set_text("|delta| per frame")
            bar_ax.set_ylim(0, 50)
        for lo_line, hi_line in range_lines:
            lo_line.set_visible(show_cal_lines)
            hi_line.set_visible(show_cal_lines)
        fig.canvas.draw_idle()

    radio.on_clicked(on_mode_change)

    # ── calibration overlay ───────────────────────────────────────────────────
    cal_ax = fig.add_axes((0.08, 0.38, 0.78, 0.54))
    cal_ax.set_facecolor(t["cal_bg"])
    cal_ax.set_xlim(0, 1)
    cal_ax.set_ylim(0, 1)
    cal_ax.set_axis_off()

    cal_title = cal_ax.text(0.5, 0.72, "", ha="center", va="center",
                            color=t["text"], fontsize=16, fontweight="bold",
                            transform=cal_ax.transAxes)
    cal_sub   = cal_ax.text(0.5, 0.55, "", ha="center", va="center",
                            color=t["text_sub"], fontsize=11,
                            transform=cal_ax.transAxes)
    prog_bg   = mpatches.FancyBboxPatch((0.15, 0.35), 0.70, 0.06,
                    boxstyle="round,pad=0.01", linewidth=0,
                    facecolor=t["cal_prog_bg"],
                    transform=cal_ax.transAxes, zorder=2)
    prog_fill = mpatches.FancyBboxPatch((0.15, 0.35), 0.0, 0.06,
                    boxstyle="round,pad=0.01", linewidth=0,
                    facecolor="#7c4dff", transform=cal_ax.transAxes, zorder=3)
    cal_ax.add_patch(prog_bg)
    cal_ax.add_patch(prog_fill)
    cal_phase_done_shown: list[bool] = [False]

    def update(_frame: object):  # type: ignore[return]
        phase    = cal_phase[0]
        progress = cal_progress[0]

        if phase != CAL_DONE:
            cal_ax.set_visible(True)
            fig.suptitle("")
            if phase == CAL_EXTEND:
                cal_title.set_text("Step 1 / 2 — Open hand")
                cal_sub.set_text("Fully extend all fingers and hold still")
                prog_fill.set_facecolor("#7c4dff")
            else:
                cal_title.set_text("Step 2 / 2 — Make a fist")
                cal_sub.set_text("Curl all fingers into a fist and hold still")
                prog_fill.set_facecolor("#ff4d7c")
            prog_fill.set_width(0.70 * progress)
            return [*scatters, *bars]

        if not cal_phase_done_shown[0]:
            cal_ax.set_visible(False)
            fig.suptitle("Leap Motion → LTS Spike Raster",
                         color=t["text"], fontsize=13, y=0.98)
            bar_ax.set_ylim(0, max(max(cal_max) * 1.15, 200))
            for i, (lo_line, hi_line) in enumerate(range_lines):
                for line, y in ((lo_line, cal_min[i]), (hi_line, cal_max[i])):
                    line.set_ydata([y, y])
                    line.set_alpha(0.6)
            cal_phase_done_shown[0] = True

        now   = sim_time_ref[0]
        t_min = now - WINDOW_MS
        for i in range(N_CHANNELS):
            pts = [(t - t_min, i) for t, ch in spike_history if ch == i and t >= t_min]
            scatters[i].set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        raster_ax.set_xlim(0, WINDOW_MS)

        for i, bar in enumerate(bars):
            bar.set_height(bar_values[i])

        return [*scatters, *bars]

    start_background_threads()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)  # noqa: F841
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-udp", action="store_true", help="disable UDP spike sending")
    parser.add_argument("--udp-ip", default="10.97.150.1", help="UDP target IP (default: 10.97.150.1)")
    parser.add_argument("--light", action="store_true", help="use light color theme")
    args = parser.parse_args()

    if not args.no_udp:
        configure_udp(ip=args.udp_ip)

    listener   = FingertipListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        try:
            run_gui(theme="light" if args.light else "dark")
        finally:
            stop_event.set()


if __name__ == "__main__":
    main()
