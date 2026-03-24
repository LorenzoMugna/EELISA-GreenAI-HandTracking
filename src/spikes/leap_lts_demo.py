"""
Leap Motion → LTS neuron spike raster.

Two encoding modes (toggle with the radio buttons):
  Tonic  — current ∝ fingertip distance / palm rotation angle
  Phasic — current ∝ |Δdistance| or |Δangle| per frame (change-selective)

A two-step GUI calibration phase (tonic mode only) auto-detects the
extended and fist poses before the raster starts.

Run with:
    uv run src/spikes/leap_lts_demo.py
The Ultraleap service must be running.
"""

import collections
import math
import queue
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import leap
from leap import datatypes as ldt
from leap.events import Event, TrackingEvent
from matplotlib.widgets import RadioButtons

from lts_neuron import B, C, DT, neuron_step
from encoding import (N_FINGERS, N_CHANNELS,
                      tonic_features_to_currents,
                      phasic_features_to_currents,
                      rotation_deg as compute_rotation_deg)

# ── constants ─────────────────────────────────────────────────────────────────
WINDOW_MS      = 500.0
BATCH_MS       = 10.0
FINGER_NAMES   = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
CHANNEL_NAMES  = FINGER_NAMES + ["Rotation"]
DECAY          = 0.80

HOLD_S        = 0.8
EXTEND_MIN_MM = 90.0
FIST_MAX_MM   = 65.0
STABLE_STD_MM = 8.0

CAL_EXTEND = 0
CAL_FIST   = 1
CAL_DONE   = 2

MODE_TONIC = "Tonic"
MODE_PHASIC = "Phasic"

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_CHANNELS))

# ── shared state ──────────────────────────────────────────────────────────────
spike_queue:   queue.Queue[tuple[float, int]]        = queue.Queue()
currents:      list[float]                           = [0.0] * N_CHANNELS
distances_mm:  list[float]                           = [0.0] * N_FINGERS
rotation_deg:  list[float]                           = [0.0]
# bar chart shows raw input value per channel (mm for fingers, ° for rotation,
# or |delta| in velocity mode)
bar_values:    list[float]                           = [0.0] * N_CHANNELS
stop_event:    threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref:  list[float]                           = [0.0]
enc_mode:      list[str]                             = [MODE_TONIC]

cal_min:      list[float] = [30.0]  * N_FINGERS
cal_max:      list[float] = [160.0] * N_FINGERS
cal_phase:    list[int]   = [CAL_EXTEND]
cal_progress: list[float] = [0.0]


# ── helpers ───────────────────────────────────────────────────────────────────
def vec_dist(a: ldt.Vector, b: ldt.Vector) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


# ── Leap listener ─────────────────────────────────────────────────────────────
class FingertipListener(leap.Listener):
    def __init__(self) -> None:
        self._buf: collections.deque[tuple[float, list[float]]] = collections.deque()
        self._stable_since: float | None = None
        self._prev_dists: list[float] | None = None
        self._prev_rot: float | None = None

    def on_tracking_event(self, event: Event) -> None:
        assert isinstance(event, TrackingEvent)
        if not event.hands:
            for i in range(N_CHANNELS):
                currents[i] *= DECAY
                bar_values[i] *= DECAY
            for i in range(N_FINGERS):
                distances_mm[i] *= DECAY
            rotation_deg[0] *= DECAY
            self._prev_dists = None
            self._prev_rot   = None
            if cal_phase[0] != CAL_DONE:
                self._buf.clear()
                self._stable_since = None
                cal_progress[0] = 0.0
            return

        hand     = event.hands[0]
        palm_pos = hand.palm.position
        dists    = [vec_dist(hand.digits[i].distal.next_joint, palm_pos)
                    for i in range(N_FINGERS)]
        for i in range(N_FINGERS):
            distances_mm[i] = dists[i]

        normal = hand.palm.normal
        rot    = compute_rotation_deg(float(normal.y))
        rotation_deg[0] = rot

        if cal_phase[0] != CAL_DONE:
            self._update_calibration(dists)
            return

        mode = enc_mode[0]
        if mode == MODE_TONIC:
            new_currents = tonic_features_to_currents(dists, rot, cal_min, cal_max)
            for i in range(N_FINGERS):
                bar_values[i] = dists[i]
            bar_values[N_FINGERS] = rot
        else:
            # phasic encoding — need previous frame
            if self._prev_dists is None or self._prev_rot is None:
                self._prev_dists = dists[:]
                self._prev_rot   = rot
                new_currents = [0.0] * N_CHANNELS
            else:
                deltas   = [dists[i] - self._prev_dists[i] for i in range(N_FINGERS)]
                delta_rot = rot - self._prev_rot
                new_currents = phasic_features_to_currents(deltas, delta_rot)
                for i in range(N_FINGERS):
                    bar_values[i] = abs(deltas[i])
                bar_values[N_FINGERS] = abs(delta_rot)
            self._prev_dists = dists[:]
            self._prev_rot   = rot

        for i in range(N_CHANNELS):
            currents[i] = new_currents[i]

    def _update_calibration(self, dists: list[float]) -> None:
        now = time.perf_counter()
        self._buf.append((now, dists))
        cutoff = now - HOLD_S
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 5:
            cal_progress[0] = 0.0
            return

        arr    = np.array([d for _, d in self._buf])
        means  = arr.mean(axis=0)
        stds   = arr.std(axis=0)
        stable = bool(np.all(stds < STABLE_STD_MM))

        phase = cal_phase[0]
        pose_ok = (stable and bool(np.all(means > EXTEND_MIN_MM)) if phase == CAL_EXTEND
                   else stable and bool(np.all(means < FIST_MAX_MM)))

        if pose_ok:
            if self._stable_since is None:
                self._stable_since = now
            elapsed = now - self._stable_since
            cal_progress[0] = min(elapsed / HOLD_S, 1.0)
            if elapsed >= HOLD_S:
                if phase == CAL_EXTEND:
                    for i in range(N_FINGERS):
                        cal_max[i] = float(means[i])
                    cal_phase[0] = CAL_FIST
                else:
                    for i in range(N_FINGERS):
                        cal_min[i] = float(means[i])
                    cal_phase[0] = CAL_DONE
                self._buf.clear()
                self._stable_since = None
                cal_progress[0] = 0.0
        else:
            self._stable_since = None
            cal_progress[0] = 0.0


# ── simulation thread ─────────────────────────────────────────────────────────
def simulation_thread() -> None:
    v        = [C]     * N_CHANNELS
    u        = [B * C] * N_CHANNELS
    sim_time = 0.0
    steps    = int(BATCH_MS / DT)

    while not stop_event.is_set():
        t0 = time.perf_counter()
        for _ in range(steps):
            for i in range(N_CHANNELS):
                v[i], u[i], spiked = neuron_step(v[i], u[i], currents[i])
                if spiked:
                    spike_queue.put((sim_time, i))
            sim_time += DT
        sim_time_ref[0] = sim_time
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, BATCH_MS / 1000.0 - elapsed))


# ── GUI ───────────────────────────────────────────────────────────────────────
def run_gui() -> None:
    fig = plt.figure(figsize=(12, 7), facecolor="#12121f")

    # ── raster ────────────────────────────────────────────────────────────────
    raster_ax = fig.add_axes((0.08, 0.38, 0.78, 0.54))
    raster_ax.set_facecolor("#0a0a1a")
    raster_ax.set_xlim(0, WINDOW_MS)
    raster_ax.set_ylim(-0.5, N_CHANNELS - 0.5)
    raster_ax.set_xlabel("time  (ms)", color="white", labelpad=6)
    raster_ax.tick_params(colors="white")
    raster_ax.set_yticks(range(N_CHANNELS))
    raster_ax.set_yticklabels(CHANNEL_NAMES, color="white")
    for spine in raster_ax.spines.values():
        spine.set_edgecolor("#333")
    for i in range(N_CHANNELS):
        raster_ax.axhline(i, color="#222", linewidth=0.5, zorder=0)

    scatters = [
        raster_ax.scatter([], [], s=60, color=COLORS[i], marker="|", linewidths=2, zorder=3)
        for i in range(N_CHANNELS)
    ]

    # ── input bar chart ───────────────────────────────────────────────────────
    bar_ax = fig.add_axes((0.08, 0.07, 0.78, 0.24))
    bar_ax.set_facecolor("#0a0a1a")
    bar_ax.set_xlim(-0.5, N_CHANNELS - 0.5)
    bar_ax.set_ylim(0, 200)
    bar_ax.set_xticks(range(N_CHANNELS))
    bar_ax.set_xticklabels(CHANNEL_NAMES, color="white", fontsize=8)
    bar_ax.set_ylabel("input value", color="white", labelpad=6)
    bar_ax.tick_params(colors="white")
    for spine in bar_ax.spines.values():
        spine.set_edgecolor("#333")

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
    radio_ax = fig.add_axes((0.87, 0.12, 0.12, 0.18), facecolor="#1e1e32")
    radio_ax.set_title("Encoding", color="white", fontsize=9, pad=4)
    radio = RadioButtons(
        radio_ax,
        labels=[MODE_TONIC, MODE_PHASIC],
        active=0,
        label_props={"color": ["white", "white"], "fontsize": [10, 10]},
        radio_props={"facecolor": ["#7c4dff", "#ff4d7c"]},
    )

    bar_ylabel = bar_ax.yaxis.get_label()

    def on_mode_change(label: str | None) -> None:
        if label is None:
            return
        enc_mode[0] = label
        if label == MODE_TONIC:
            bar_ylabel.set_text("input value")
            bar_ax.set_ylim(0, max(max(cal_max) * 1.15, 200))
        else:
            bar_ylabel.set_text("|delta| per frame")
            bar_ax.set_ylim(0, 50)
        fig.canvas.draw_idle()

    radio.on_clicked(on_mode_change)

    # ── calibration overlay ───────────────────────────────────────────────────
    cal_ax = fig.add_axes((0.08, 0.38, 0.78, 0.54))
    cal_ax.set_facecolor("#12121fee")
    cal_ax.set_xlim(0, 1)
    cal_ax.set_ylim(0, 1)
    cal_ax.set_axis_off()

    cal_title = cal_ax.text(0.5, 0.72, "", ha="center", va="center",
                            color="white", fontsize=16, fontweight="bold",
                            transform=cal_ax.transAxes)
    cal_sub   = cal_ax.text(0.5, 0.55, "", ha="center", va="center",
                            color="#aaaacc", fontsize=11,
                            transform=cal_ax.transAxes)
    prog_bg   = mpatches.FancyBboxPatch((0.15, 0.35), 0.70, 0.06,
                    boxstyle="round,pad=0.01", linewidth=0,
                    facecolor="#2a2a3e", transform=cal_ax.transAxes, zorder=2)
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
                         color="white", fontsize=13, y=0.98)
            bar_ax.set_ylim(0, max(max(cal_max) * 1.15, 200))
            for i, (lo_line, hi_line) in enumerate(range_lines):
                for line, y in ((lo_line, cal_min[i]), (hi_line, cal_max[i])):
                    line.set_ydata([y, y])
                    line.set_alpha(0.6)
            cal_phase_done_shown[0] = True

        # drain spike queue
        while not spike_queue.empty():
            try:
                t, ch = spike_queue.get_nowait()
                spike_history.append((t, ch))
            except queue.Empty:
                break

        now   = sim_time_ref[0]
        t_min = now - WINDOW_MS
        for i in range(N_CHANNELS):
            pts = [(t - t_min, i) for t, ch in spike_history if ch == i and t >= t_min]
            scatters[i].set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        raster_ax.set_xlim(0, WINDOW_MS)

        for i, bar in enumerate(bars):
            bar.set_height(bar_values[i])

        return [*scatters, *bars]

    threading.Thread(target=simulation_thread, daemon=True).start()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841
    plt.show()


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    listener   = FingertipListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        try:
            run_gui()
        finally:
            stop_event.set()


if __name__ == "__main__":
    main()
