"""
Leap Motion → LTS neuron spike raster.

Each of the 5 fingers maps to one LTS neuron.  The Euclidean distance from
the fingertip to the palm centre is normalised to a driving current:
  extended finger (far)  → high current → more spikes
  curled finger   (near) → low current  → fewer spikes

A two-step GUI calibration phase auto-detects the extended and fist poses
before the raster starts.

Run with:
    uv run src/leap_lts_demo.py
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

from lts_neuron import B, C, DT, neuron_step

# ── constants ─────────────────────────────────────────────────────────────────
N_FINGERS      = 5
WINDOW_MS      = 500.0   # visible history in raster (ms)
BATCH_MS       = 10.0    # sim chunk per thread tick (ms)
CURRENT_MAX    = 50.0    # μA at full extension
FINGER_NAMES   = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
DECAY          = 0.80    # current multiplier per tracking frame when no hand

# calibration detection
HOLD_S         = 0.8     # seconds the pose must be held stably
EXTEND_MIN_MM  = 90.0    # all fingers must exceed this to count as extended
FIST_MAX_MM    = 65.0    # all fingers must be below this to count as fist
STABLE_STD_MM  = 8.0     # max per-finger std-dev to consider pose stable

# calibration phases
CAL_EXTEND = 0
CAL_FIST   = 1
CAL_DONE   = 2

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_FINGERS))

# ── shared state ──────────────────────────────────────────────────────────────
spike_queue:   queue.Queue[tuple[float, int]]        = queue.Queue()
currents:      list[float]                           = [0.0] * N_FINGERS
distances_mm:  list[float]                           = [0.0] * N_FINGERS
stop_event:    threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref:  list[float]                           = [0.0]

# calibration — written by listener thread, read by GUI thread
cal_min:      list[float] = [30.0]  * N_FINGERS
cal_max:      list[float] = [160.0] * N_FINGERS
cal_phase:    list[int]   = [CAL_EXTEND]
cal_progress: list[float] = [0.0]   # 0.0 – 1.0 for current phase hold


# ── helpers ───────────────────────────────────────────────────────────────────
def vec_dist(a: ldt.Vector, b: ldt.Vector) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def dist_to_current(dist_mm: float, finger: int) -> float:
    lo, hi = cal_min[finger], cal_max[finger]
    t = (dist_mm - lo) / (hi - lo) if hi > lo else 0.0
    return float(np.clip(t * CURRENT_MAX, 0.0, CURRENT_MAX))


# ── Leap listener ──────────────────────────────────────────────────────────
class FingertipListener(leap.Listener):
    def __init__(self) -> None:
        # rolling buffer of (wall_time, [dist_f0..f4])
        self._buf: collections.deque[tuple[float, list[float]]] = collections.deque()
        self._stable_since: float | None = None

    def on_tracking_event(self, event: Event) -> None:
        assert isinstance(event, TrackingEvent)
        if not event.hands:
            for i in range(N_FINGERS):
                distances_mm[i] *= DECAY
                currents[i]     *= DECAY
            if cal_phase[0] != CAL_DONE:
                self._buf.clear()
                self._stable_since = None
                cal_progress[0] = 0.0
            return

        hand     = event.hands[0]
        palm_pos = hand.palm.position
        dists    = [
            vec_dist(hand.digits[i].distal.next_joint, palm_pos)
            for i in range(N_FINGERS)
        ]
        for i in range(N_FINGERS):
            distances_mm[i] = dists[i]

        if cal_phase[0] == CAL_DONE:
            for i in range(N_FINGERS):
                currents[i] = dist_to_current(dists[i], i)
        else:
            self._update_calibration(dists)

    def _update_calibration(self, dists: list[float]) -> None:
        now = time.perf_counter()
        self._buf.append((now, dists))

        # trim readings older than HOLD_S
        cutoff = now - HOLD_S
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 5:
            cal_progress[0] = 0.0
            return

        arr    = np.array([d for _, d in self._buf])   # (N, 5)
        means  = arr.mean(axis=0)
        stds   = arr.std(axis=0)
        stable = bool(np.all(stds < STABLE_STD_MM))

        phase = cal_phase[0]
        if phase == CAL_EXTEND:
            pose_ok = stable and bool(np.all(means > EXTEND_MIN_MM))
        else:
            pose_ok = stable and bool(np.all(means < FIST_MAX_MM))

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


# ── simulation thread ──────────────────────────────────────────────────────
def simulation_thread() -> None:
    v        = [C]     * N_FINGERS
    u        = [B * C] * N_FINGERS
    sim_time = 0.0
    steps    = int(BATCH_MS / DT)

    while not stop_event.is_set():
        t0 = time.perf_counter()
        for _ in range(steps):
            for i in range(N_FINGERS):
                v[i], u[i], spiked = neuron_step(v[i], u[i], currents[i])
                if spiked:
                    spike_queue.put((sim_time, i))
            sim_time += DT
        sim_time_ref[0] = sim_time
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, BATCH_MS / 1000.0 - elapsed))


# ── GUI ───────────────────────────────────────────────────────────────────────
def run_gui() -> None:
    fig = plt.figure(figsize=(11, 7), facecolor="#12121f")

    # ── raster ────────────────────────────────────────────────────────────────
    raster_ax = fig.add_axes((0.08, 0.38, 0.88, 0.54))
    raster_ax.set_facecolor("#0a0a1a")
    raster_ax.set_xlim(0, WINDOW_MS)
    raster_ax.set_ylim(-0.5, N_FINGERS - 0.5)
    raster_ax.set_xlabel("time  (ms)", color="white", labelpad=6)
    raster_ax.tick_params(colors="white")
    raster_ax.set_yticks(range(N_FINGERS))
    raster_ax.set_yticklabels(FINGER_NAMES, color="white")
    for spine in raster_ax.spines.values():
        spine.set_edgecolor("#333")
    for i in range(N_FINGERS):
        raster_ax.axhline(i, color="#222", linewidth=0.5, zorder=0)

    scatters = [
        raster_ax.scatter([], [], s=60, color=COLORS[i], marker="|", linewidths=2, zorder=3)
        for i in range(N_FINGERS)
    ]

    # ── distance bars ─────────────────────────────────────────────────────────
    bar_ax = fig.add_axes((0.08, 0.07, 0.88, 0.24))
    bar_ax.set_facecolor("#0a0a1a")
    bar_ax.set_xlim(-0.5, N_FINGERS - 0.5)
    bar_ax.set_ylim(0, 200)
    bar_ax.set_xticks(range(N_FINGERS))
    bar_ax.set_xticklabels(FINGER_NAMES, color="white")
    bar_ax.set_ylabel("tip distance  (mm)", color="white", labelpad=6)
    bar_ax.tick_params(colors="white")
    for spine in bar_ax.spines.values():
        spine.set_edgecolor("#333")

    bars = bar_ax.bar(range(N_FINGERS), [0.0] * N_FINGERS,
                      color=COLORS, width=0.6, zorder=3)
    # calibration range markers (updated after calibration)
    range_lines = [
        (bar_ax.plot([i - 0.3, i + 0.3], [0, 0],
                     color=COLORS[i], lw=1.2, ls="--", alpha=0.0)[0],
         bar_ax.plot([i - 0.3, i + 0.3], [0, 0],
                     color=COLORS[i], lw=1.2, ls="--", alpha=0.0)[0])
        for i in range(N_FINGERS)
    ]

    # ── calibration overlay ────────────────────────────────────────────────────
    cal_ax = fig.add_axes((0.08, 0.38, 0.88, 0.54))
    cal_ax.set_facecolor("#12121fee")
    cal_ax.set_xlim(0, 1)
    cal_ax.set_ylim(0, 1)
    cal_ax.set_axis_off()

    cal_title = cal_ax.text(
        0.5, 0.72, "", ha="center", va="center",
        color="white", fontsize=16, fontweight="bold",
        transform=cal_ax.transAxes,
    )
    cal_sub = cal_ax.text(
        0.5, 0.55, "", ha="center", va="center",
        color="#aaaacc", fontsize=11,
        transform=cal_ax.transAxes,
    )
    # progress bar background + fill
    prog_bg   = mpatches.FancyBboxPatch(
        (0.15, 0.35), 0.70, 0.06,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#2a2a3e", transform=cal_ax.transAxes, zorder=2,
    )
    prog_fill = mpatches.FancyBboxPatch(
        (0.15, 0.35), 0.0, 0.06,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#7c4dff", transform=cal_ax.transAxes, zorder=3,
    )
    cal_ax.add_patch(prog_bg)
    cal_ax.add_patch(prog_fill)
    cal_phase_done_shown: list[bool] = [False]

    def update(_frame: object):  # type: ignore[return]
        phase    = cal_phase[0]
        progress = cal_progress[0]

        # ── calibration overlay ──────────────────────────────────────────────
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

        # first frame after calibration done
        if not cal_phase_done_shown[0]:
            cal_ax.set_visible(False)
            fig.suptitle("Leap Motion → LTS Spike Raster",
                         color="white", fontsize=13, y=0.98)
            # update bar chart y-limit and range markers
            bar_ax.set_ylim(0, max(cal_max) * 1.15)
            for i, (lo_line, hi_line) in enumerate(range_lines):
                for line, y in ((lo_line, cal_min[i]), (hi_line, cal_max[i])):
                    line.set_ydata([y, y])
                    line.set_alpha(0.6)
            cal_phase_done_shown[0] = True

        # ── drain spike queue ────────────────────────────────────────────────
        while not spike_queue.empty():
            try:
                t, ch = spike_queue.get_nowait()
                spike_history.append((t, ch))
            except queue.Empty:
                break

        now   = sim_time_ref[0]
        t_min = now - WINDOW_MS
        for i in range(N_FINGERS):
            pts = [(t - t_min, i) for t, ch in spike_history if ch == i and t >= t_min]
            scatters[i].set_offsets(np.array(pts) if pts else np.empty((0, 2)))
        raster_ax.set_xlim(0, WINDOW_MS)

        for bar, d in zip(bars, distances_mm):
            bar.set_height(d)

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
