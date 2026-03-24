"""
Leap Motion → LTS neuron spike raster.

Each of the 5 fingers maps to one LTS neuron.  The Euclidean distance from
the fingertip to the palm centre is normalised to a driving current:
  extended finger (far)  → high current → more spikes
  curled finger   (near) → low current  → fewer spikes

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
import numpy as np
import leap
from leap import datatypes as ldt
from leap.events import Event, TrackingEvent

from lts_neuron import B, C, DT, neuron_step

# ── constants ─────────────────────────────────────────────────────────────────
N_FINGERS    = 5
WINDOW_MS    = 500.0   # visible history in raster
BATCH_MS     = 10.0    # sim chunk per thread tick
DIST_MIN_MM  = 30.0    # fingertip distance at full curl  (mm)
DIST_MAX_MM  = 160.0   # fingertip distance fully extended (mm)
CURRENT_MAX  = 50.0    # μA at full extension
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
DECAY        = 0.80   # current multiplier per tracking frame when no hand visible

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_FINGERS))

# ── shared state ──────────────────────────────────────────────────────────────
spike_queue:  queue.Queue[tuple[float, int]]        = queue.Queue()
currents:     list[float]                           = [0.0] * N_FINGERS
distances_mm: list[float]                           = [0.0] * N_FINGERS
stop_event:   threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref: list[float]                           = [0.0]


# ── distance → current mapping ─────────────────────────────────────────────
def dist_to_current(dist_mm: float) -> float:
    t = (dist_mm - DIST_MIN_MM) / (DIST_MAX_MM - DIST_MIN_MM)
    return float(np.clip(t * CURRENT_MAX, 0.0, CURRENT_MAX))


def vec_dist(a: ldt.Vector, b: ldt.Vector) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


# ── Leap listener ──────────────────────────────────────────────────────────
class FingertipListener(leap.Listener):
    def on_tracking_event(self, event: Event) -> None:
        assert isinstance(event, TrackingEvent)
        if not event.hands:
            for i in range(N_FINGERS):
                distances_mm[i] *= DECAY
                currents[i] *= DECAY
            return
        # use the first detected hand
        hand = event.hands[0]
        palm_pos = hand.palm.position
        for i in range(N_FINGERS):
            tip = hand.digits[i].distal.next_joint
            distances_mm[i] = vec_dist(tip, palm_pos)
            currents[i] = dist_to_current(distances_mm[i])


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
def main() -> None:
    fig = plt.figure(figsize=(11, 7), facecolor="#12121f")
    fig.suptitle("Leap Motion → LTS Spike Raster", color="white", fontsize=13, y=0.98)

    # Raster axes (upper 55%)
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

    # Distance bar chart (lower 28%)
    bar_ax = fig.add_axes((0.08, 0.07, 0.88, 0.24))
    bar_ax.set_facecolor("#0a0a1a")
    bar_ax.set_xlim(-0.5, N_FINGERS - 0.5)
    bar_ax.set_ylim(0, DIST_MAX_MM * 1.1)
    bar_ax.set_xticks(range(N_FINGERS))
    bar_ax.set_xticklabels(FINGER_NAMES, color="white")
    bar_ax.set_ylabel("tip distance  (mm)", color="white", labelpad=6)
    bar_ax.tick_params(colors="white")
    bar_ax.axhline(DIST_MIN_MM, color="#555", linewidth=0.8, linestyle="--")
    bar_ax.axhline(DIST_MAX_MM, color="#555", linewidth=0.8, linestyle="--")
    for spine in bar_ax.spines.values():
        spine.set_edgecolor("#333")

    bars = bar_ax.bar(
        range(N_FINGERS),
        [0.0] * N_FINGERS,
        color=COLORS,
        width=0.6,
        zorder=3,
    )

    def update(_frame: object):  # type: ignore[return]
        # drain spike queue
        while not spike_queue.empty():
            try:
                t, ch = spike_queue.get_nowait()
                spike_history.append((t, ch))
            except queue.Empty:
                break

        # raster
        now   = sim_time_ref[0]
        t_min = now - WINDOW_MS
        for i in range(N_FINGERS):
            pts = [(t - t_min, i) for t, ch in spike_history if ch == i and t >= t_min]
            if pts:
                scatters[i].set_offsets(np.array(pts))
            else:
                scatters[i].set_offsets(np.empty((0, 2)))
        raster_ax.set_xlim(0, WINDOW_MS)

        # distance bars
        for bar, d in zip(bars, distances_mm):
            bar.set_height(d)

        return [*scatters, *bars]

    # Start simulation thread
    threading.Thread(target=simulation_thread, daemon=True).start()

    # Start Leap listener
    listener  = FingertipListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841

    try:
        with connection.open():
            plt.show()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
