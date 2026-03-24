"""
Interactive GUI for the LTS neuron simulation.

Scrolling spike raster + per-channel current sliders.
Run with:  uv run src/lts_gui.py
"""

import collections
import queue
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from lts_neuron import A, B, C, D, DT, V_PEAK

N_CHANNELS  = 5
WINDOW_MS   = 500.0   # visible time in the raster (ms)
CURRENT_MIN = 0.0
CURRENT_MAX = 50.0
DEFAULT_CURRENT = 10.0
BATCH_MS    = 10.0    # simulation chunk size per thread iteration

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_CHANNELS))

# ── shared state ──────────────────────────────────────────────────────────────
spike_queue: queue.Queue[tuple[float, int]]       = queue.Queue()
currents: list[float]                             = [DEFAULT_CURRENT] * N_CHANNELS
stop_event: threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref: list[float]      = [0.0]   # updated every batch regardless of spikes


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
                v[i] += DT * (0.04 * v[i]**2 + 5.0 * v[i] + 140.0 - u[i] + currents[i])
                u[i] += DT * A * (B * v[i] - u[i])
                if v[i] >= V_PEAK:
                    v[i]  = C
                    u[i] += D
                    spike_queue.put((sim_time, i))
            sim_time += DT
        sim_time_ref[0] = sim_time
        # pace the thread to roughly real-time
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, BATCH_MS / 1000.0 - elapsed))


# ── GUI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    fig = plt.figure(figsize=(11, 7), facecolor="#12121f")

    # Raster axes
    raster_ax = fig.add_axes((0.08, 0.38, 0.88, 0.56))
    raster_ax.set_facecolor("#0a0a1a")
    raster_ax.set_xlim(0, WINDOW_MS)
    raster_ax.set_ylim(-0.5, N_CHANNELS - 0.5)
    raster_ax.set_xlabel("time  (ms)", color="white", labelpad=6)
    raster_ax.set_title("LTS Spike Raster — live", color="white", pad=10)
    raster_ax.tick_params(colors="white")
    raster_ax.set_yticks(range(N_CHANNELS))
    raster_ax.set_yticklabels([f"ch {i}" for i in range(N_CHANNELS)], color="white")
    for spine in raster_ax.spines.values():
        spine.set_edgecolor("#333")
    # subtle horizontal grid lines per channel
    for i in range(N_CHANNELS):
        raster_ax.axhline(i, color="#222", linewidth=0.5, zorder=0)

    scatters = [
        raster_ax.scatter([], [], s=60, color=COLORS[i], marker="|", linewidths=2, zorder=3)
        for i in range(N_CHANNELS)
    ]

    # Sliders — stacked below the raster
    slider_h   = 0.028
    slider_gap = 0.008
    sliders    = []

    for i in range(N_CHANNELS):
        bottom = 0.26 - i * (slider_h + slider_gap)
        ax_s = fig.add_axes((0.12, bottom, 0.78, slider_h), facecolor="#1e1e32")
        s = Slider(
            ax_s, f"ch {i}",
            CURRENT_MIN, CURRENT_MAX,
            valinit=DEFAULT_CURRENT,
            color=COLORS[i],
            track_color="#2a2a3e",
        )
        s.label.set_color("white")
        s.label.set_fontsize(9)
        s.valtext.set_color("white")
        s.valtext.set_fontsize(9)

        def _cb(val, idx=i):
            currents[idx] = val

        s.on_changed(_cb)
        sliders.append(s)

    def update(_frame):
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
            if pts:
                scatters[i].set_offsets(np.array(pts))
            else:
                scatters[i].set_offsets(np.empty((0, 2)))

        raster_ax.set_xlim(0, WINDOW_MS)
        return scatters

    thread = threading.Thread(target=simulation_thread, daemon=True)
    thread.start()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841

    try:
        plt.show()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
