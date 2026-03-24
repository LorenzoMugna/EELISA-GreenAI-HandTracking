"""
Interactive GUI for the LTS neuron simulation.

Scrolling spike raster + per-channel current sliders.
Two encoding modes (radio buttons):
  Absolute  — current = slider value directly
  Velocity  — current ∝ |Δslider| per animation frame

Run with:  uv run src/spikes/lts_gui_demo.py
"""

import collections
import queue
import threading
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from lts_neuron import B, C, DT, neuron_step
from encoding import velocity_to_current

N_CHANNELS      = 5
WINDOW_MS       = 500.0
CURRENT_MIN     = 0.0
CURRENT_MAX     = 50.0
DEFAULT_CURRENT = 10.0
BATCH_MS        = 10.0
VELOCITY_SCALE  = CURRENT_MAX / 2   # slider delta that maps to full current

MODE_ABSOLUTE = "Absolute"
MODE_VELOCITY = "Velocity"

COLORS = plt.colormaps["plasma"](np.linspace(0.2, 0.9, N_CHANNELS))

# ── shared state ───────────────────────────────────────────────────────────────
spike_queue:   queue.Queue[tuple[float, int]]        = queue.Queue()
currents:      list[float]                           = [DEFAULT_CURRENT] * N_CHANNELS
slider_vals:   list[float]                           = [DEFAULT_CURRENT] * N_CHANNELS
stop_event:    threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref:  list[float]                           = [0.0]
enc_mode:      list[str]                             = [MODE_ABSOLUTE]


# ── simulation thread ──────────────────────────────────────────────────────────
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


# ── GUI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    fig = plt.figure(figsize=(12, 7), facecolor="#12121f")

    # ── raster ────────────────────────────────────────────────────────────────
    raster_ax = fig.add_axes((0.08, 0.38, 0.78, 0.56))
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
    for i in range(N_CHANNELS):
        raster_ax.axhline(i, color="#222", linewidth=0.5, zorder=0)

    scatters = [
        raster_ax.scatter([], [], s=60, color=COLORS[i], marker="|", linewidths=2, zorder=3)
        for i in range(N_CHANNELS)
    ]

    # ── sliders ────────────────────────────────────────────────────────────────
    slider_h   = 0.028
    slider_gap = 0.008
    sliders: list[Slider] = []

    for i in range(N_CHANNELS):
        bottom = 0.26 - i * (slider_h + slider_gap)
        ax_s = fig.add_axes((0.12, bottom, 0.68, slider_h), facecolor="#1e1e32")
        s = Slider(ax_s, f"ch {i}", CURRENT_MIN, CURRENT_MAX,
                   valinit=DEFAULT_CURRENT, color=COLORS[i], track_color="#2a2a3e")
        s.label.set_color("white")
        s.label.set_fontsize(9)
        s.valtext.set_color("white")
        s.valtext.set_fontsize(9)

        def _cb(val: float, idx: int = i) -> None:
            slider_vals[idx] = val
            if enc_mode[0] == MODE_ABSOLUTE:
                currents[idx] = val

        s.on_changed(_cb)
        sliders.append(s)

    # ── encoding mode radio buttons ────────────────────────────────────────────
    radio_ax = fig.add_axes((0.87, 0.12, 0.12, 0.18), facecolor="#1e1e32")
    radio_ax.set_title("Encoding", color="white", fontsize=9, pad=4)
    radio = RadioButtons(
        radio_ax,
        labels=[MODE_ABSOLUTE, MODE_VELOCITY],
        active=0,
        label_props={"color": ["white", "white"], "fontsize": [10, 10]},
        radio_props={"facecolor": ["#7c4dff", "#ff4d7c"]},
    )

    def on_mode_change(label: str | None) -> None:
        if label is None:
            return
        enc_mode[0] = label
        if label == MODE_ABSOLUTE:
            for i in range(N_CHANNELS):
                currents[i] = slider_vals[i]

    radio.on_clicked(on_mode_change)

    # ── update loop ────────────────────────────────────────────────────────────
    prev_vals: list[float] = slider_vals[:]

    def update(_frame: object):  # type: ignore[return]
        # velocity mode: drive currents from |delta| since last frame
        if enc_mode[0] == MODE_VELOCITY:
            for i in range(N_CHANNELS):
                delta = slider_vals[i] - prev_vals[i]
                currents[i] = velocity_to_current(delta, VELOCITY_SCALE)
                prev_vals[i] = slider_vals[i]

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
        return list(scatters)

    threading.Thread(target=simulation_thread, daemon=True).start()

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)  # noqa: F841

    try:
        plt.show()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
