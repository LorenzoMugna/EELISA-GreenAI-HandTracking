"""
Core logic for Leap Motion → LTS spike demo.

Shared state, Leap listener, and simulation thread live here.
The GUI in leap_lts_demo.py only reads from these.
Spike dispatch (e.g. UDP) is wired in via the on_spike callback
passed to start_background_threads().
"""

import collections
import math
import queue
import threading
import time
from typing import Callable

import numpy as np
import leap
from leap import datatypes as ldt
from leap.events import Event, TrackingEvent

import pathlib, sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import udp

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

MODE_TONIC  = "Tonic"
MODE_PHASIC = "Phasic"

# ── shared state ──────────────────────────────────────────────────────────────
spike_queue:   queue.Queue[tuple[float, int]]        = queue.Queue()
currents:      list[float]                           = [0.0] * N_CHANNELS
distances_mm:  list[float]                           = [0.0] * N_FINGERS
rotation_deg:  list[float]                           = [0.0]
bar_values:    list[float]                           = [0.0] * N_CHANNELS
stop_event:    threading.Event                       = threading.Event()
spike_history: collections.deque[tuple[float, int]] = collections.deque(maxlen=20_000)
sim_time_ref:  list[float]                           = [0.0]
enc_mode:      list[str]                             = [MODE_TONIC]

cal_min:      list[float] = [30.0]  * N_FINGERS
cal_max:      list[float] = [160.0] * N_FINGERS
cal_phase:    list[int]   = [CAL_EXTEND]
cal_progress: list[float] = [0.0]

use_udp: list[bool] = [False]


def configure_udp(ip: str) -> None:
    udp.configure(ip=ip)
    use_udp[0] = True


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
            if self._prev_dists is None or self._prev_rot is None:
                self._prev_dists = dists[:]
                self._prev_rot   = rot
                new_currents = [0.0] * N_CHANNELS
            else:
                deltas    = [dists[i] - self._prev_dists[i] for i in range(N_FINGERS)]
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

        phase   = cal_phase[0]
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


# ── spike dispatch thread ─────────────────────────────────────────────────────
def spike_dispatch_thread() -> None:
    """Drain spike_queue, send over UDP if enabled, append to spike_history."""
    while not stop_event.is_set():
        try:
            t, ch = spike_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if use_udp[0]:
            udp.send_spike(ch)
        spike_history.append((t, ch))


def start_background_threads() -> None:
    threading.Thread(target=simulation_thread,     daemon=True).start()
    threading.Thread(target=spike_dispatch_thread, daemon=True).start()
