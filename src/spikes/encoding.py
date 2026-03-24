"""
Shared feature → current encoding for LTS neurons.

Used by both the live GUI (leap_lts_demo.py) and the offline
data processor (process_data.py).

Channel layout:
  0-4  Finger tip distances  (Thumb … Pinky)
  5    Palm rotation          arccos(-normal.y)  [0°, 180°]
"""

import math

N_FINGERS  = 5
N_CHANNELS = N_FINGERS + 1   # 6 total
CURRENT_MAX = 50.0            # μA


def rotation_deg(normal_y: float) -> float:
    """arccos(-normal_y) → [0°, 180°].  Palm-down = 0°, palm-up = 180°."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -normal_y))))


def dist_to_current(dist_mm: float, lo: float, hi: float) -> float:
    """Map a fingertip distance to a driving current using calibrated range."""
    t = (dist_mm - lo) / (hi - lo) if hi > lo else 0.0
    return max(0.0, min(CURRENT_MAX, t * CURRENT_MAX))


def rotation_to_current(rot_deg: float) -> float:
    """Map rotation angle [0°, 180°] to driving current."""
    return rot_deg / 180.0 * CURRENT_MAX


def features_to_currents(
    distances: list[float],
    rot_deg: float,
    cal_min: list[float],
    cal_max: list[float],
) -> list[float]:
    """Convert one frame of hand features to a current vector (length N_CHANNELS)."""
    currents = [dist_to_current(distances[i], cal_min[i], cal_max[i])
                for i in range(N_FINGERS)]
    currents.append(rotation_to_current(rot_deg))
    return currents
