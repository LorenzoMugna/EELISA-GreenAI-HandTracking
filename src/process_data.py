"""
Offline spike encoder for recorded hand-tracking data.

For each CSV in ./data:
  1. Compute per-file min/max for each feature (distances + rotation)
  2. Run the same LTS neuron simulation used in the live GUI
  3. Write a spike CSV alongside the source file

Input columns used:
  digit_0_distance .. digit_4_distance   (mm)
  palm_normal_y                           (used to derive rotation angle)

Output CSV columns:
  label, frame, sim_time_ms,
  spike_ch0 .. spike_ch5   (1 if the channel fired during this frame, else 0)

Run with:
  uv run src/process_data.py [data_dir]   (default: ./data)
"""

import csv
import math
import sys
from pathlib import Path

from encoding import (N_CHANNELS, N_FINGERS, features_to_currents,
                      rotation_deg as compute_rotation_deg)
from lts_neuron import B, C, DT, neuron_step

BATCH_MS = 10.0   # simulated ms per recorded frame (matches GUI)
DIST_COLS   = [f"digit_{i}_distance" for i in range(N_FINGERS)]
NORMAL_Y_COL = "palm_normal_y"
LABEL_COL    = "label"


def process_file(src: Path) -> None:
    # ── load ──────────────────────────────────────────────────────────────────
    with src.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    if not rows:
        print(f"  skipping {src.name} (empty)")
        return

    # ── compute per-file calibration (min/max) ────────────────────────────────
    distances_all = [[float(r[c]) for c in DIST_COLS] for r in rows]
    rotations_all = [compute_rotation_deg(float(r[NORMAL_Y_COL])) for r in rows]

    cal_min = [min(distances_all[f][i] for f in range(len(rows)))
               for i in range(N_FINGERS)]
    cal_max = [max(distances_all[f][i] for f in range(len(rows)))
               for i in range(N_FINGERS)]

    # ── simulate ───────────────────────────────────────────────────────────────
    v        = [C]     * N_CHANNELS
    u        = [B * C] * N_CHANNELS
    sim_time = 0.0
    steps    = int(BATCH_MS / DT)

    out_rows: list[dict[str, object]] = []
    for frame_idx, (row, dists, rot) in enumerate(
            zip(rows, distances_all, rotations_all)):
        currents = features_to_currents(dists, rot, cal_min, cal_max)

        spikes = [0] * N_CHANNELS
        for _ in range(steps):
            for i in range(N_CHANNELS):
                v[i], u[i], spiked = neuron_step(v[i], u[i], currents[i])
                if spiked:
                    spikes[i] = 1
            sim_time += DT

        out_row: dict[str, object] = {
            LABEL_COL: row[LABEL_COL],
            "frame": frame_idx,
            "sim_time_ms": round(sim_time, 3),
        }
        for i in range(N_CHANNELS):
            out_row[f"spike_ch{i}"] = spikes[i]
        out_rows.append(out_row)

    # ── write ──────────────────────────────────────────────────────────────────
    dst = src.with_stem(src.stem + "_spikes")
    fieldnames = [LABEL_COL, "frame", "sim_time_ms",
                  *[f"spike_ch{i}" for i in range(N_CHANNELS)]]
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(out_rows)

    total_spikes = sum(
        r[f"spike_ch{i}"] == 1  # type: ignore[comparison-overlap]
        for r in out_rows for i in range(N_CHANNELS)
    )
    print(f"  {src.name} → {dst.name}  "
          f"({len(rows)} frames, {total_spikes} spikes)")


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not data_dir.is_dir():
        print(f"directory not found: {data_dir}")
        sys.exit(1)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"no CSV files in {data_dir}")
        sys.exit(1)

    print(f"processing {len(csv_files)} file(s) in {data_dir}/")
    for p in csv_files:
        process_file(p)


if __name__ == "__main__":
    main()
