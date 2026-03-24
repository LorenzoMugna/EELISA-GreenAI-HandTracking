"""
Offline spike encoder for recorded hand-tracking data.

For each CSV in ./data, outputs two spike files:
  <name>_spikes_tonic.csv  — current ∝ normalised feature value
  <name>_spikes_phasic.csv  — current ∝ |Δfeature| per frame

Input columns used:
  digit_0_distance .. digit_4_distance   (mm)
  palm_normal_y                           (used to derive rotation angle)

Output CSV columns:
  label, frame, sim_time_ms,
  spike_ch0 .. spike_ch5   (1 if the channel fired during this frame, else 0)

Run with:
  uv run src/spikes/process_data.py [data_dir]   (default: ./data)
"""

import csv
import sys
from pathlib import Path

from encoding import (N_CHANNELS, N_FINGERS,
                      tonic_features_to_currents,
                      phasic_features_to_currents,
                      rotation_deg as compute_rotation_deg)
from lts_neuron import B, C, DT, neuron_step

BATCH_MS     = 10.0
DIST_COLS    = [f"digit_{i}_distance" for i in range(N_FINGERS)]
NORMAL_Y_COL = "palm_normal_y"
LABEL_COL    = "label"
FIELDNAMES   = [LABEL_COL, "frame", "sim_time_ms",
                *[f"spike_ch{i}" for i in range(N_CHANNELS)]]


def simulate(currents_seq: list[list[float]]) -> list[list[int]]:
    """Run the LTS simulation over a sequence of current vectors.
    Returns a list of spike vectors (one per frame)."""
    v        = [C]     * N_CHANNELS
    u        = [B * C] * N_CHANNELS
    steps    = int(BATCH_MS / DT)
    results: list[list[int]] = []

    for currents in currents_seq:
        spikes = [0] * N_CHANNELS
        for _ in range(steps):
            for i in range(N_CHANNELS):
                v[i], u[i], spiked = neuron_step(v[i], u[i], currents[i])
                if spiked:
                    spikes[i] = 1
        results.append(spikes)
    return results


def write_spikes(src: Path, rows: list[dict[str, str]],
                 spikes_seq: list[list[int]], suffix: str) -> None:
    dst = src.with_stem(src.stem + suffix)
    out_rows: list[dict[str, object]] = []
    sim_time = 0.0
    for frame_idx, (row, spikes) in enumerate(zip(rows, spikes_seq)):
        sim_time += BATCH_MS
        out_row: dict[str, object] = {
            LABEL_COL: row[LABEL_COL],
            "frame": frame_idx,
            "sim_time_ms": round(sim_time, 3),
        }
        for i in range(N_CHANNELS):
            out_row[f"spike_ch{i}"] = spikes[i]
        out_rows.append(out_row)

    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(out_rows)

    total = sum(
        s == 1  # type: ignore[comparison-overlap]
        for r in out_rows for s in [r[f"spike_ch{i}"] for i in range(N_CHANNELS)]
    )
    print(f"    → {dst.name}  ({total} spikes)")


def process_file(src: Path) -> None:
    with src.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    if not rows:
        print(f"  skipping {src.name} (empty)")
        return

    print(f"  {src.name}  ({len(rows)} frames)")

    distances_all = [[float(r[c]) for c in DIST_COLS] for r in rows]
    rotations_all = [compute_rotation_deg(float(r[NORMAL_Y_COL])) for r in rows]

    # per-file calibration for tonic encoding
    cal_min = [min(distances_all[f][i] for f in range(len(rows)))
               for i in range(N_FINGERS)]
    cal_max = [max(distances_all[f][i] for f in range(len(rows)))
               for i in range(N_FINGERS)]

    # ── tonic currents ─────────────────────────────────────────────────────
    tonic_currents = [
        tonic_features_to_currents(dists, rot, cal_min, cal_max)
        for dists, rot in zip(distances_all, rotations_all)
    ]

    # ── phasic currents ─────────────────────────────────────────────────────
    phasic_currents: list[list[float]] = []
    for idx, (dists, rot) in enumerate(zip(distances_all, rotations_all)):
        if idx == 0:
            phasic_currents.append([0.0] * N_CHANNELS)
        else:
            deltas    = [dists[i] - distances_all[idx - 1][i] for i in range(N_FINGERS)]
            delta_rot = rot - rotations_all[idx - 1]
            phasic_currents.append(phasic_features_to_currents(deltas, delta_rot))

    # ── simulate & write ──────────────────────────────────────────────────────
    write_spikes(src, rows, simulate(tonic_currents),  "_spikes_tonic")
    write_spikes(src, rows, simulate(phasic_currents),  "_spikes_phasic")


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    if not data_dir.is_dir():
        print(f"directory not found: {data_dir}")
        sys.exit(1)

    csv_files = sorted(p for p in data_dir.glob("*.csv")
                       if "_spikes" not in p.stem)
    if not csv_files:
        print(f"no CSV files in {data_dir}")
        sys.exit(1)

    print(f"processing {len(csv_files)} file(s) in {data_dir}/")
    for p in csv_files:
        process_file(p)


if __name__ == "__main__":
    main()
