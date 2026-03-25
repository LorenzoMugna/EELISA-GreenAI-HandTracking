import csv
import math
import numpy as np
from typing import List, Tuple


def _read_spike_file(path: str) -> List[np.ndarray]:
	"""Read CSV where each row is comma-separated timestamps in ms.

	Returns a list of 1D numpy arrays (dtype=float) with timestamps (ms) per channel.
	"""
	channels: List[np.ndarray] = []
	with open(path, "r", newline="") as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				channels.append(np.array([], dtype=float))
				continue
			# filter out empty strings
			vals = [float(x) for x in row if x != ""]
			channels.append(np.array(vals, dtype=float))
	return channels


def sliding_window_stats(
	spikes: np.ndarray,
	window_ms: float,
	step_ms: float,
	start_ms: float,
	end_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute sliding-window rate (Hz), mean ISI (ms) and ISI variance (ms^2).

	Returns (times_ms, rate_hz, mean_isi_ms, var_isi_ms) as numpy arrays of same length.
	"""
	if spikes.size == 0:
		times = np.arange(start_ms, end_ms - window_ms + 1e-9, step_ms)
		n = times.size
		return times, np.full(n, 0.0), np.full(n, np.nan)

	# ensure sorted
	spikes = np.sort(spikes)
	times = np.arange(start_ms+window_ms, end_ms , step_ms)
	rates = np.empty(times.size, dtype=float)
	mean_isis = np.empty(times.size, dtype=float)
	var_isis = np.empty(times.size, dtype=float)

	for i, t0 in enumerate(times):
		wstart = t0 - window_ms
		wend = t0
		lo = np.searchsorted(spikes, wstart, side="left")
		hi = np.searchsorted(spikes, wend, side="right")
		seg = spikes[lo:hi]
		count = seg.size
		# rate in Hz (spikes per second)
		rates[i] = count / (window_ms / 1000.0)

		if count < 2:
			mean_isis[i] = math.nan
			var_isis[i] = math.nan
		else:
			diffs = np.diff(seg)
			mean_isis[i] = float(diffs.mean())
			var_isis[i] = float(diffs.var())

	return times, rates, var_isis


def write_output_csv(
	label: str,
	out_path: str,
	times: np.ndarray,
	per_channel_stats: List[Tuple[np.ndarray, np.ndarray]],
	append: bool = False
):
	"""Write CSV with columns: time_ms, (for each channel) rate_hz, mean_isi_ms, var_isi_ms"""
	n_ch = len(per_channel_stats)
	header = ["label", "time_ms"]
	for ch in range(n_ch):
		header += [f"ch{ch}_rate_hz", f"ch{ch}_var_isi_ms"]

	if append:
		mode = "a"
	else:
		mode = "w"

	with open(out_path, mode, newline="") as f:
		writer = csv.writer(f)
		if not append:
			writer.writerow(header)
		for i, t in enumerate(times):
			row = [label, f"{t:.3f}"]
			for rates, vars in per_channel_stats:
				r = rates[i]
				v = vars[i]
				row += [f"{r:.6f}", ("NaN" if math.isnan(v) else f"{v:.6f}")]
			writer.writerow(row)


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Compute sliding-window spike stats from per-channel CSV")
	parser.add_argument("input", help="Input CSV with one row per channel containing comma-separated timestamps (ms)")
	parser.add_argument("--output", "-o", default="spike_stats.csv", help="Output CSV file")
	parser.add_argument("--window", "-w", type=float, default=500.0, help="Window size in ms (time-based)")
	parser.add_argument("--step", "-s", type=float, default=100.0, help="Step between windows in ms")
	parser.add_argument("--label", default=None, help="Label for the collected data (used in output filename)")
	args = parser.parse_args()

	channels = _read_spike_file(args.input)

	# determine global time bounds
	all_times = np.concatenate([ch for ch in channels if ch.size > 0]) if channels else np.array([])
	if all_times.size == 0:
		print("No spikes found in file; producing empty output")
		start_ms = 0.0
		end_ms = args.window
	else:
		start_ms = max([ch.min() for ch in channels if ch.size > 0])
		end_ms = min([ch.max() for ch in channels if ch.size > 0])

		if end_ms - start_ms < args.window:
			print("Warning: time range of spikes is smaller than window size; producing single window")
			end_ms = start_ms + args.window

	# align windows from start to end (inclusive of last possible window start)
	times = np.arange(start_ms+args.window, end_ms, args.step)
	if times.size == 0:
		times = np.array([start_ms])

	per_channel_stats = []
	for ch in channels:
		t, rates, vars = sliding_window_stats(ch, args.window, args.step, start_ms, end_ms)
		per_channel_stats.append((rates, vars))

	# Extract label from filename if not provided
	if args.label is not None:
		label = args.label
	else:
		import os
		base = os.path.basename(args.input)
		label = base.split(".")[0]  # filename without extension
		label = label.split("_")[-1]  # last part after underscore
	

	# Check if output file is empty
	import os
	out_exists = os.path.isfile(args.output)
	out_empty = (not out_exists) or os.path.getsize(args.output) == 0

	write_output_csv(label, args.output, t, per_channel_stats, append=not out_empty)
	print(f"Wrote sliding-window stats to {args.output}")


if __name__ == "__main__":
	main()
