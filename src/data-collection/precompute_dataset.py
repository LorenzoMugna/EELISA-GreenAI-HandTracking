import csv
import numpy as np

def parse_row(row: list[str])->str:
	label, palm_position_x, palm_position_y, palm_position_z, palm_normal_x, palm_normal_y, palm_normal_z, *digits = row
	palm_position_xf = float(palm_position_x)
	palm_position_yf = float(palm_position_y)
	palm_position_zf = float(palm_position_z)
	palm_pos = np.array([palm_position_xf, palm_position_yf, palm_position_zf])

	palm_normal_xf = float(palm_normal_x)
	palm_normal_yf = float(palm_normal_y)
	palm_normal_zf = float(palm_normal_z)
	palm_normal = np.array([palm_normal_xf, palm_normal_yf, palm_normal_zf])

	digits = [float(x) for x in digits]
	digits_np = np.array(digits).reshape(5, 3)
	diffs = digits_np - palm_pos
	distances = [np.linalg.norm(diff) for diff in diffs]
	
	diffs = diffs.reshape(-1)
	diffs = [str(x) for x in diffs]
	return ";".join([
		label,
		palm_normal_x,
		palm_normal_y,
		palm_normal_z,
		";".join(diffs),
		";".join([str(x) for x in distances])
		])

with open("../../data/parsed_data.csv", "w") as fw:
# Write header
	fw.write("label;palm_normal_x;palm_normal_y;palm_normal_z;")
	fw.write(";".join([f"digit_{i}_diff_x;digit_{i}_diff_y;digit_{i}_diff_z" for i in range(5)]) + ";")
	fw.write(";".join([f"digit_{i}_distance" for i in range(5)]) + "\n")

	with open("../../data/training_data.csv", "r") as fr:
		reader = csv.reader(fr, delimiter=";")
		head = reader.__next__() # skip header
		for row in reader:
			outrow = parse_row(row)
			fw.write(outrow + "\n")