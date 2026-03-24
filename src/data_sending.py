import udp
import numpy as np
import leap
import time
import argparse

def format_and_send(hand):

	digits_x = list(map(lambda x: x.distal.next_joint.x, hand.digits))
	digits_y = list(map(lambda x: x.distal.next_joint.y, hand.digits))
	digits_z = list(map(lambda x: x.distal.next_joint.z, hand.digits))
	digits = []
	for x, y, z in zip(digits_x, digits_y, digits_z):
		digits.extend([x, y, z])

	data = dict()
	#send relative xyz for each finger

	palm_pos = np.array([hand.palm.position.x,
					  hand.palm.position.y,
					  hand.palm.position.z])
	palm_normal = np.array([hand.palm.normal.x,
						 hand.palm.normal.y,
						 hand.palm.normal.z])
	digits = [float(x) for x in digits]
	digits_np = np.array(digits).reshape(5, 3)
	diffs = digits_np - palm_pos
	distances = [np.linalg.norm(diff) for diff in diffs]
	
	udp.send_coordinate("palm_normal", *palm_normal)

	for i, diff in enumerate(diffs):
		udp.send_coordinate(f"digit_{i}_diff", *diff)

	for i, dist in enumerate(distances):
		udp.send_value(f"digit_{i}_dist", dist)



class MyListener(leap.Listener):
	def __init__(self):
		pass

	def on_connection_event(self, event):
		print("Connected")

	def on_device_event(self, event):
		try:
			with event.device.open():
				info = event.device.get_info()
		except leap.LeapCannotOpenDeviceError:
			info = event.device.get_info()

		print(f"Found device {info.serial}")

	def init_file(self):
		with open(self.output, "w+") as f:
			if len(f.readlines()) == 0:
				print("Output file is empty, writing header.")
				f.write("label;palm_position_x;palm_position_y;palm_position_z;palm_normal_x;palm_normal_y;palm_normal_z;")
				for i in range(5):
					f.write(f"digit_{i}_distal_next_joint_x;digit_{i}_distal_next_joint_y;digit_{i}_distal_next_joint_z;")
				f.write("\n")

	def on_tracking_event(self, event):
		if event.tracking_frame_id % 20 != 0:
			return 
		if len(event.hands) != 1:
			print("Skipping frame because it does not contain exactly one hand.")
			return
		print("Sending")
		format_and_send(event.hands[0])


parser = argparse.ArgumentParser()
# parser.add_argument("--output", help="Output file for training data", default="training_data.csv")
# parser.add_argument("--label", help="Label to associate with the training data", default="0")
# parser.add_argument("--init", help="Whether to initialize the output file with a header", action="store_true")
def main():
	
	# args = parser.parse_args()
	# output_file = args.output
	# label_to_use = args.label


	# if args.init:
	# 	my_listener.init_file()

	my_listener = MyListener()
	connection = leap.Connection()
	connection.add_listener(my_listener)

	running = True

	with connection.open():
		connection.set_tracking_mode(leap.TrackingMode.Desktop)
		while running:
			time.sleep(1)


if __name__ == "__main__":
	main()
