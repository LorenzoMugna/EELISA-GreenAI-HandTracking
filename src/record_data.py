from ast import arg
import leap
import time
import argparse

def format(label, hand):
    digits_format_x = list(map(lambda x: str(x.distal.next_joint.x), hand.digits))
    digits_format_y = list(map(lambda x: str(x.distal.next_joint.y), hand.digits))
    digits_format_z = list(map(lambda x: str(x.distal.next_joint.z), hand.digits))
    digits_format = []
    for x, y, z in zip(digits_format_x, digits_format_y, digits_format_z):
        digits_format.extend([x, y, z])

    return ";".join([
        label,
        str(hand.palm.position.x),
        str(hand.palm.position.y),
        str(hand.palm.position.z),
        str(hand.palm.normal.x),
        str(hand.palm.normal.y),
        str(hand.palm.normal.z),
        ";".join(digits_format)
    ]) + "\n"

class MyListener(leap.Listener):
    label: str
    output: str
    def __init__(self, label, output):
        self.label = label
        self.output = output

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

        print(f"Frame {event.tracking_frame_id} with {len(event.hands)} hands.")

        with open(self.output, "a") as f:
            timestamp = time.time()
            for hand in event.hands:
                f.write(
                    format(self.label, hand)
                )


parser = argparse.ArgumentParser()
parser.add_argument("--output", help="Output file for training data", default="training_data.txt")
parser.add_argument("--label", help="Label to associate with the training data", default="0")
parser.add_argument("--init", help="Whether to initialize the output file with a header", action="store_true")
def main():
    
    args = parser.parse_args()
    output_file = args.output
    label_to_use = args.label

    my_listener = MyListener(label_to_use, output_file)
    if args.init:
        my_listener.init_file()

    connection = leap.Connection()
    connection.add_listener(my_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            time.sleep(1)


if __name__ == "__main__":
    main()
