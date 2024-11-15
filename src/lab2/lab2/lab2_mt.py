import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from lab2.utils import UTILS, undistort_from_saved_data
import math
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor
import threading
from queue import Queue
from lab2.detect_markers import process_frame
from rclpy.qos import QoSProfile, ReliabilityPolicy
from lab2.measure import location_dict


class LAB2(Node):
    def __init__(self):
        self.find_markers = True

        super().__init__("lab_2")
        self.utils = UTILS(self)

        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        self.br = CvBridge()
        self.cameras = [
            "/rae/right/image_raw",
            "/rae/left_back/image_raw",
            # "/rae/stereo_front/image_raw",
            # "/rae/stereo_back/image_raw",
        ]
        self.frames = [None] * len(self.cameras)

        self.calibration_npz = "./src/lab2/lab2/calibration_data.npz"
        #   self.calibration_npz = "src/lab2/lab2/5coeff_calibration_data.npz"

        self.location_dict = location_dict

        for topic in self.cameras:
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic_name=topic: self.image_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

        self.move_publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )

        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

    def image_callback(self, data, topic_name):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        idx = self.cameras.index(topic_name)
        self.frames[idx] = current_frame

    def display_frames(self):
        """Thread to handle displaying frames."""
        while True:
            try:
                frame_id, frame = self.display_queue.get()
                if frame is None:
                    break  # Exit the thread
                cv2.imshow(f"Frame {frame_id}", frame)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"Error in display thread: {e}")

    def bilateration(self, points):
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1
        x0, y0, r0 = points[0]
        x1, y1, r1 = points[1]

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # non intersecting
        if d > r0 + r1:
            return None
        # One circle within other
        if d < abs(r0 - r1):
            return None
        # coincident circles
        if d == 0 and r0 == r1:
            return None
        else:
            a = (r0**2 - r1**2 + d**2) / (2 * d)
            h = math.sqrt(r0**2 - a**2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d

            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d

            if y4 > 300:
                return [x3, y3]
            else:
                return [x4, y4]

    def trilateration(self, points):
        points = np.array(points)
        # Known positions of markers (x, y coordinates)
        markers = points[:, :2]  # Add as many markers as needed

        # Measured distances to each marker
        distances = points[:, 2]

        # Function to calculate residuals
        def residuals(position, markers, distances):
            return np.linalg.norm(markers - position, axis=1) - distances

        # Initial guess for the position
        initial_guess = np.mean(markers, axis=0)

        # Use least squares to minimize the residuals
        result = least_squares(residuals, initial_guess, args=(markers, distances))

        # Estimated position
        loc = result.x

        return loc

    def locate(self, point_list):
        if len(point_list) < 2:
            loc = None
        if len(point_list) == 2:
            loc = self.bilateration(point_list[:2])
            print("Two points")
        if len(point_list) > 2:
            print("Three or more points")
            loc = self.trilateration(point_list)
        if loc is not None:
            print(loc)
        return loc

    def timer_callback(self):
        frame_ids = [0, 1]  # Front and back camera

        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit processing tasks with necessary arguments
            futures = [
                executor.submit(
                    process_frame,
                    self.frames[i],
                    i,
                    self.location_dict,
                    self.calibration_npz,
                )
                for i in frame_ids
            ]

            # Combine results after tasks complete
            point_list = []
            for future in futures:
                try:
                    result, frame, frame_id = (
                        future.result()
                    )  # Get processed points and frame
                    point_list.extend(result)
                    if frame is not None:
                        # Display the frame here
                        self.display_queue.put((frame_id, frame))
                except Exception as e:
                    self.get_logger().error(f"Error processing frame: {e}")

        if not point_list:
            return

        # Triangulate
        loc = self.locate(point_list)

    def stop(self, signum=None, frame=None):
        self.utils.set_leds("#ce10e3")
        self.move_publisher.publish(Twist())
        self.utils.draw_text(f"Shut down")
        self.destroy_node()
        rclpy.shutdown()
        exit()


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    lab2 = LAB2()

    try:
        # Spin the node to call callback functions
        rclpy.spin(lab2)
    except KeyboardInterrupt:
        lab2.get_logger().info("Keyboard interrupt caught in main loop.")
    finally:
        # Ensure node is properly destroyed and stopped on shutdown
        lab2.destroy_node()
        lab2.stop()


if __name__ == "__main__":
    main()
