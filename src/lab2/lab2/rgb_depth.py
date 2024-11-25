import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
from queue import Queue
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from lab2.calibrate import undistort_from_saved_data


class Depth_avoidance(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")
        self.obj_pct = 2.75

        # ROI for object avoidance
        self.roi = np.array(
            [[(1280, 670), (0, 670), (0, 800), (1280, 800)]],
            dtype=np.int32,
        )

        # Thread for displaying images
        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        # Subscribe to depth camera
        depth_cameras = [
            "/rae/right/image_raw/compressed",
            "/rae/left/image_raw/compressed",
        ]
        self.depth_npz = [
            "./src/lab2/lab2/calibration_data_hd.npz",
            "./src/lab2/lab2/calibration_data_hd_left.npz",
        ]
        for topic in depth_cameras:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, topic_name=topic: self.image_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

        self.left_image = None
        self.right_image = None
        self.frame = 0

        self.publisher_ = self.create_publisher(Point, "/object", 10)

        self.bridge = CvBridge()

        # Timer to check for objects
        self.timer = self.create_timer(
            timer_period_sec=0.1, callback=self.timer_callback
        )

    def display_frames(self):
        # Thread to handle displaying frames
        while True:
            try:
                frame_id, frame = self.display_queue.get()
                if frame is None:
                    break
                cv2.imshow(f"Frame {frame_id}", frame)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().error(f"Error in display thread: {e}")

    def image_callback(self, msg, topic_name):
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        if "left" in topic_name:
            self.left_image = image
        elif "right" in topic_name:
            self.right_image = image

    def timer_callback(self):
        if type(self.left_image) != np.ndarray or type(self.right_image) != np.ndarray:
            return

        left_image = self.left_image.copy()
        right_image = self.right_image.copy()
        self.display_queue.put(("cam", left_image))
        if self.frame % 10 == 0:
            print("saving")
            cv2.imwrite(f"depth/left_{self.frame}.png", left_image)
            cv2.imwrite(f"depth/right_{self.frame}.png", right_image)

        self.frame += 1

    #     msg = Point()
    #     msg.x = float(centroid_x)
    #     msg.y = float(detection)
    #     msg.z = 0.0  # Not used
    #     self.publisher_.publish(msg)

    def stop_robot(self):
        # Publish zero velocities to stop the robot
        self.utils.set_leds("#00FF00")
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.move_publisher.publish(twist)
        self.utils.draw_text(f"Shut down")


def main(args=None):
    rclpy.init(args=args)
    node = Depth_avoidance()
    try:
        # Spin the node to call callback functions
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt caught in main loop.")
    finally:
        # Ensure node is properly destroyed and stopped on shutdown
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
