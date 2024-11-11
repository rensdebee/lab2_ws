import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

from lab2.utils import UTILS, undistort_from_saved_data


class LAB2(Node):
    def __init__(self):
        self.calibrate_camera = True
        super().__init__("lab_2")
        self.utils = UTILS(self)
        self.br = CvBridge()
        self.cameras = [
            "/rae/right/image_raw",
            "/rae/stereo_front/image_raw",
            "/rae/left_back/image_raw",
            "/rae/stereo_back/image_raw",
        ]
        self.frames = [None] * len(self.cameras)
        for topic in self.cameras:
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic_name=topic: self.image_callback(msg, topic_name),
                10,
            )

        self.move_publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )
        return

    def image_callback(self, data, topic_name):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        idx = self.cameras.index(topic_name)
        # Undistort image using found calibration
        if self.calibrate_camera:
            self.frames[idx] = undistort_from_saved_data(
                "./src/lab2/lab2/calibration_data.npz", current_frame
            )
        else:
            self.frames[idx] = current_frame

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
