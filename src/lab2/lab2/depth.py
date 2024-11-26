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
from sensor_msgs.msg import Image
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
        depth_cameras = ["/rae/stereo_front/image_raw"]
        self.image = None
        for topic in depth_cameras:
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic_name=topic: self.image_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

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
        image = self.bridge.imgmsg_to_cv2(msg)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image

    def timer_callback(self):
        threshold_up = [462]
        threshold_down = [0]
        if type(self.image) != np.ndarray:
            return

        image = self.image.copy()
        # Create visualization image
        visualization = image.copy()
        visualization = (visualization / visualization.max()) * 255
        visualization = visualization.astype(np.uint8)
        visualization = cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR)
        # self.display_queue.put(("Stereo", image))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygon_points = np.array([self.roi], dtype=np.int32)
        cv2.fillPoly(mask, polygon_points, 255)
        # Create black pixel mask by checking each channel
        black_mask = np.logical_and(image > threshold_down[0], image < threshold_up[0])
        # Combine with polygon mask to only get black pixels inside polygon
        black_mask = black_mask & (mask > 0)

        # Count black pixels and calculate area
        black_pixel_count = np.sum(black_mask)
        polygon_area = np.sum(mask > 0)

        # Calculate percentage of black pixels within polygon
        black_percentage = (
            (black_pixel_count / polygon_area) * 100 if polygon_area > 0 else 0
        )
        detection = black_percentage > self.obj_pct
        # Highlight detected black pixels in the visualization
        visualization[black_mask] = [0, 0, 255]  # Red color

        cv2.polylines(
            visualization,
            polygon_points,
            True,
            (0, 255, 0) if detection == 0 else (0, 0, 255),
            2,
        )

        # Calculate centroid of black pixels
        centroid = None
        centroid_x = 1280
        if np.any(black_mask):
            y_coords, x_coords = np.where(black_mask)
            centroid_x = int(np.median(x_coords))
            centroid_y = int(np.median(y_coords))
            centroid = (centroid_x, centroid_y)

            # Draw centroid on visualization
            cv2.circle(visualization, centroid, 5, (0, 255, 255), -1)  # Yellow dot
            cv2.putText(
                visualization,
                f"({centroid_x}, {centroid_y})",
                (centroid_x + 10, centroid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        text = [
            f"Black_pixels: {black_pixel_count}",
            f"Black_percentage: {black_percentage}",
            f"Black_mean: {centroid}",
        ]
        text_x, text_y = (0, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)  # Green color for the text
        thickness = 2
        for i, line in enumerate(text):
            cv2.putText(
                visualization,
                line,
                (text_x, text_y + i * 25),
                font,
                font_scale,
                color,
                thickness,
            )

        # Display the depth image with ROI
        self.display_queue.put(("depth", visualization))

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
