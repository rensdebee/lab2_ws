import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
from queue import Queue
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
from sensor_msgs.msg import CompressedImage


class Color_avoidance(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        # ROI for object avoidance
        self.roi = np.array(
            [[(1280, 580), (600, 650), (0, 580), (0, 800), (1280, 800)]],
            dtype=np.int32,
        )

        # Thread for displaying images
        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        # Subscribe to RGB camera
        depth_cameras = ["/rae/right/image_raw/compressed"]
        for topic in depth_cameras:
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, topic_name=topic: self.depth_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

        # Publisher to signal if object detected
        self.publisher_ = self.create_publisher(Point, "/object", 10)

        self.bridge = CvBridge()

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

    def depth_callback(self, msg, topic_name):
        try:
            # Convert ROS Image to OpenCV format
            image = self.bridge.compressed_imgmsg_to_cv2(msg)

            # BGR thresholds upper and lower limit
            threshold_up = [90, 35, 60]
            threshold_down = [0, 0, 0]

            # Get binary mask of points inside ROI
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            polygon_points = np.array([self.roi], dtype=np.int32)
            cv2.fillPoly(mask, polygon_points, 255)

            # Create visualization image
            visualization = image.copy()

            # Create black pixel mask by checking each channel
            black_mask = np.logical_and.reduce(
                (
                    (image[:, :, 0] > threshold_down[0])
                    & (image[:, :, 0] < threshold_up[0]),  # B channel
                    (image[:, :, 1] > threshold_down[1])
                    & (image[:, :, 1] < threshold_up[1]),  # G channel
                    (image[:, :, 2] > threshold_down[2])
                    & (image[:, :, 2] < threshold_up[2]),  # R channel
                )
            )

            # Combine with polygon mask to only get black pixels inside polygon
            black_mask = black_mask & (mask > 0)
            binary_image = (black_mask * 255).astype("uint8")

            # Highlight detected black pixels in the visualization
            visualization[black_mask] = [0, 0, 255]  # Red color

            # Find contours
            contours, hierarchy = cv2.findContours(
                binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter contours based on area and aspect ratio and
            # Save closest object (has highest Y coordinate)
            highest_y = 0
            lowest_countour = None
            for contour in contours:
                # Contour properties including bbox
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                M = cv2.moments(contour)
                aspect_ratio = w / h if h != 0 else 0
                # Draw all contours
                cv2.drawContours(visualization, [contour], -1, (255, 0, 0), 2)
                if area > 5500 and aspect_ratio > 1.5 and M["m00"] != 0:
                    # Draw only bboxs if contour is likely rae
                    cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if y + h > highest_y:
                        highest_y = y + h
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        lowest_countour = [contour, (cx, cy), (x, y, w, h)]

            # Check if lowest contour is close enough such that we need to start avoiding
            detection = False
            centroid_x = 0
            if lowest_countour is not None:
                # Need to avoid
                if highest_y > 720:
                    detection = True
                # Draw white centroid dot on closest bbox
                cx, cy = lowest_countour[1]
                centroid_x = cx
                cv2.circle(visualization, (cx, cy), 5, (255, 255, 255), -1)

            # Visualize ROI
            cv2.polylines(
                visualization,
                polygon_points,
                True,
                (0, 255, 0) if detection == 0 else (0, 0, 255),
                2,
            )

            # Display the image
            self.display_queue.put((topic_name, visualization))

            # Publish if object needs to be avoided
            # Abuse/Treat Y as binary value indicating if need to avoid
            msg = Point()
            msg.x = float(centroid_x)
            msg.y = float(detection)
            msg.z = 0.0  # Not used
            self.publisher_.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Depth image processing failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = Color_avoidance()
    try:
        # Spin the node to call callback functions
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt caught in main loop.")
    finally:
        # Ensure node is properly destroyed and stopped on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
