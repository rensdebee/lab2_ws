import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from lab2.utils import UTILS, undistort_from_saved_data


class LAB2(Node):
    def __init__(self):
        self.find_markers = True
        self.no_id_filter = False

        super().__init__("lab_2")
        self.utils = UTILS(self)
        self.br = CvBridge()
        self.cameras = [
            "/rae/right/image_raw",
            # "/rae/stereo_front/image_raw",
            # "/rae/left_back/image_raw",
            # "/rae/stereo_back/image_raw",
        ]
        self.frames = [None] * len(self.cameras)

        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeMax = 5
        arucoParams.adaptiveThreshWinSizeStep = 1
        arucoParams.minMarkerPerimeterRate = 0.005
        # arucoParams.maxMarkerPerimeterRate = 0.3
        arucoParams.minCornerDistanceRate = 0.005
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.h11_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)
        self.h11_ids = [0, 19, 8, 29, 59, 99, 79, 69, 18, 9, 39, 49, 89]

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
        self.seven_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)
        self.seven_ids = [37, 27, 7, 17]

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
        self.h12_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)
        self.h12_ids = [47, 57, 67, 77]

        self.markers = [None] * len(self.cameras)

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

        self.timer = self.create_timer(
            timer_period_sec=0.01, callback=self.timer_callback
        )

    def image_callback(self, data, topic_name):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        idx = self.cameras.index(topic_name)
        self.frames[idx] = current_frame

    def detect_makers(self, current_frame):

        (corners, ids, rejected) = self.h11_detector.detectMarkers(current_frame)

        if ids is not None:
            for c, id in zip(corners, ids):
                if id in self.h11_ids or self.no_id_filter:
                    cv2.aruco.drawDetectedMarkers(
                        current_frame, np.array([c]), np.array([id])
                    )

        (corners, ids, rejected) = self.seven_detector.detectMarkers(current_frame)
        if ids is not None:
            for c, id in zip(corners, ids):
                if id in self.seven_ids or self.no_id_filter:
                    cv2.aruco.drawDetectedMarkers(
                        current_frame, np.array([c]), np.array([id])
                    )

        (corners, ids, rejected) = self.h12_detector.detectMarkers(current_frame)
        if ids is not None:
            for c, id in zip(corners, ids):
                if id in self.h12_ids or self.no_id_filter:
                    cv2.aruco.drawDetectedMarkers(
                        current_frame, np.array([c]), np.array([id])
                    )

        return (corners, ids, rejected)

    def timer_callback(self):
        if type(self.frames[0]) != np.ndarray:
            return
        image = self.frames[0].copy()
        (corners, ids, rejected) = self.detect_makers(image)
        cv2.imshow("image", image)
        cv2.waitKey(1)

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
