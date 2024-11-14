import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from lab2.utils import UTILS, undistort_from_saved_data
import math


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

        self.calibration_npz = "./src/lab2/lab2/calibration_data.npz"
        # self.calibration_npz = "src/lab2/lab2/5coeff_calibration_data.npz"

        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeMax = 5
        arucoParams.adaptiveThreshWinSizeStep = 1
        arucoParams.minMarkerPerimeterRate = 0.005
        arucoParams.minCornerDistanceRate = 0.005
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementMinAccuracy = 0.001
        arucoParams.cornerRefinementMaxIterations = 50

        self.location_dict = {
            "h11": {8: [[320, 197], 27.5], 16: [[320, -120], 23.5]},
            "h12": {},
            "seven": {},
        }

        # 3.2
        # -1,2
        # 1.97
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.h11_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)
        self.h11_ids = [0, 19, 8, 29, 59, 99, 79, 69, 18, 9, 39, 49, 89, 16]

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
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
            timer_period_sec=0.06, callback=self.timer_callback
        )

    def image_callback(self, data, topic_name):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        idx = self.cameras.index(topic_name)
        self.frames[idx] = current_frame

    def detect_makers(self, current_frame):

        found_ids = {}

        (corners, ids, _) = self.h11_detector.detectMarkers(current_frame)
        if ids is not None:
            found_ids["h11"] = self.filter_points(corners, ids, "h11", current_frame)
        else:
            found_ids["h11"] = {}

        (corners, ids, _) = self.seven_detector.detectMarkers(current_frame)
        if ids is not None:
            found_ids["seven"] = self.filter_points(
                corners, ids, "seven", current_frame
            )
        else:
            found_ids["seven"] = {}

        (corners, ids, _) = self.h12_detector.detectMarkers(current_frame)
        if ids is not None:
            found_ids["h12"] = self.filter_points(corners, ids, "h12", current_frame)
        else:
            found_ids["h12"] = {}

        return found_ids

    def filter_points(self, corners, ids, tag_family, current_frame=None):
        point_dict = {}
        if ids is not None:
            for c, id in zip(corners, ids):
                if id[0] in self.location_dict[tag_family] or self.no_id_filter:
                    if not self.no_id_filter:
                        point_dict[id[0]] = self.get_point_pose(
                            c, self.location_dict[tag_family][id[0]][1], current_frame
                        )
                    if current_frame is not None:
                        cv2.aruco.drawDetectedMarkers(
                            current_frame, np.array([c]), np.array([id])
                        )
        return point_dict

    def get_point_pose(self, corner, marker_size, current_frame=None):
        marker = self.get_marker_points(marker_size)
        data = np.load(self.calibration_npz)
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"][0]
        success, r_vec, t_vec = cv2.solvePnP(marker, corner, camera_matrix, dist_coeffs)
        pose_dict = {
            "corners": corner,
            "t_vec": t_vec,
            "r_vec": r_vec,
            "distance": math.sqrt(
                t_vec[0][0] ** 2 + t_vec[2][0] ** 2
            ),  # pythagoras between Z coordinate and x
        }
        if current_frame is not None:
            axis_length = 100
            cv2.drawFrameAxes(
                current_frame, camera_matrix, dist_coeffs, r_vec, t_vec, axis_length
            )

            tvec_text = (
                f"x:{t_vec[0][0]:.2f} , y:{t_vec[1][0]:.2f} z:{t_vec[2][0]:.2f} cm"
            )

            # Define the position for the text (top-left corner of the image)
            text_position = tuple(corner[0][0].ravel().astype(int))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green color for the text
            thickness = 2

            # Put the text on the image
            cv2.putText(
                current_frame,
                tvec_text,
                text_position,
                font,
                font_scale,
                color,
                thickness,
            )

        return pose_dict

    def get_marker_points(self, marker_size):
        half_size = marker_size / 2
        object_points = np.array(
            [
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0],
            ],
            dtype=np.float32,
        )
        return object_points

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

            return (x3, y3, x4, y4)

    def locate(self, point_dict):
        point_list = []
        for tag_family, points in point_dict.items():
            for id, point_info in points.items():
                if id in self.location_dict[tag_family]:
                    point_location = self.location_dict[tag_family][id][0]
                    distance = point_info["distance"]
                    point_list.append(point_location + [distance])
        if len(point_list) >= 2:
            x3, y3, x4, y4 = self.bilateration(point_list[:2])
            print(x3, y3, x4, y4)
            return x4, y4
        return None

    def timer_callback(self):
        if type(self.frames[0]) != np.ndarray:
            return
        image = self.frames[0].copy()
        # Detect points
        point_dict = self.detect_makers(image)
        # Triangulate
        loc = self.locate(point_dict)
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
