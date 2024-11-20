import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge
import cv2
import numpy as np
from lab2.utils import UTILS, undistort_from_saved_data
import math
from scipy.optimize import least_squares
from rclpy.qos import QoSProfile, ReliabilityPolicy
from lab2.measure import location_dict
import threading
from queue import Queue


class LAB2(Node):
    def __init__(self):
        self.find_markers = True
        self.check_in_field = False

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

        self.calibration_npzs = [
            "./src/lab2/lab2/calibration_data.npz",
            "./src/lab2/lab2/calibration_data_back.npz",
        ]
        #   self.calibration_npz = "src/lab2/lab2/5coeff_calibration_data.npz"

        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.adaptiveThreshWinSizeMin = 3
        arucoParams.adaptiveThreshWinSizeMax = 21
        arucoParams.adaptiveThreshWinSizeStep = 5
        arucoParams.polygonalApproxAccuracyRate = 0.04
        arucoParams.minCornerDistanceRate = 0.001
        arucoParams.perspectiveRemovePixelPerCell = 8
        arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.3

        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementMinAccuracy = 0.001
        arucoParams.cornerRefinementMaxIterations = 100

        self.location_dict = location_dict

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.h11_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
        self.seven_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
        self.h12_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

        for topic in self.cameras:
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic_name=topic: self.image_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

        self.publisher_ = self.create_publisher(Marker, "/marker_loc", 10)
        self.marker = Marker()
        self.marker.header.frame_id = "odom"
        self.marker.ns = "points"
        self.marker.type = Marker.POINTS
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.1  # Size of the point
        self.marker.scale.y = 0.1
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0  # Alpha (transparency)

    def image_callback(self, data, topic_name):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        idx = self.cameras.index(topic_name)
        self.frames[idx] = current_frame

        # self.display_queue.put((idx, current_frame))

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

    def detect_makers(self, cam_id, current_frame):
        point_list = []

        # Resize and sharpen the image
        scale_factor = 2
        current_frame = cv2.resize(
            current_frame,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        current_frame = cv2.filter2D(current_frame, -1, kernel)

        (corners, ids, _) = self.h11_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += self.filter_points(corners, ids, "h11", cam_id, current_frame)

        (corners, ids, _) = self.seven_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += self.filter_points(
                corners, ids, "seven", cam_id, current_frame
            )

        (corners, ids, _) = self.h12_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += self.filter_points(corners, ids, "h12", cam_id, current_frame)

        self.display_queue.put((cam_id, current_frame))
        return point_list

    def filter_points(self, corners, ids, tag_family, cam_id=0, current_frame=None):
        point_list = []
        if ids is not None:
            for c, id in zip(corners, ids):
                if id[0] in self.location_dict[tag_family]:
                    point_list.append(
                        self.get_point_pose(
                            c,
                            id[0],
                            tag_family,
                            cam_id,
                            current_frame,
                        )
                    )
                    if current_frame is not None:
                        cv2.aruco.drawDetectedMarkers(
                            current_frame, np.array([c]), np.array([id])
                        )
        return point_list

    def get_point_pose(self, corner, id, tag_family, cam_id=0, current_frame=None):
        marker = self.get_marker_points(self.location_dict[tag_family][id][1])
        data = np.load(self.calibration_npzs[cam_id])
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
            "distance_angle": (math.cos(r_vec[0][0]) * t_vec[2][0]),
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

        return self.location_dict[tag_family][id][0] + [pose_dict["distance"]]

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

        # non intersecting[INFO] [1732098052.885673243] [accumulate_odometry]: Accumulated Position -> x: -0.48, y: -0.94
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

            return [[x3, y3], [x4, y4]]

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

        return [loc]

    def locate(self, point_list):
        if len(point_list) < 2:
            loc = None
        if len(point_list) == 2:
            loc = self.bilateration(point_list[:2])
            print("Two points")
        if len(point_list) > 2:
            print("Three or more points")
            loc = self.trilateration(point_list)

        loc = self.check_field(loc)
        if loc is not None:
            print(loc)
            point = Point()
            point.x = loc[0]
            point.y = loc[1]
            point.z = 0.0
            self.marker.points.append(point)
            self.marker.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(self.marker)

        return loc

    def check_field(self, loc):
        if loc is None:
            return None
        if self.check_in_field == False:
            return loc[0]
        for xy in loc:
            x = xy[0]
            y = xy[1]
            if (-450 < x < 450) and (-300 < y < 300):
                return [x, y]
        return None

    def procces_frame(self, cam_id):
        point_list = []
        if type(self.frames[cam_id]) != np.ndarray:
            return point_list
        current_frame = self.frames[cam_id].copy()
        # Detect points
        point_list += self.detect_makers(cam_id, current_frame)

        return point_list

    def timer_callback(self):
        point_list = []
        for cam_id in [0]:
            point_list += self.procces_frame(cam_id)
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
