import cv2
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import math


def process_frame(frame, i, location_dict, calibration_npz):
    if not isinstance(frame, np.ndarray):
        return [], None, i

    point_list = []

    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.adaptiveThreshWinSizeMin = 3
    arucoParams.adaptiveThreshWinSizeMax = 4
    arucoParams.adaptiveThreshWinSizeStep = 1
    # arucoParams.adaptiveThreshConstant = 1 # This makes it hella slow
    arucoParams.minMarkerPerimeterRate = 0.00005
    arucoParams.minCornerDistanceRate = 0.00005
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    arucoParams.cornerRefinementMinAccuracy = 0.001
    arucoParams.cornerRefinementMaxIterations = 100

    h11_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), arucoParams
    )
    seven_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100), arucoParams
    )
    h12_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12), arucoParams
    )

    # Marker detection logic
    def detect_markers(detector, tag_family, ids, point_list):
        nonlocal frame
        (corners, ids_detected, _) = detector.detectMarkers(frame)
        if ids_detected is not None:
            for c, id in zip(corners, ids_detected):
                if id[0] in ids:
                    marker = get_marker_points(location_dict[tag_family][id[0]][1])
                    data = np.load(calibration_npz)
                    camera_matrix = data["camera_matrix"]
                    dist_coeffs = data["dist_coeffs"][0]
                    _, r_vec, t_vec = cv2.solvePnP(
                        marker, c, camera_matrix, dist_coeffs
                    )
                    axis_length = 100
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        r_vec,
                        t_vec,
                        axis_length,
                    )

                    tvec_text = f"x:{t_vec[0][0]:.2f} , y:{t_vec[1][0]:.2f} z:{t_vec[2][0]:.2f} cm"

                    # Define the position for the text (top-left corner of the image)
                    text_position = tuple(c[0][0].ravel().astype(int))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    color = (0, 255, 0)  # Green color for the text
                    thickness = 2

                    # Put the text on the image
                    cv2.putText(
                        frame,
                        tvec_text,
                        text_position,
                        font,
                        font_scale,
                        color,
                        thickness,
                    )
                    distance = math.sqrt(t_vec[0][0] ** 2 + t_vec[2][0] ** 2)
                    point_list.append(location_dict[tag_family][id[0]][0] + [distance])
                    cv2.aruco.drawDetectedMarkers(frame, np.array([c]), np.array([id]))

    # Detect for each tag family
    detect_markers(h11_detector, "h11", location_dict["h11"], point_list)
    detect_markers(seven_detector, "seven", location_dict["seven"], point_list)
    detect_markers(h12_detector, "h12", location_dict["h12"], point_list)

    return point_list, frame, i


def get_marker_points(marker_size):
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
