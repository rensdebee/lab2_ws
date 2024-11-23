import cv2
from measure import location_dict
from bob import dist_dict
import numpy as np
import math
import glob

calibration_npzs = [
    "./calibration_data_hd.npz",
    "./calibration_data.npz",
    "./calibration_data_back.npz",
]

cam_id = 0
scale_factor = 2

arucoParams = cv2.aruco.DetectorParameters()

# DANTE: Here both the step size and the min-max range affect the speed at which a frame is processed.
# Higher max, minimal min=3, and a step of 1 is almost always better for detection performance as more thresholding windows 
# are applied that way, but there is a trade-off here between detection performance and speed.
# Futhermore, going beyound a certain Max size doesn't make much sense as applying adaptive thresholding over to large of 
# an area, won't lead to any benefits.
arucoParams.adaptiveThreshWinSizeMin = 3
arucoParams.adaptiveThreshWinSizeMax = 21
# Speed up -> 1 is best but maybe dif between 3 and 4 not that big so go in steps of 2 /3 ?
arucoParams.adaptiveThreshWinSizeStep = 3
# arucoParams.adaptiveThreshConstant = 1 # This makes it hella slow
# arucoParams.minMarkerPerimeterRate = 0.001
# This is supposed to increase the max error to the approximated square shape. With a higher ratio increase the error in pixels that is allowed.
# Strangely the range of 0.05-0.1 seems te work best after which we lose detection accuracy again. not sure why tho...
# TODO maybe keep at 0.03 -> otherwise you get id=57 false positive, or we can remove this one from our filtered list ?
arucoParams.polygonalApproxAccuracyRate = 0.05
# Keep this 0.001 as it expresses the min distance between corners of detected markers
arucoParams.minCornerDistanceRate = 0.001
# Mininmal distance from a corner of a marker to the border of the image (to handle occlusions.) -> set to one pixel
# arucoParams.minDistanceToBorder = 1

#TODO BIT EXTRACTION
# arucoParams.maxErroneousBitsInBorderRate = 1
# TODO DEZE MOET PER FAMILY GEZET WORDEN
arucoParams.perspectiveRemovePixelPerCell = 8
# The amount of each grid that when reconstructing for the perspective that counts for the tag identification
# TODO THIS ONE IS REAALLY GOOOD
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.3


#TODO MARKER IDENTIFICATION
# allow more speling in the white pixels in the boarder (black of the marker)
# arucoParams.maxErroneousBitsInBorderRate = 0.5


arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
arucoParams.cornerRefinementMinAccuracy = 0.001
arucoParams.cornerRefinementMaxIterations = 100

# test
# cornerRefinementWinSize
# markerBorderBits
# maxErroneousBitsInBorderRate
# useAruco3Detection
# relativeCornerRefinmentWinSize

location_dict = location_dict
dist_dict = dist_dict

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
h11_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
seven_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
h12_detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

def detect_makers(cam_id, current_frame, dist_img=None):
        point_list = []

        (corners, ids, _) = h11_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += filter_points(corners, ids, "h11", cam_id, current_frame)

        (corners, ids, _) = seven_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += filter_points(
                corners, ids, "seven", cam_id, current_frame
            )

        (corners, ids, _) = h12_detector.detectMarkers(current_frame)
        if ids is not None:
            point_list += filter_points(corners, ids, "h12", cam_id, current_frame)

        # cv2.imshow("test", current_frame)
        # cv2.waitKey(0)   
        return point_list, current_frame

def filter_points(corners, ids, tag_family, cam_id=0, current_frame=None):
    point_list = []
    if ids is not None:
        for c, id in zip(corners, ids):
            if id[0] in location_dict[tag_family]:
                point_list.append(
                    get_point_pose(
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

def get_point_pose(corner, id, tag_family, cam_id=0, current_frame=None):
    marker = get_marker_points(location_dict[tag_family][id][1])

    # Rescale the corners
    # corner = corner / scale_factor

    data = np.load(calibration_npzs[cam_id])
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"][0]
    success, r_vec, t_vec = cv2.solvePnP(marker, corner, camera_matrix, dist_coeffs)

    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        marker, corner, camera_matrix, dist_coeffs, r_vec, t_vec
    )

    pose_dict = {
        "corners": corner,
        "t_vec": t_vec,
        "r_vec": r_vec,
        "distance": math.sqrt(
            t_vec[0][0] ** 2 + t_vec[2][0] ** 2
        ),  # pythagoras between Z coordinate and x
        "distance_angle": (math.cos(r_vec[0][0]) * t_vec[2][0]),
        "norm": np.linalg.norm(t_vec),
    }

    refined_pose_dict = {
        "corners": corner,
        "t_vec": tvec_refined,
        "r_vec": rvec_refined,
        "distance": math.sqrt(
            tvec_refined[0][0] ** 2 + tvec_refined[2][0] ** 2
        ),  # pythagoras between Z coordinate and x
        "distance_angle": (math.cos(rvec_refined[0][0]) * tvec_refined[2][0]),
    }

    # if current_frame is not None:
    #     axis_length = 100
        # cv2.drawFrameAxes(
        #     current_frame, camera_matrix, dist_coeffs, r_vec, t_vec, axis_length
        # )

        # tvec_text = (
        #     f"x:{t_vec[0][0]:.2f} , y:{t_vec[1][0]:.2f} z:{t_vec[2][0]:.2f} cm"
        # )

        # Define the position for the text (top-left corner of the image)
        # text_position = tuple(corner[0][0].ravel().astype(int))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.6
        # color = (0, 255, 0)  # Green color for the text
        # thickness = 2

        # # Put the text on the image
        # cv2.putText(
        #     current_frame,
        #     tvec_text,
        #     text_position,
        #     font,
        #     font_scale,
        #     color,
        #     thickness,
        # )

    # TODO REMOVE [id] +
    return [id] + location_dict[tag_family][id][0] + [pose_dict["distance"]] + [refined_pose_dict['distance']] + [pose_dict["norm"]]

def get_marker_points(marker_size):
    half_size = marker_size / 2
    object_points = np.array(
        [
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        dtype=np.float32,
    )
    return object_points

# def calc_dist(id, camera_matrix, corner, marker_width):
#     width_in_cm = marker_width

#     # Calculate pixel width of marker
#     x_min = np.max(corner[:,:,0])
#     x_max = np.min(corner[:,:,0])
#     x_width_pixels = abs(x_min - x_max)
#     y_min = np.max(corner[:,:,1])
#     y_max = np.min(corner[:,:,1])
#     y_height_pixels = abs(y_min - y_max)
#     width_pixels = (x_width_pixels + y_height_pixels) / 2

#     focal_length = camera_matrix[0,0]
    
#     distance = (width_in_cm * focal_length) / width_pixels
#     print(id, distance)


if __name__ == "__main__":
    num_imgs = {"a": 5, "b": 6, "c":7, "d": 4, "e":4}

    all_letters = ["a", "b", "c", "d", "e"]

    ############################## AAA
    # for letter in all_letters:
    #     target_letter = letter
        
    #     for i in range(1, num_imgs[target_letter]+1):
    #         print(f"{letter}{i}")
    #         image = cv2.imread(f"./dante_pictures/{target_letter}{i}.png")

    #         # Resize image
    #         image_size_2 = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    #         kernel = np.array([[0, -1, 0],
    #                [-1, 5, -1],
    #                [0, -1, 0]])
    #         sharpened = cv2.filter2D(image_size_2, -1, kernel)


    #         point_list = detect_makers(cam_id=cam_id, current_frame=sharpened)

    ############################## AAA

    ############################### BBB
    # target_letter = "e"
    # for i in range(1, num_imgs[target_letter]+1):
    #     print(f"{target_letter}{i}")
    #     image = cv2.imread(f"./dante_pictures/{target_letter}{i}.png")
    #     # Resize image
    #     image_size_2 = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    #     kernel = np.array([[0, -1, 0],
    #         [-1, 5, -1],
    #         [0, -1, 0]])
    #     sharpened = cv2.filter2D(image_size_2, -1, kernel)


    #     point_list = detect_makers(cam_id=cam_id, current_frame=sharpened)
    ############################### BBB

    ############################### CCC -> Distance detection stuffs
    # dist_images = ["dist1", "dist2"]

    # total_dist_err_list = []
    # total_norm_err_list =[]
    # total_2d_norm_err_list = []

    # for dist_img in dist_images:
    #     image = cv2.imread(f"./dante_pictures/{dist_img}.png")

    #     # Resize image
    #     image_size_2 = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    #     kernel = np.array([[0, -1, 0],
    #         [-1, 5, -1],
    #         [0, -1, 0]])
    #     sharpened = cv2.filter2D(image_size_2, -1, kernel)

    #     point_list, dist_list, norm_list, norm_2d_dist_list = detect_makers(cam_id=cam_id, current_frame=sharpened, dist_img=dist_img)
    #     total_dist_err_list += dist_list
    #     total_norm_err_list += norm_list
    #     total_2d_norm_err_list += norm_2d_dist_list


    # print(f"TOTAL AVERAGE DISTANCE ERROR = {np.mean(total_dist_err_list)}")
    # print(f"TOTAL AVERAGE NORM 3D ERROR = {np.mean(total_norm_err_list)}")
    # print(f"TOTAL AVERAGE NORM 2D (downproject) EROOR = {np.mean(norm_2d_dist_list)}")
    ############################### CCC


    ############################## DDD -> loop over images

    all_images = glob.glob("./report_images/position/*.png")

    for image in all_images:
        current_frame = cv2.imread(image)
        image_tag = image.rsplit("\\", 1)[1].split("_")
        true_robo_x = int(image_tag[0])
        true_robo_y = int(image_tag[1])
        
    #         # Resize image
    #         image_size_2 = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    #         kernel = np.array([[0, -1, 0],
    #                [-1, 5, -1],
    #                [0, -1, 0]])
    #         sharpened = cv2.filter2D(image_size_2, -1, kernel)

        point_list, current_frame = detect_makers(cam_id=cam_id, current_frame=current_frame)

        print("---------")
        for point in point_list:
            id = point[0]
            marker_x = point[1]
            marker_y = point[2]
            marker_z = point[3]

            est_dist_2d = point[4]
            est_dist_3d = point[6]
            est_dist_2d_down_proj = math.sqrt(est_dist_3d ** 2 - marker_z ** 2)

            true_dist_2d = math.sqrt(abs(marker_x - true_robo_x) ** 2 + abs(marker_y - true_robo_y) ** 2)
            print(true_robo_x, true_robo_y)
            print(marker_x, marker_y)
            true_dist_3d = math.sqrt(abs(marker_x - true_robo_x) ** 2 + abs(marker_y - true_robo_y) ** 2 + abs(marker_z + 0) **2)

            error_2d = abs(true_dist_2d - est_dist_2d)
            error_3d = abs(true_dist_3d - est_dist_3d)
            error_2d_down_proj = abs(true_dist_2d - est_dist_2d_down_proj)

            print("XXXXXX")
            print(f"""ID: {id}, true_dist_2d {true_dist_2d}, est_dist_2d: {est_dist_2d}, error_2d: {error_2d}
                  true_dist_3d: {true_dist_3d}, est_dist_3d: {est_dist_3d}, error_3d: {error_3d}
                    est_dist_2d_down_proj: {est_dist_2d_down_proj}, error_2d_down_proj: {error_2d_down_proj}""")
            print("XXXXXX")


        print("---------")

        cv2.imshow("test", current_frame)
        cv2.waitKey(0)   
        # exit()

    ############################## DDD


    #Proberen -> Verschillende parameters
    #Proberen -> Gaussian blur voor "lens flare"
    #Proberen -> Inzoomen voor smaller tags?






    