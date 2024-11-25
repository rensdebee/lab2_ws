import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt


def calibrate():
    """
    Calibrate according to Zhangs method
    """
    board_size = (9, 6)
    square_size = 40

    # Generate the 3d points corresponding to each corner of the chessboard
    points_3d = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    points_3d[:, :2] = (
        np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2) * square_size
    )

    # Store 3d and corresponding 2d points from all images
    object_points = []
    image_points = []

    images = glob.glob("./calibrationdata/*.png")

    for image in images:
        img = cv.imread(image)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Detect the "inner" corner points of the chessboard pattern
        ret, corners = cv.findChessboardCorners(img, board_size, None)

        # If all corners are found
        if ret:
            # Find more exact 2D corner positions (more exact than integer pixels).
            corners_refined = cv.cornerSubPix(
                img,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )

            # Save the 3D corner point coordinates
            object_points.append(points_3d)
            # Save the 2D corner points
            image_points.append(corners_refined)

    # Using the cv.CALIB_RATIONAL_MODEL flag allows us to model the lens distortion with additional coefficients (giving us a more flexible model
    # that can represent complex higher-oder distortions) which provides a more accurate correction for the sever distortion of the fisheye lens
    # this allows for an 8 parameter radial distortion model
    flags = cv.CALIB_RATIONAL_MODEL
    # Neither the CALIB_TILTED_MODEL flag -> allows the model to include the fact that the camera sensor might not be perfectly parallel to the lens plane
    # Or CALIB_THIN_PRISM_MODEL -> model asymmetrical lens distortion
    # seem to affect the undistorted image greatly

    # Perform camera calibration using openCV and the point correspondences
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, img.shape[::-1], None, None, flags=flags
    )

    # Print calibration results
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    print("\nRotation Vectors:\n", rvecs)
    print("\nTranslation Vectors:\n", tvecs)

    np.savez(
        "calibration_data.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )

    return camera_matrix, dist_coeffs


def undistort_img(camera_matrix, dist_coeffs, img):
    h, w = img.shape[:2]

    new_cam_mat, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h), centerPrincipalPoint=True
    )

    undist_img = cv.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mat)

    return undist_img


def undistort_from_saved_data(npz_file, img):
    data = np.load(npz_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"][0]
    return undistort_img(camera_matrix, dist_coeffs, img)


if __name__ == "__main__":
    cam_mat, dist_coeffs = calibrate()
    undistort_img(cam_mat, dist_coeffs[0])
    # undistort_from_saved_data("calibration_data.npz")

    # Camera intrinsic matrix and distortion coefficients from the ros algorithm.
    # cam_mat = np.array([[311.71773,   0.     , 307.39619],
    #                     [0.     , 313.31956, 191.23788],
    #                     [0.     ,   0.     ,   1.     ]])
    # dist_coeffs = np.array([-0.244207, 0.043980, 0.000955, -0.000878, 0.000000])
    # undistort_img(cam_mat, dist_coeffs[0])  
