import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from robot_localization.srv import SetPose
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist, PointStamped
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
from queue import Queue
from pykalman import KalmanFilter
import numpy as np
from numpy import ma


class AccumulateOdometry(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        self.should_move = True
        self.print_pos = True
        self.print_heading = False
        self.publish_path = True

        self.start_x = 0
        self.start_y = 0
        self.start_yaw = math.radians(0)

        self.error_radius = 0.2

        # # First right
        # self.target_x = 2
        # self.target_y = 2.805

        # # First left
        # self.target_x = -2
        # self.target_y = 2.805

        # Second right
        # self.target_x = 1.15
        # self.target_y = 3.9

        # # Second left
        self.target_x = -1.15
        self.target_y = 3.9

        # # Middle
        # self.target_x = 0
        # self.target_y = 3.2
        # self.error_radius = 0.05

        self.angular_gain = 3
        self.linear_gain = 1
        self.max_angular = 2.0
        self.max_speed = 0.1

        self.detect_obstacles = False

        self.localization_list = []
        self.kalman_filter = True

        self.roi_polygon = np.array(
            [[(0, 380), (0, 275), (640, 275), (640, 380)]],
            dtype=np.int32,
        )
        self.min_clearance = 25  # Minimum allowed pixels for obstacles in the ROI
        self.front_obstacle_detected = False
        self.back_obstacle_detected = False

        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        self.client = self.create_client(SetPose, "/set_pose")

        # Ensure the service is available before proceeding
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")

        self.reset_position()

        self.move_publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )

        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

        self.odom_subscriber = self.create_subscription(
            Odometry, "/diff_controller/odom", self.odometry_callback, 10
        )
        self.x = self.start_x
        self.y = self.start_y
        self.pos_init = False
        self.target_angle = 0
        # /odometry/filtered
        # /diff_controller/odom # Negtive X
        cameras = ["/rae/stereo_front/image_raw"]
        # "/rae/stereo_back/image_raw"

        for topic in cameras:
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic_name=topic: self.depth_callback(msg, topic_name),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )
        self.bridge = CvBridge()

        self.create_subscription(PointStamped, "/marker_loc", self.marker_callback, 10)

        self.create_subscription(Imu, "/rae/imu/data", self.imu_callback, 10)
        self.imu_offset_initialized = False
        self.yaw = self.start_yaw

        if self.publish_path:
            self.path_publisher = self.create_publisher(Path, "/accumulated_path", 50)

        self.path = Path()
        self.path.header.frame_id = "odom"

        self.get_logger().info("Accumulated Odometry Node Initialized")

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

    def timer_callback(self):
        # Compute the distance and angle to the target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)

        # Stop the robot if within the error radius
        if distance_to_target < self.error_radius:
            self.stop_robot()
            self.get_logger().info("Target reached!")
            exit()
            return

        if self.front_obstacle_detected and self.detect_obstacles:
            self.avoid_obstacle()
            return

        # Compute the angle difference
        target_angle = math.atan2(delta_x, delta_y) % (2 * math.pi)
        angle_diff = target_angle - self.yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        self.target_angle = target_angle

        angular_velocity = -1 * self.angular_gain * angle_diff
        linear_velocity = self.linear_gain * distance_to_target

        # Create and publish the Twist message
        twist = Twist()
        twist.linear.x = min(linear_velocity, self.max_speed)
        twist.angular.z = max(
            -1 * self.max_angular, min(angular_velocity, self.max_angular)
        )

        # if twist.angular.z < 0:
        #     print("Going right")
        # elif twist.angular.z > 0:
        #     print("Going left")
        if self.should_move:
            self.move_publisher.publish(twist)

    def depth_callback(self, msg, topic_name):
        """Process the depth camera image and display it."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            # Create a mask for the ROI
            mask = np.zeros_like(cv_image)
            cv2.fillPoly(mask, self.roi_polygon, 255)
            # Apply the mask to the depth image
            roi = cv2.bitwise_and(cv_image, mask)

            object_detected = np.mean(roi) < self.min_clearance
            if "back" in topic_name:
                # print(np.mean(roi))
                self.back_obstacle_detected = object_detected
            if "front" in topic_name:
                self.front_obstacle_detected = object_detected

            # Draw the ROI polygon on the depth image
            cv_image = cv_image
            cv_image = (cv_image / cv_image.max()) * 255
            cv_image = cv_image.astype(np.uint8)
            image_with_roi = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(
                image_with_roi,
                self.roi_polygon,
                isClosed=True,
                color=(0, 0, 255) if object_detected else (255, 0, 0),
                thickness=2,
            )

            # Display the depth image with ROI
            self.display_queue.put((topic_name, image_with_roi))

        except Exception as e:
            self.get_logger().error(f"Depth image processing failed: {e}")

    def avoid_obstacle(self):
        """Avoid the obstacle by moving forward and turning."""

        if not self.detect_obstacles:
            return
        self.get_logger().warn("Obstacle detected! Avoiding...")

        twist = Twist()
        twist.linear.x = 0.1  # Move forward slowly
        twist.angular.z = 2.5  # Turn slightly
        twist.angular.z = -1 * twist.angular.z

        if self.should_move:
            self.move_publisher.publish(twist)

    def stop_robot(self):
        # Publish zero velocities to stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.move_publisher.publish(twist)

    def angle_difference(self, target_angle, current_angle):
        """Compute the smallest angular difference between two angles."""
        diff = target_angle - current_angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def normalize_angle(self, angle):
        return angle % (2 * math.pi)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def marker_callback(self, msg):
        """
        x and y are location from localization
        self.x and self.y are the location from odometry

        """
        x = msg.point.x
        y = msg.point.y

        if self.kalman_filter:
            ##apply kalman filter on the localization data
            self.localization_list.append(np.asarray([x, y]))

            measurements = np.array(self.localization_list)

            transition_matrices = [[1, 1], 
                        [0, 1]]
            # transition_matrices = np.identity(len(measurements))
            observation_matrices = [[1, 0], 
                                    [0, 1]]
            kf = KalmanFilter(transition_matrices = transition_matrices, observation_matrices = observation_matrices)

            kf = kf.em(measurements)
            (_, _) = kf.filter(measurements)
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

            x, y = smoothed_state_means[-1]

        if np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) < 1:
            self.x = 0.2 * x + 0.8 * self.x
            self.y = 0.2 * y + 0.8 * self.y

            if self.publish_path:
                # Update the path
                new_pose = PoseStamped()
                new_pose.header = msg.header
                new_pose.header.frame_id = "odom"
                new_pose.pose.position.x = float(self.x)
                new_pose.pose.position.y = float(self.y)
                new_pose.pose.position.z = 0.0
                self.path.poses.append(new_pose)
                self.path_publisher.publish(self.path)

    def imu_callback(self, msg):
        # Extract quaternion from IMU data
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w

        # Convert quaternion to yaw (heading)
        _, _, yaw = euler_from_quaternion([qx, qy, qz, qw])
        yaw = -1 * yaw
        if not self.imu_offset_initialized:
            self.yaw_offset = yaw + self.start_yaw
            self.imu_offset_initialized = True

        self.yaw = self.normalize_angle(yaw - self.yaw_offset)

        if self.print_heading:
            self.get_logger().info(f"Heading: {math.degrees(self.yaw):.2f} degrees")

    def odometry_callback(self, msg):
        # Extract pose from odometry message
        position = msg.pose.pose.position
        rotation = msg.pose.pose.orientation

        _, _, yaw = euler_from_quaternion(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )
        if self.pos_init:
            self.correct_position(position.x, position.y)
        self.pos_init = True
        self.prev_x = position.x
        self.prev_y = position.y

        if self.print_pos:
            self.get_logger().info(
                f"Accumulated Position -> x: {self.x:.2f}, y: {self.y:.2f}, Heading: {math.degrees(self.yaw):.2f}, Target Heading: {math.degrees(self.target_angle):.2f}",
            )

        if self.publish_path:
            # Update the path
            new_pose = PoseStamped()
            new_pose.header = msg.header
            new_pose.header.frame_id = "odom"
            new_pose.pose.position.x = float(self.x)
            new_pose.pose.position.y = float(self.y)
            new_pose.pose.position.z = 0.0
            self.path.poses.append(new_pose)
            self.path_publisher.publish(self.path)

    def correct_position(self, x, y):
        length = np.sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)

        self.x += length * math.sin(self.yaw)
        self.y += length * math.cos(self.yaw)

    def reset_position(self):
        # Create a request to reset the pose
        self.request = SetPose.Request()

        # Initialize the pose message
        self.request.pose.header.stamp.sec = 0
        self.request.pose.header.stamp.nanosec = 0
        self.request.pose.header.frame_id = (
            "odom"  # Make sure to use the correct frame ID
        )

        # Corrected access to position (pose.pose.position)
        self.request.pose.pose.pose.position.x = float(self.start_x)
        self.request.pose.pose.pose.position.y = float(self.start_y)
        self.request.pose.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0, 0, 0)
        self.request.pose.pose.pose.orientation.x = qx
        self.request.pose.pose.pose.orientation.y = qy
        self.request.pose.pose.pose.orientation.z = qz
        self.request.pose.pose.pose.orientation.w = qw

        # Set the covariance to zero (can be adjusted if needed)
        self.request.pose.pose.covariance = [
            0.0
        ] * 36  # 6x6 covariance matrix, all zeros

        # Call the service with the prepared request
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Pose reset successful")
        else:
            self.get_logger().error("Failed to reset pose")


def main(args=None):
    rclpy.init(args=args)
    node = AccumulateOdometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
