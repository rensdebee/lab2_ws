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
from lab2.utils import UTILS


class AccumulateOdometry(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        # Send driving commands to robot
        self.should_move = True
        # Use markers to relocalize
        self.use_markers = True
        # Try to avoid obstacle
        self.detect_obstacles = False

        # Log position to terminal
        self.print_pos = True
        # Publish path for rviz2
        self.publish_path = True
        # Combine marker using kalman filter
        self.kalman_filter = True
        # Avoid obstacle for x amount of clock ticks
        self.obj_ticks = 13

        # Gain for driving
        self.angular_gain = 2.5
        self.linear_gain = 1  # 1

        # Max speeds
        self.max_angular = 2.0  # 2.0
        self.max_speed = 0.1  # 0.1

        # Starting coordinates and heading (yaw)
        self.start_x = 0
        self.start_y = 0
        self.start_yaw = math.radians(0)

        # List of targets to drive via
        self.targets = []

        # Add target [X, Y, error radius]
        # self.targets.append([-2, 2, 0.2])

        # # First right
        self.targets.append([2, 2.805, 0.2])

        # # First left
        # self.targets.append([-2, 2.805, 0.2])

        # Second right
        # self.targets.append([1.15, 3.9, 0.2])

        # # Second left
        # self.targets.append([-1.15, 3.9, 0.2])

        # # Middle
        # self.targets.append([0, 3.2, 0.2])

        # Target info
        self.target_x = self.targets[0][0]
        self.target_y = self.targets[0][1]
        self.error_radius = self.targets[0][2]
        self.distance_to_target = np.inf

        # List of recieved marker localizations
        self.localization_list = []

        # ROI for object avoidance
        self.roi_left = np.array(
            [[(0, 670), (700, 670), (700, 800), (0, 800)]],
            dtype=np.int32,
        )
        self.roi_right = np.array(
            [[(700, 670), (1280, 670), (1280, 800), (700, 800)]],
            dtype=np.int32,
        )
        self.roi_middle = np.array(
            [[(600, 670), (450, 670), (450, 800), (600, 800)]],
            dtype=np.int32,
        )

        # Check if detected object in ROI
        self.left_detected = 0
        self.right_detected = 0

        # Thread for displaying images
        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        # Utils to set led or screen
        self.utils = UTILS(self)

        # Timer to update move commands
        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

        # Publisher to send move commands to robot
        self.move_publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )

        # Subscribe to motor odometry
        self.odom_subscriber = self.create_subscription(
            Odometry, "/diff_controller/odom", self.odometry_callback, 10
        )

        # Init position
        self.x = self.start_x
        self.y = self.start_y
        self.pos_init = False
        self.target_angle = 0

        # Subscribe to depth camera
        # depth_cameras = ["/rae/stereo_back/image_raw"]
        # for topic in camdepth_cameraseras:
        #     self.create_subscription(
        #         Image,
        #         topic,
        #         lambda msg, topic_name=topic: self.depth_callback(msg, topic_name),
        #         qos_profile=QoSProfile(
        #             depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        #         ),
        #     )
        # self.bridge = CvBridge()

        # Subscribe to maker localization
        self.create_subscription(PointStamped, "/marker_loc", self.marker_callback, 10)

        # Subsctie to IMU
        self.create_subscription(Imu, "/rae/imu/data", self.imu_callback, 10)
        # Init heading (yaw)
        self.imu_offset_initialized = False
        self.yaw = self.start_yaw

        # Path publisher
        if self.publish_path:
            self.path_publisher = self.create_publisher(Path, "/accumulated_path", 50)

        self.path = Path()
        self.path.header.frame_id = "odom"

        self.utils.set_leds("#FF00FF")

        self.get_logger().info("Accumulated Odometry Node Initialized")

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

    def timer_callback(self):
        # Compute the distance to the target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        self.distance_to_target = distance_to_target

        # Switch to next target if within error radius
        if distance_to_target < self.error_radius:
            self.utils.set_leds("#FFFF00")
            self.get_logger().info("Target reached!")
            self.targets.pop(0)
            # If no more targets left stop robot
            if len(self.targets) == 0:
                self.stop_robot()
                exit()
                return
            # else set new target
            self.target_x = self.targets[0][0]
            self.target_y = self.targets[0][1]
            self.error_radius = self.targets[0][2]

        # Compute the distance to the possible new target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        self.distance_to_target = distance_to_target

        # Avoid obstacles
        if (
            self.left_detected > 0 or self.right_detected > 0
        ) and self.detect_obstacles:
            self.avoid_obstacle()
            return
        self.utils.set_leds("#FF00FF")

        # Compute the angle difference
        target_angle = math.atan2(delta_x, delta_y) % (2 * math.pi)
        angle_diff = target_angle - self.yaw
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
        self.target_angle = target_angle

        # Set velocities tuned by the gain
        angular_velocity = -1 * self.angular_gain * angle_diff
        linear_velocity = self.linear_gain * distance_to_target

        # Create the Twist message
        twist = Twist()
        twist.linear.x = min(linear_velocity, self.max_speed)
        if self.distance_to_target > 2.5:
            max_angular = 0.75
        else:
            max_angular = self.max_angular
        twist.angular.z = max(-1 * max_angular, min(angular_velocity, max_angular))

        # Send message
        if self.should_move:
            self.move_publisher.publish(twist)

    def depth_callback(self, msg, topic_name):
        if self.right_detected > 0 or self.left_detected > 0:
            return
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            mask = np.zeros_like(cv_image)
            cv2.fillPoly(mask, self.roi_left, 255)
            left_roi = cv2.bitwise_and(cv_image, mask)

            mask = np.zeros_like(cv_image)
            cv2.fillPoly(mask, self.roi_right, 255)
            right_roi = cv2.bitwise_and(cv_image, mask)

            mask = np.zeros_like(cv_image)
            cv2.fillPoly(mask, self.roi_middle, 255)
            middle_roi = cv2.bitwise_and(cv_image, mask)

            print(np.mean(left_roi), np.mean(middle_roi), np.mean(right_roi))
            if "front" in topic_name and (
                np.mean(left_roi) < 5.4 or np.mean(middle_roi) < 5.4
            ):
                self.left_detected = self.obj_ticks
            if "front" in topic_name and np.mean(right_roi) < 3.9:
                self.right_detected = self.obj_ticks

            # Draw the ROI polygon on the depth image
            object_detected = False
            cv_image = cv_image
            cv_image = (cv_image / cv_image.max()) * 255
            cv_image = cv_image.astype(np.uint8)
            image_with_roi = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(
                image_with_roi,
                self.roi_left,
                isClosed=True,
                color=(0, 0, 255) if self.left_detected > 0 else (255, 0, 0),
                thickness=2,
            )
            cv2.polylines(
                image_with_roi,
                self.roi_right,
                isClosed=True,
                color=(0, 0, 255) if self.right_detected > 0 else (255, 0, 0),
                thickness=2,
            )
            cv2.polylines(
                image_with_roi,
                self.roi_middle,
                isClosed=True,
                color=(0, 0, 255) if np.mean(middle_roi) < 5 else (255, 0, 0),
                thickness=2,
            )
            self.distance_to_target
            text = f"Distance to target: {self.distance_to_target:.4f}m"

            # Define the position for the text (top-left corner of the image)
            text_position = (0, 40)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green color for the text
            thickness = 2

            # Put the text on the image
            cv2.putText(
                image_with_roi,
                text,
                text_position,
                font,
                font_scale,
                color,
                thickness,
            )

            # Display the depth image with ROI
            self.display_queue.put((topic_name, image_with_roi))

        except Exception as e:
            self.get_logger().error(f"Depth image processing failed: {e}")

    def avoid_obstacle(self):
        """Avoid the obstacle by moving forward and turning."""

        if not self.detect_obstacles:
            return
        # self.get_logger().warn("Obstacle detected! Avoiding...")

        twist = Twist()
        twist.linear.x = 0.1  # Move forward slowly
        twist.angular.z = 1.2  # Turn slightly
        if self.left_detected > 0:
            twist.angular.z = -1 * twist.angular.z
        # print(twist.angular.z, self.left_detected, self.right_detected)
        if self.should_move:
            self.utils.set_leds("#FF0000")
            self.move_publisher.publish(twist)

        if self.left_detected > 0:
            self.left_detected -= 1
        if self.right_detected > 0:
            self.right_detected -= 1

    def stop_robot(self):
        # Publish zero velocities to stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.move_publisher.publish(twist)
        self.utils.set_leds("#00FF00")
        rclpy.shutdown()

    def angle_difference(self, target_angle, current_angle):
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
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        if not self.use_markers or distance_to_target > 2.5:
            return
        x = msg.point.x
        y = msg.point.y

        if self.kalman_filter:
            ##apply kalman filter on the localization data
            if len(self.localization_list) == 0:
                self.localization_list.append(np.asarray([x, y]))
                x, y = x, y
            else:
                self.localization_list.append(np.asarray([x, y]))

                if len(self.localization_list) < 7:
                    measurements = np.array(self.localization_list)
                else:
                    measurements = np.array(self.localization_list[-7:])

                transition_matrices = [[1, 1], [0, 1]]
                # transition_matrices = np.identity(len(measurements))
                observation_matrices = [[1, 0], [0, 1]]
                kf = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                )

                kf = kf.em(measurements)
                (_, _) = kf.filter(measurements)
                (smoothed_state_means, smoothed_state_covariances) = kf.smooth(
                    measurements
                )

                x, y = smoothed_state_means[-1]

        # if np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) < 1:
        self.x = 0.5 * x + 0.5 * self.x
        self.y = 0.5 * y + 0.5 * self.y

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

        # Set yaw between 0 and 360
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

        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        self.distance_to_target = distance_to_target

        if self.print_pos:
            self.get_logger().info(
                f"Accumulated Position -> x: {self.x:.2f}, y: {self.y:.2f}, Heading: {math.degrees(self.yaw):.2f}, Target Heading: {math.degrees(self.target_angle):.2f}, Distance: {self.distance_to_target:.4f}",
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


def main(args=None):
    rclpy.init(args=args)
    node = AccumulateOdometry()
    try:
        # Spin the node to call callback functions
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt caught in main loop.")
    finally:
        # Ensure node is properly destroyed and stopped on shutdown
        node.destroy_node()
        node.stop_robot()


if __name__ == "__main__":
    main()
