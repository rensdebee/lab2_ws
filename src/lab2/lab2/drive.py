import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, PointStamped, Point
import math
from sensor_msgs.msg import Imu
import numpy as np
import cv2
import threading
from queue import Queue
from filterpy.kalman import KalmanFilter
import numpy as np
from lab2.utils import UTILS
from time import sleep


class AccumulateOdometry(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        # Send driving commands to robot
        self.should_move = True
        # Use markers to relocalize
        self.use_markers = True
        # Try to avoid obstacle
        self.detect_obstacles = True

        # Log position to terminal
        self.print_pos = True
        # Publish path for rviz2
        self.publish_path = True
        # Combine marker using kalman filter
        self.kalman_filter = True
        # Avoid obstacle for x amount of clock ticks
        self.obj_ticks = 10
        # If object x is below this number avoid by going to the right
        self.x_right = 640

        # Gain for driving
        self.angular_gain = 2.5
        self.linear_gain = 1.5  # 1

        # Max speeds
        self.max_angular = 2.0  # 2.0
        self.max_speed = 0.2  # 0.1

        # Starting coordinates and heading (yaw)
        self.start_x = 0
        self.start_y = -3.2
        self.start_yaw = math.radians(0)

        # List of targets to drive via
        self.targets = []

        # Add target [X, Y, error radius]
        # Avoid
        # self.targets.append([1, 3.4, 0.2])

        # # Middle
        # self.targets.append([0, 3.2, 0.1])

        # Second right
        # self.targets.append([1.1, 3.9, 0.1])

        # # Second left
        # self.targets.append([-1.1, 3.9, 0.1])

        # # First right
        self.targets.append([2, 2.85, 0.1])

        # # First left
        # self.targets.append([-2, 2.805, 0.1])

        # Target info
        self.target_x = self.targets[0][0]
        self.target_y = self.targets[0][1]
        self.error_radius = self.targets[0][2]
        self.distance_to_target = np.inf

        # List of recieved marker localizations
        self.localization_list = []

        # ROI for object avoidance
        self.roi = np.array(
            [[(1280, 670), (0, 670), (0, 800), (1280, 800)]],
            dtype=np.int32,
        )

        # Check if detected object in ROI
        self.obj_detected = 0
        self.obj_x = None

        # Thread for displaying images
        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

        # Utils to set led or screen
        self.utils = UTILS(self)

        # Timer to update move commands
        self.timer = self.create_timer(
            timer_period_sec=0.1, callback=self.timer_callback
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
        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        # Initial state (X, Y)
        self.kf.x = np.array([self.x, self.y])  # Starting at origin
        # State transition matrix (F) - constant position model
        self.kf.F = np.eye(2)
        # Observation matrix (H)
        self.kf.H = np.eye(2)
        # Process noise covariance (Q)
        self.kf.Q = np.eye(2) * 0.1  # Small process noise
        # State covariance matrix (P)
        self.kf.P = np.eye(2) * 1.0  # Initial uncertainty
        # Measurement uncertainty for odometry
        self.R_odom = np.eye(2) * 0.2  # Higher uncertainty for odometry
        # Measurement uncertainty for marker
        self.R_marker = np.eye(2) * 0.13  # Lower uncertainty for marker

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

        # Subscribe to object avoidance node
        self.subscription = self.create_subscription(
            Point, "/object", self.object_callback, 10
        )

        # Set leds
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
                self.utils.set_leds("#00FF00")
                self.stop_robot()
                self.utils.set_leds("#00FF00")
                sleep(3)
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
        if self.obj_detected > 0 and self.detect_obstacles:
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

    def object_callback(self, msg):
        # Get X centroid of object
        self.obj_x = msg.x
        # Check if needs to avoid object
        if bool(msg.y) and self.obj_detected == 0 and self.detect_obstacles:
            self.obj_detected = self.obj_ticks

    def avoid_obstacle(self):
        # Avoid the obstacle by moving forward and turning.
        if not self.detect_obstacles:
            return

        # Set leds to red
        self.utils.set_leds("#FF0000")

        twist = Twist()
        twist.linear.x = 0.1  # Move forward slowly
        twist.angular.z = 1.2  # Turn slightly

        # Determine turn left or right based on x coordinate of object
        if self.obj_x < self.x_right:
            twist.angular.z = -1 * twist.angular.z
        if self.should_move:
            self.move_publisher.publish(twist)
        if self.obj_detected > 0:
            self.obj_detected -= 1

    def angle_difference(self, target_angle, current_angle):
        # Computes shortets angle differnce between two targets
        diff = target_angle - current_angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def normalize_angle(self, angle):
        # Normalize between 0-360 degrees in radians
        return angle % (2 * math.pi)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def marker_callback(self, msg):
        # Update location based on new marker predicted location
        if not self.use_markers:
            return
        x = msg.point.x
        y = msg.point.y

        # Don't use estimate if more then 1.5m from current (estimated) location
        error = np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
        if error > 1.5:
            return

        # Combine using EKF
        self.kf.predict()
        self.kf.R = self.R_marker
        self.kf.update([x, y])
        self.x = self.kf.x.flatten()[0]
        self.y = self.kf.x.flatten()[1]

        # Update distance to target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        self.distance_to_target = distance_to_target

        # Update the path for visualization in RVIZ2
        if self.publish_path:
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

    def odometry_callback(self, msg):
        # Extract pose from odometry message
        position = msg.pose.pose.position

        # Correct position using yaw if we have two measurments
        if self.pos_init:
            self.correct_position(position.x, position.y)
        self.pos_init = True
        self.prev_x = position.x
        self.prev_y = position.y

        # Update distance to target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        self.distance_to_target = distance_to_target

        if self.print_pos:
            self.get_logger().info(
                f"Accumulated Position -> x: {self.x:.2f}, y: {self.y:.2f}, Heading: {math.degrees(self.yaw):.2f}, Target Heading: {math.degrees(self.target_angle):.2f}, Distance: {self.distance_to_target:.4f}, Uncertainty: {np.sqrt(np.diag(self.kf.P))}",
            )

        # Update the path for visualization in RVIZ2
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
        # Combine using EKF
        self.kf.predict()
        self.kf.R = self.R_odom
        self.kf.update(
            [
                self.x + (length * math.sin(self.yaw)),
                self.y + (length * math.cos(self.yaw)),
            ]
        )
        self.x = self.kf.x.flatten()[0]
        self.y = self.kf.x.flatten()[1]

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
    node = AccumulateOdometry()
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
