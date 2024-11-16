import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class AccumulateOdometry(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        self.should_move = True

        self.roi_polygon = np.array(
            [[(120, 380), (120, 275), (640 - 120, 275), (640 - 120, 380)]],
            dtype=np.int32,
        )
        self.min_clearance = 12  # Minimum allowed pixels for obstacles in the ROI
        self.obstacle_detected = False

        self.print_pos = False
        self.publish_path = True

        self.x = 0
        self.y = 0
        self.theta = 0

        self.target_x = 0.5
        self.target_y = 0
        self.error_radius = 0.05

        # Control gains
        self.linear_gain = 1.0  # Gain for linear velocity
        self.angular_gain = 5.0  # Gain for angular velocity
        self.max_speed = 0.1

        self.move_publisher = self.create_publisher(
            Twist, "/cmd_vel", rclpy.qos.qos_profile_system_default
        )

        self.timer = self.create_timer(
            timer_period_sec=0.06, callback=self.timer_callback
        )

        self.odom_subscriber = self.create_subscription(
            Odometry, "/diff_controller/odom", self.odometry_callback, 10
        )
        # /odometry/filtered
        # /diff_controller/odom

        self.create_subscription(
            Image,
            "/rae/stereo_front/image_raw",
            self.depth_callback,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.bridge = CvBridge()

        if self.publish_path:
            self.path_publisher = self.create_publisher(Path, "/accumulated_path", 50)

        self.path = Path()
        self.path.header.frame_id = "odom"
        self.first_pose_initialized = False  # Flag to track first pose initialization

        self.get_logger().info("Accumulated Odometry Node Initialized")

    def timer_callback(self):
        if self.obstacle_detected:
            self.avoid_obstacle()
            return
        # Compute the distance and angle to the target
        delta_x = self.target_x - self.x
        delta_y = self.target_y - self.y
        distance_to_target = math.sqrt(delta_x**2 + delta_y**2)
        target_angle = math.atan2(delta_y, delta_x)

        # Stop the robot if within the error radius
        if distance_to_target < self.error_radius:
            self.stop_robot()
            self.get_logger().info("Target reached!")
            exit()
            return

        # Compute control signals
        linear_velocity = self.linear_gain * distance_to_target
        angular_velocity = self.angular_gain * self.angle_difference(
            target_angle, self.theta
        )

        # Create and publish the Twist message
        twist = Twist()
        twist.linear.x = min(
            linear_velocity, self.max_speed
        )  # Limit max linear velocity
        twist.angular.z = angular_velocity
        print(twist.linear.x, twist.angular.z)
        if self.should_move:
            self.move_publisher.publish(twist)

    def depth_callback(self, msg):
        """Process the depth camera image and display it."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg)

            # Create a mask for the ROI
            mask = np.zeros_like(cv_image)
            cv2.fillPoly(mask, self.roi_polygon, 255)
            # Apply the mask to the depth image
            roi = cv2.bitwise_and(cv_image, mask)
            self.obstacle_detected = np.mean(roi) < self.min_clearance

            # Draw the ROI polygon on the depth image
            cv_image = cv_image
            cv_image = (cv_image / cv_image.max()) * 255
            cv_image = cv_image.astype(np.uint8)
            image_with_roi = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            cv2.polylines(
                image_with_roi,
                self.roi_polygon,
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )

            # Display the depth image with ROI
            cv2.imshow("Depth Image with ROI", image_with_roi)
            cv2.waitKey(1)  # Required to update the display

        except Exception as e:
            self.get_logger().error(f"Depth image processing failed: {e}")

    def avoid_obstacle(self):
        """Avoid the obstacle by moving forward and turning."""
        self.get_logger().warn("Obstacle detected! Avoiding...")

        twist = Twist()
        twist.linear.x = 0.05  # Move forward slowly
        twist.angular.z = 1.0  # Turn slightly

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

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def odometry_callback(self, msg):
        # Extract pose from odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        if not self.first_pose_initialized:
            # Initialize the first pose
            self.x_offset = position.x
            self.y_offset = position.y
            # Convert quaternion to Euler angles to get yaw (theta)
            _, _, self.theta_offset = euler_from_quaternion(
                [
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                ]
            )

            initial_pose = PoseStamped()
            initial_pose.header = msg.header
            initial_pose.header.frame_id = "odom"
            initial_pose.pose.position.x = 0.0
            initial_pose.pose.position.y = 0.0
            initial_pose.pose.position.z = 0.0
            initial_pose.pose.orientation.x = orientation.x
            initial_pose.pose.orientation.y = orientation.y
            initial_pose.pose.orientation.z = orientation.z
            initial_pose.pose.orientation.w = orientation.w

            self.path.poses.append(initial_pose)
            self.first_pose_initialized = True
            self.get_logger().info("Initialized first pose at (0, 0, 0)")
            return

        # Update the path
        new_pose = PoseStamped()
        new_pose.header = msg.header
        new_pose.header.frame_id = "odom"
        new_pose.pose.position.x = position.x - self.x_offset
        new_pose.pose.position.y = position.y - self.y_offset
        new_pose.pose.position.z = 0.0

        new_pose.pose.orientation.x = orientation.x
        new_pose.pose.orientation.y = orientation.y
        new_pose.pose.orientation.z = orientation.z
        new_pose.pose.orientation.w = orientation.w

        self.path.poses.append(new_pose)
        if self.publish_path:
            self.path_publisher.publish(self.path)

        # Convert quaternion to Euler angles to get yaw (theta)
        _, _, theta = euler_from_quaternion(
            [
                new_pose.pose.orientation.x,
                new_pose.pose.orientation.y,
                new_pose.pose.orientation.z,
                new_pose.pose.orientation.w,
            ]
        )
        self.theta = theta - self.theta_offset
        self.x = new_pose.pose.position.x
        self.y = new_pose.pose.position.y

        if self.print_pos:
            self.get_logger().info(
                f"Accumulated Position -> x: {self.x:.2f}, y: {self.y:.2f}, theta: {self.theta:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AccumulateOdometry()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
