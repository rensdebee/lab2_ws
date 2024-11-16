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
import threading
from queue import Queue


class AccumulateOdometry(Node):
    def __init__(self):
        super().__init__("accumulate_odometry")

        self.should_move = True
        self.print_pos = False
        self.publish_path = True

        self.roi_polygon = np.array(
            [[(0, 380), (0, 275), (640, 275), (640, 380)]],
            dtype=np.int32,
        )
        self.min_clearance = 17  # Minimum allowed pixels for obstacles in the ROI
        self.front_obstacle_detected = False
        self.back_obstacle_detected = False

        self.x = 0
        self.y = 0
        self.theta = 0

        self.target_x = 0.75
        self.target_y = 0
        self.error_radius = 0.05

        # Control gains
        self.linear_gain = 1.0  # Gain for linear velocity
        self.angular_gain = 1.0  # Gain for angular velocity
        self.max_speed = 0.075
        self.max_angular = 1.5

        self.display_queue = Queue()
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()

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
        cameras = ["/rae/stereo_front/image_raw", "/rae/stereo_back/image_raw"]

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

        if self.publish_path:
            self.path_publisher = self.create_publisher(Path, "/accumulated_path", 50)

        self.path = Path()
        self.path.header.frame_id = "odom"
        self.first_pose_initialized = False  # Flag to track first pose initialization

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
        target_angle = math.atan2(delta_y, delta_x)

        # Stop the robot if within the error radius
        if distance_to_target < self.error_radius:
            self.stop_robot()
            self.get_logger().info("Target reached!")
            exit()
            return

        # Compute the angle difference
        angle_diff = self.angle_difference(target_angle, self.theta)

        # Determine whether to drive forward or backward
        if abs(angle_diff) > math.pi / 2:  # Target is behind the robot
            # Adjust angle for driving backward
            angle_diff = self.angle_difference(target_angle, self.theta + math.pi)
            linear_velocity = -self.linear_gain * distance_to_target  # Move backward
            if self.back_obstacle_detected:
                self.avoid_obstacle(back=True)
                return
        else:
            linear_velocity = self.linear_gain * distance_to_target  # Move forward
            if self.front_obstacle_detected:
                self.avoid_obstacle(front=True)
                return

        # Compute angular velocity
        angular_velocity = self.angular_gain * angle_diff

        # Create and publish the Twist message
        twist = Twist()
        twist.linear.x = max(
            -1 * self.max_speed, min(linear_velocity, self.max_speed)
        )  # Limit max linear velocity
        twist.angular.z = max(
            -1 * self.max_angular, min(angular_velocity, self.max_angular)
        )
        # print(twist.linear.x, twist.angular.z)
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
                print(np.mean(roi))
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

    def avoid_obstacle(self, front=False, back=False):
        """Avoid the obstacle by moving forward and turning."""
        self.get_logger().warn(
            "Obstacle detected! Avoiding..." + "front" if front else "back"
        )

        twist = Twist()
        twist.linear.x = 0.1  # Move forward slowly
        if back == True:
            twist.linear.x = -1 * twist.linear.x

        twist.angular.z = 2.5  # Turn slightly
        if front:
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
