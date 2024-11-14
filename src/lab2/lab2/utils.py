from sensor_msgs.msg import BatteryState, Image
from std_msgs.msg import ColorRGBA
from rae_msgs.msg import LEDControl, ColorPeriod
from cv_bridge import CvBridge
from PIL import Image as PILImage, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import random


class UTILS:
    def __init__(self, node):
        self.node = node
        self.publisher = self.node.create_publisher(LEDControl, "/leds", 40)
        self.publisher_image = self.node.create_publisher(Image, "lcd", 10)
        self.publisher_led = self.node.create_publisher(LEDControl, "leds", 10)

        self.bridge = CvBridge()

    def random_hex_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def draw_text(self, text, size=25):
        # Define image size, bar, and padding properties
        width, height = 160, 80

        # Create new image with white background
        img = PILImage.new("RGB", (width, height), "black")
        d = ImageDraw.Draw(img)

        font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, size=size)
        text_width, text_height = d.textsize(text, font=font)
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        d.text((text_x, text_y), text, fill="white", font=font)

        # img.show()
        # Convert PIL image to OpenCV image
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Convert OpenCV image to ROS image and publish
        img_msg = self.bridge.cv2_to_imgmsg(img_cv, encoding="bgr8")
        self.publisher_image.publish(img_msg)

    def _publish(self, msg):
        self.publisher.publish(msg)

    def _hex_to_rgb(self, hex):
        """Convert a hex color to an RGB tuple."""
        value = hex.lstrip("#")
        lv = len(value)
        return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def _normalize(self, num):
        """Normalize a number to a float between 0 and 1."""
        if num == 0:
            return float(num)
        else:
            return float(num) / 255.0

    def _color(self, hex, brightness, freq):
        r, g, b = self._hex_to_rgb(hex)
        color_msg = ColorPeriod()
        color_msg.frequency = freq
        color_msg.color.a = float(brightness) / 100
        color_msg.color.r = self._normalize(r)
        color_msg.color.g = self._normalize(g)
        color_msg.color.b = self._normalize(b)
        return color_msg

    def set_leds(self, hex, brightness=100, freq=0.0):
        led_msg = LEDControl()
        data = [self._color(hex, brightness, freq)]
        led_msg.data = data
        led_msg.control_type = 0
        self._publish(led_msg)


def undistort_img(camera_matrix, dist_coeffs, img):
    h, w = img.shape[:2]

    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h), centerPrincipalPoint=True
    )

    undist_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam_mat)

    return undist_img


def undistort_from_saved_data(npz_file, img):
    data = np.load(npz_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"][0]
    return undistort_img(camera_matrix, dist_coeffs, img)
