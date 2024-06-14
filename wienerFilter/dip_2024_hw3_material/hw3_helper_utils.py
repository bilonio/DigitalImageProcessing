import numpy as np
import cv2


def create_motion_blur_filter(length, angle):
    """
    Generates a 2D kernel, which, when convolved with an image, simulates the motion blur effect.

    :param length: (int) the length of the camera motion, in pixels
    :param angle: (float) the direction of the camera motion, in degrees (measured from the upper left corner)

    :return: a 2D np array
    """

    theta_rad = np.deg2rad(angle)

    x, y = length * np.cos(theta_rad), length * np.sin(theta_rad)

    size_x = max(1, round(x))
    size_y = max(1, round(y))

    img = np.zeros(shape=(size_x, size_y))

    cv2.line(img, pt1=(0, 0), pt2=(size_y - 1, size_x - 1), color=1.0)
    img = img / np.sum(img)

    return img
