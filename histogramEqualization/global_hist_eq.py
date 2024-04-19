import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_equalization_transform_of_img(img):

    hist = custom_histogram(img)  # get the histogram of the image
    u = [0] * len(hist)  # get the cumulative distribution function
    y = [0] * len(hist)  # get the equalization transform
    u[0] = hist[0]  # initialize the first element of the cdf
    y[0] = 0
    for k in range(1, 256):  # calculate the cdf and equalization transform
        u[k] = hist[k] + u[k - 1]
        y[k] = round(((u[k] - u[0]) / (1 - u[0])) * (256 - 1))
    equalization_transform = y

    return equalization_transform  # return the equalization transform


def perform_global_hist_transform(img):
    equalized_img = np.zeros(img.shape)  # initialize the equalized image
    equalized_img_hist = get_equalization_transform_of_img(
        img
    )  # get the equalization transform of the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            equalized_img[i, j] = equalized_img_hist[
                img[i, j]
            ]  # get the equalized image transformation of the pixel
    return equalized_img  # return the equalized image


def custom_histogram(img):
    # Initialize histogram
    hist = [0] * 256

    # Calculate histogram
    for row in img:
        for pixel in row:
            hist[pixel] += 1

    # Normalize histogram to get probability density
    total_pixels = sum(hist)
    hist = [count / total_pixels for count in hist]

    return hist
