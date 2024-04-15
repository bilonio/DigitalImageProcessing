from global_hist_eq import get_equalization_transform_of_img
import numpy as np


def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w):
    region_to_eq_transform = {}
    for i in range(0, img_array.shape[0], region_len_h):
        for j in range(0, img_array.shape[1], region_len_w):
            region = img_array[i : i + region_len_h, j : j + region_len_w]
            region_to_eq_transform[(i, j)] = get_equalization_transform_of_img(region)
            print((i, j))
    return region_to_eq_transform


def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):
    a = 0
    b = 0
    T1 = T2 = T3 = T4 = np.zeros(img_array.shape)

    region_to_eq_transform = calculate_eq_transformations_of_regions(
        img_array, region_len_h, region_len_w
    )
    equalized_img = np.zeros(img_array.shape)
    # pixel_regions = find_pixel_regions(img_array,region_len_h,region_len_w);

    for x in range(0, img_array.shape[0]):
        for y in range(0, img_array.shape[1]):
            if (
                x <= region_len_h // 2
                or x >= img_array.shape[0] - region_len_h // 2
                or y <= region_len_w // 2
                or y >= img_array.shape[1] - region_len_w // 2
            ):
                pixel_region = find_pixel_regions(
                    img_array, region_len_h, region_len_w, x, y
                )
                equalized_img[x, y] = region_to_eq_transform[pixel_region][
                    img_array[x, y]
                ]
            elif (
                x + region_len_h // 2 < img_array.shape[0]
                and y + region_len_w // 2 < img_array.shape[1]
            ):
                top_left, top_right, bottom_left, bottom_right = (
                    find_nearest_contextual_centers(
                        x, y, region_len_h, region_len_w, img_array
                    )
                )
                top_left_center = get_center_from_region(
                    top_left, region_len_h, region_len_w
                )
                top_right_center = get_center_from_region(
                    top_right, region_len_h, region_len_w
                )
                bottom_left_center = get_center_from_region(
                    bottom_left, region_len_h, region_len_w
                )
                b = (x - top_left_center[0]) / (
                    bottom_left_center[0] - top_left_center[0]
                )
                a = (y - top_left_center[1]) / (
                    top_right_center[1] - top_left_center[1]
                )
                T1 = region_to_eq_transform[top_left]
                T2 = region_to_eq_transform[bottom_left]
                T3 = region_to_eq_transform[top_right]
                T4 = region_to_eq_transform[bottom_right]
                equalized_img[x, y] = (
                    (1 - a) * (1 - b) * T1[img_array[x, y]]
                    + (1 - a) * b * T2[img_array[x, y]]
                    + a * (1 - b) * T3[img_array[x, y]]
                    + a * b * T4[img_array[x, y]]
                )

    return equalized_img


def get_center_from_region(top_left, region_len_h, region_len_w):
    center = (top_left[0] + region_len_h / 2, top_left[1] + region_len_w / 2)
    return center


def find_nearest_contextual_centers(x, y, region_len_h, region_len_w, img_array):

    top_left = find_pixel_regions(
        img_array,
        region_len_h,
        region_len_w,
        x - region_len_h // 2,
        y - region_len_w // 2,
    )
    top_right = find_pixel_regions(
        img_array,
        region_len_h,
        region_len_w,
        x - region_len_h // 2,
        y + region_len_w // 2,
    )
    bottom_left = find_pixel_regions(
        img_array,
        region_len_h,
        region_len_w,
        x + region_len_h // 2,
        y - region_len_w // 2,
    )
    bottom_right = find_pixel_regions(
        img_array,
        region_len_h,
        region_len_w,
        x + region_len_h // 2,
        y + region_len_w // 2,
    )
    # print("top_left: ",top_left)

    return top_left, top_right, bottom_left, bottom_right


def find_pixel_regions(img_array, region_len_h, region_len_w, x, y):
    pixel_region = (
        (x // region_len_h) * region_len_h,
        (y // region_len_w) * region_len_w,
    )

    return pixel_region
