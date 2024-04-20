from global_hist_eq import get_equalization_transform_of_img
import numpy as np


def calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w):
    region_to_eq_transform = (
        {}
    )  # initialize the dictionary to store the equalization transform of each region
    for i in range(0, img_array.shape[0], region_len_h):
        for j in range(0, img_array.shape[1], region_len_w):
            region = img_array[
                i : i + region_len_h, j : j + region_len_w
            ]  # region of the image
            region_to_eq_transform[(i, j)] = get_equalization_transform_of_img(
                region
            )  # transform the region and store the equalization transform

    return region_to_eq_transform  # return the dictionary of equalization transforms


def perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w):
    a = 0
    b = 0
    T1 = T2 = T3 = T4 = np.zeros(
        img_array.shape
    )  # initialize the equalization transforms of the regions

    region_to_eq_transform = calculate_eq_transformations_of_regions(
        img_array,
        region_len_h,
        region_len_w,  # region_len_h and region_len_w are the dimensions of the region
    )
    equalized_img = np.zeros(
        img_array.shape
    )  # initialize the equalization transforms of the regions

    for x in range(0, img_array.shape[0]):
        for y in range(0, img_array.shape[1]):
            if (  # check if the pixel is at the boundary of the image
                x <= region_len_h // 2
                or x >= img_array.shape[0] - region_len_h // 2
                or y <= region_len_w // 2
                or y >= img_array.shape[1] - region_len_w // 2
            ):
                pixel_region = find_pixel_regions(
                    
                    region_len_h,
                    region_len_w,
                    x,
                    y,  # get the region of the pixel
                )
                equalized_img[x, y] = region_to_eq_transform[pixel_region][
                    img_array[
                        x, y
                    ]  # get the equalization transform of the pixel region
                ]
            elif (
                x + region_len_h // 2 < img_array.shape[0]
                and y + region_len_w // 2 < img_array.shape[1]
            ):
                top_left, top_right, bottom_left, bottom_right = (
                    find_nearest_contextual_regions(  # get the nearest contextual regions
                        x, y, region_len_h, region_len_w
                    )
                )
                top_left_center = get_center_from_region(
                    top_left,
                    region_len_h,
                    region_len_w,  # get the center of the top left region
                )
                top_right_center = get_center_from_region(
                    top_right,
                    region_len_h,
                    region_len_w,  # get the center of the top right region
                )
                bottom_left_center = get_center_from_region(
                    bottom_left,
                    region_len_h,
                    region_len_w,  # get the center of the bottom left region
                )
                b = (x - top_left_center[0]) / (
                    bottom_left_center[0]
                    - top_left_center[0]  # calculate the interpolation factor b
                )
                a = (y - top_left_center[1]) / (
                    top_right_center[1]
                    - top_left_center[1]  # calculate the interpolation factor a
                )
                T1 = region_to_eq_transform[
                    top_left
                ]  # calculate the equalization transform of the top left region
                T2 = region_to_eq_transform[
                    bottom_left
                ]  # calculate the equalization transform of the bottom left region
                T3 = region_to_eq_transform[
                    top_right
                ]  # calculate the equalization transform of the top right region
                T4 = region_to_eq_transform[
                    bottom_right
                ]  # calculate the equalization transform of the bottom right region
                equalized_img[x, y] = (
                    (1 - a) * (1 - b) * T1[img_array[x, y]]
                    + (1 - a)
                    * b
                    * T2[img_array[x, y]]  # calculate the equalized pixel value
                    + a * (1 - b) * T3[img_array[x, y]]
                    + a * b * T4[img_array[x, y]]
                )

    return equalized_img  # return the equalized image


def get_center_from_region(top_left, region_len_h, region_len_w):
    center = (
        top_left[0] + region_len_h / 2,
        top_left[1] + region_len_w / 2,
    )  # calculate the center of the region
    return center


def find_nearest_contextual_regions(x, y, region_len_h, region_len_w):

    top_left = find_pixel_regions(  # get the top left region of the pixel
        
        region_len_h,
        region_len_w,
        x - region_len_h // 2,
        y - region_len_w // 2,
    )
    top_right = find_pixel_regions(  # get the top right region of the pixel
        
        region_len_h,
        region_len_w,
        x - region_len_h // 2,
        y + region_len_w // 2,
    )
    bottom_left = find_pixel_regions(  # get the bottom left region of the pixel
        
        region_len_h,
        region_len_w,
        x + region_len_h // 2,
        y - region_len_w // 2,
    )
    bottom_right = find_pixel_regions(  # get the bottom right region of the pixel
        
        region_len_h,
        region_len_w,
        x + region_len_h // 2,
        y + region_len_w // 2,
    )

    return (
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    )  # return the nearest contextual regions


def find_pixel_regions(region_len_h, region_len_w, x, y):
    pixel_region = (
        (x // region_len_h) * region_len_h,  # get the region of the pixel
        (y // region_len_w) * region_len_w,
    )

    return pixel_region


def perform_adaptive_no_interp(img_array, region_len_h, region_len_w):
    equalized_img = np.zeros(
        img_array.shape
    )  # initialize the equalization transforms of the regions
    region_to_eq_transform = calculate_eq_transformations_of_regions(
        img_array,
        region_len_h,
        region_len_w,  # region_len_h and region_len_w are the dimensions of the region
    )

    for x in range(0, img_array.shape[0]):
        for y in range(0, img_array.shape[1]):
            pixel_region = find_pixel_regions(
                
                region_len_h,
                region_len_w,
                x,
                y,  # get the region of the pixel
            )
            equalized_img[x, y] = region_to_eq_transform[pixel_region][
                img_array[x, y]  # get the equalization transform of the pixel region
            ]
    return equalized_img  # return the equalized image
