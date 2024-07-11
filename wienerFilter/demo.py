from PIL import Image
from scipy.ndimage import convolve
from dip_2024_hw3_material import hw3_helper_utils
import numpy as np
import matplotlib.pyplot as plt
from wiener_filtering import my_wiener_filter
from inverseFilter import inverse_filter
from find_optimal import find_optimal_k
from plot_region_optimal import plot_region_optimal

# Paths to the images
image_path1 = "wienerFilter/dip_2024_hw3_material/cameraman.tif"
image_path2 = "wienerFilter/dip_2024_hw3_material/checkerboard.tif"

for image_path in [image_path1, image_path2]:
    # 'x' is the input grayscale image, of type float and normalized to [0,1]
    x = Image.open(image_path)  # read the image

    x = np.array(x, dtype=np.float32)  # convert the image to a numpy array of floats
    x = x / 255.0  # normalize the image to [0,1]

    # create white noise with level 0.02
    v = 0.02 * np.random.randn(*x.shape)

    # create motion blur filter
    h = hw3_helper_utils.create_motion_blur_filter(length=20, angle=30)

    # obtain the filtered image
    y0 = convolve(x, h, mode="wrap")

    # generate the noisy image
    y = y0 + v

    k_opt, Jmin = find_optimal_k(x, y, h)  # find the optimal k
    plot_region_optimal(k_opt, y, h, x)  # plot J values in region around the optimal k
    print(k_opt, Jmin)
    x_hat = my_wiener_filter(y, h, k_opt)  # apply the Wiener filter
    J = (x - x_hat) ** 2
    x_inv0 = inverse_filter(
        y0, h, k_opt
    )  # apply the inverse filter to the noiseless image
    x_inv = inverse_filter(y, h, k_opt)  # apply the inverse filter to the noisy image

    fig, axs = plt.subplots(
        nrows=2, ncols=3
    )  # create a figure with 2 rows and 3 columns

    axs[0][0].imshow(x, cmap="gray")
    axs[0][0].set_title("Original Image x")

    axs[0][1].imshow(y0, cmap="gray")
    axs[0][1].set_title("Clean Image y0")

    axs[0][2].imshow(y, cmap="gray")
    axs[0][2].set_title("Blurred and Noisy Image y")

    axs[1][0].imshow(x_inv0, cmap="gray")
    axs[1][0].set_title("Inverse filtering noiseless output x_inv0")

    axs[1][1].imshow(x_inv, cmap="gray")
    axs[1][1].set_title("Inverse filtering noisy output x_inv")

    axs[1][2].imshow(x_hat, cmap="gray")
    axs[1][2].set_title("Wiener filtering output x_hat")
plt.show()
