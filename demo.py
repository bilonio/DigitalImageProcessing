import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from global_hist_eq import perform_global_hist_transform
from global_hist_eq import get_equalization_transform_of_img
from adaptive_hist_eq import calculate_eq_transformations_of_regions
from adaptive_hist_eq import perform_adaptive_hist_equalization

# set the filepath to the image file
filename = "input_img.png"

# read the image into a PIL entity
img = Image.open(fp = filename)

# keep only the Luminance component of the image
bw_img= img.convert("L")

# obtain the underlying np array
img_array = np.array(bw_img)

# get the histogram of the image
hist, bins = np.histogram(img_array,bins = 256,range = (0,256),density = True);

# get the histogram of equalised image
equalized_hist = get_equalization_transform_of_img(img_array)

# get the equalised image
equalized_img = perform_global_hist_transform(img_array)

plt.figure(figsize=(12,6))

plt.subplot(2,2,2)
plt.imshow(img_array,cmap='gray')
plt.title("Original Image")
plt.axis('off')


plt.subplot(2,2,1)
#plt.plot(hist)
plt.bar(bins[:-1],hist,width=1)
plt.title("Histogram of the original image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(2,2,4)
plt.imshow(equalized_img,cmap='gray')
plt.title("Equalised Image")
plt.axis('off')


plt.subplot(2,2,3)
plt.bar(equalized_hist,height=hist,width=1)
plt.title("Histogram of the equalised image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# get the histogram of equalised image with contextual regions
equalised_regions_histogram = calculate_eq_transformations_of_regions(img_array,region_len_h=36,region_len_w=48)

# get the equalised image with contextual regions
equalized_img_regions = perform_adaptive_hist_equalization(img_array,region_len_h=36,region_len_w=48)
# Create a new figure
plt.figure(figsize=(12, 6))
plt.imshow(equalized_img_regions,cmap='gray')
plt.title("Equalised Image with contextual regions")
plt.show()
