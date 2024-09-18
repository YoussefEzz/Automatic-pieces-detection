import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, filters
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.io import imread
import os
from skimage.color import rgb2gray

cwd = os.getcwd()
# Load picture and detect edges
image = imread(cwd + '/Images/image1.png', as_gray = True)
#edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

# Apply Gaussian filter
sigma = 5.0
image_filtered = filters.gaussian(image, sigma)

# Detect two radii
hough_radii = np.arange(120, 180, 5)
hough_res = hough_circle(image_filtered, hough_radii, True, True)

# Select the most prominent 3 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=20)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
input("Press Enter to continue...")