# Import packages
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import cv2 # pip3 install opencv-python-headless

# What are the files in this dir?
import os
# print the dir for all the files within the dir
for dirname, dirnames, filenames in os.walk('./'):
    # Remove '.git' and 'compressed' from the list of subdirs to avoid walking into it
    if '.git' in dirnames:
        dirnames.remove('.git')
    if 'compressed' in dirnames:
        dirnames.remove('compressed')
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Input
image_path = str(input("Please enter the path/to/image: "))

# How many clusters? must be an integer
while True:
    try:
        k_clusters = int(input("Specify the number of dominant colors to use in compressed image: "))
        break  # If the conversion to int is successful, break out of the loop
    except ValueError:  # If the conversion fails, a ValueError is raised
        print("Invalid input. Please enter an integer.")

# Dimensions of the image in pixels and number of colors
def image_dimensions(image):
    height = image.shape[0]
    width = image.shape[1]
    c = image.shape[2]
    print(f'Height = {height}\nWidth = {width}\nNumber of colors per pixel = {c}')
    return height, width, c

# Extract image name and extension
def image_name_ext(path_to_image):
    base_name = os.path.basename(path_to_image)
    name = os.path.splitext(base_name)[0]
    extension = os.path.splitext(base_name)[-1]
    return name, extension

# Save iamge name and extension
image_name, image_extension = image_name_ext(image_path)

# Load the image and reshape it to a 2D array of pixels
original_image = cv2.imread(image_path)

# Convert to RGB (BGR is default in openCV2)
image_RGB = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Image dimensions
h, w, c =image_dimensions(image_RGB)

# Flatten to 1D array
flatten_pixels = image_RGB.reshape((-1, 3))

# To cluster around the common colors, first get the frequency for unique pixels and get their frequency and weights
pixel_freq = Counter(map(tuple, flatten_pixels))
unique_pixels = np.array(list(pixel_freq.keys()))
weights = np.array(list(pixel_freq.values()))

# Create a KMeans instance with k_clusters 
k_Means = KMeans(n_clusters=k_clusters, init='k-means++', n_init=20, max_iter=200)

# Fit weighted k_means around the most common pixels
k_Means.fit(unique_pixels, sample_weight=weights)

# Fetch the centroids and labels
cluster_centers = k_Means.cluster_centers_
compressed_labels = k_Means.labels_

# Make sure centroids are 8-bit integer (0-255) and flatten
palette = np.uint8(cluster_centers).reshape(-1, 3)

# Sort ascending by the Red channel (index 0)
sorted_palette = palette[np.argsort(-palette[:, 0])]

# Reshape it back to the original dimensions
sorted_palette = sorted_palette.reshape(1, k_clusters, c)

# Get the labels for the original flattened pixel array
original_labels = k_Means.predict(flatten_pixels)

# Map each original pixel to its corresponding centroid
compressed_pixels = cluster_centers[original_labels]

# Reshape to the original image shape
compressed_image = compressed_pixels.reshape(image_RGB.shape).astype(np.uint8)

# width and height of color bars are determined based on the resolution and number of clusters
color_bar_height = h // k_clusters
color_bar_width = w // 20 # 5% of the original image's width
dpi = 70
fig, axes = plt.subplots(1,2,figsize=(16,9))

# Create a color bar image
color_bar = np.zeros((h, color_bar_width, 3), dtype=np.uint8)

# Fill the color bar with the sorted_palette colors
for idx, color in enumerate(sorted_palette[0]):
    start_y = idx * color_bar_height
    end_y = (idx + 1) * color_bar_height
    color_bar[start_y:end_y, :] = color

# Fill any remaining space with the last color
if end_y < h:
    color_bar[end_y:, :] = sorted_palette[0][-1]

# Concatenate the compressed image and the color bar
concatenated_image = np.hstack((compressed_image, color_bar))

# Save the concatenated image
cv2.imwrite('./compressed/compressed_' + image_name + image_extension, cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR))

