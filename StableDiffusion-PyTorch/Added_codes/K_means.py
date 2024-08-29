
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the image
image = cv2.imread('/Users/abharian/LDM_Project/StableDiffusion-PyTorch/mnist/Generated_Results/image_1.png')
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Convert image to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Apply k-means clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(pixels)

# Replace each pixel value with its corresponding center value
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape).astype(np.uint8)

# Display the original and segmented images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image with k={}'.format(k))
plt.imshow(segmented_img)
plt.axis('off')

plt.show()

