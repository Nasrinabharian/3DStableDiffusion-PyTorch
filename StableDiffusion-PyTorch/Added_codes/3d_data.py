import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io
import imageio
import matplotlib.pyplot as plt


mat_contents = scipy.io.loadmat('/Users/abharian/Downloads/3D_Dataset/Folders/ISPY1_1029/06-22-1985-513406-MR BREASTUNI UE-05594/5.000000-Dynamic-3dfgre-11179/model.mat')
#print(mat_contents)
for key in mat_contents.keys():
    print(key)

# Access the region_Labels data
image_data  = mat_contents['region_Labels']


# Set up parameters
num_slices = image_data.shape[0]  # Assuming the slices are along the first dimension

# Create a list to hold the images
images = []

for i in range(num_slices):
    fig, ax = plt.subplots()
    ax.imshow(image_data[i, :, :], cmap='gray')
    ax.set_title(f'Slice {i}')
    ax.axis('off')

    # Save the current figure as an image and append it to the list
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    images.append(image)

    plt.close(fig)

# Save the images as a GIF
imageio.mimsave('3d_image.gif', images, fps=10)

#print(mat_contents['tumorlabel'])
# Assuming region_labels is a 3D NumPy array
# Select a specific slice along the z-axis (e.g., middle slice)
print(image_data.size)
slice_index = 100
slice_data = image_data[slice_index, :, :]

# Plot the selected slice
plt.figure(figsize=(8, 6))
plt.imshow(slice_data)
plt.colorbar()
plt.title(f'Slice {slice_index} of region_Labels')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

data3D = mat_contents['data3D']


# Assuming data3D is a 3D NumPy array
x, y, z = np.indices(data3D.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface or scatter plot
ax.scatter(x, y, z, c=data3D.flatten())

plt.show()
