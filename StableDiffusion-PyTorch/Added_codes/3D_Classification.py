import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the .mat file
mat = scipy.io.loadmat('/Users/abharian/Downloads/3D_Dataset/Folders/ISPY1_1078/06-15-1986-920876-Breast-Left-68528/4.000000-Dynamic-3dfgre-97710/model.mat')

# Extract the 3D image data
# Assume the variable in the .mat file containing the 3D image is named 'image_data'
image_data = mat['region_Labels']

# Define the color to count (assuming grayscale, where color is an intensity value)
Fat_specific_color = 1
Fibro_specific_color = 2  

# Count the nu5.000000-Dynamic-3dfgre-21847mber of pixels with the specific color in each slice
Fat_counts = []
Fibro_counts = []

for i in range(image_data.shape[0]):  # Assuming slices are along the first dimension
    slice_data = image_data[i, :, :]
    #print(slice_data)
    count = np.sum(slice_data == Fibro_specific_color)
    Fat_counts.append(count)
    count1 = np.sum(slice_data == Fat_specific_color)
    Fibro_counts.append(count1)
    #print(f'Slice {i}: {count} pixels with color {specific_color}')

   

# Total count across all slices
total_fat_count = np.sum(Fat_counts)
total_fibro_count = np.sum(Fibro_counts)
fibro_over_fat = total_fibro_count / (total_fibro_count + total_fat_count)
print(f'percentage: {fibro_over_fat}')
fibro_over_fat = fibro_over_fat * 100

breast_class = 0
if fibro_over_fat < 25:
    breast_class = 1
elif fibro_over_fat < 50:
    breast_class = 2
elif fibro_over_fat < 75:
    breast_class = 3
else:
    breast_class = 4

output_data = {
    **mat,  
    'fibro_over_fat': fibro_over_fat,
    'breast_class': breast_class
}

scipy.io.savemat('/Users/abharian/Downloads/3D_Dataset/train/'+ str(breast_class) + '4.000000-Dynamic-3dfgre-97710.mat', output_data)




