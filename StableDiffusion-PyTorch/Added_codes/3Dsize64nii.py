import nibabel as nib

# Define the path to your NIfTI file
nifti_file = '/Users/abharian/LDM_Project/StableDiffusion-PyTorch/Train_mat/1_64/3.000000-Dynamic-3dfgre-20133.nii'

# Load the NIfTI file
img = nib.load(nifti_file)

# Get the data array
data = img.get_fdata()

# Get the shape of the data array
shape = data.shape

# Print the shape
print(f"Shape of the NIfTI file data: {shape}")

# Determine the number of channels
num_channels = shape[-1] if len(shape) > 3 else 1

# Print the number of channels
print(f"Number of channels: {num_channels}")

'''

import os
import nibabel as nib
from scipy.ndimage import zoom
import numpy as np

def resize_image(image, new_shape):
    # Calculate the zoom factors for each dimension
    factors = [n / o for n, o in zip(new_shape, image.shape)]
    return zoom(image, factors, order=1)  # Use order=1 for bilinear interpolation

def process_directory(directory, new_shape, out):
    for filename in os.listdir(directory):
        if filename.endswith(".nii"):
            filepath = os.path.join(directory, filename)
            
            # Load the NIfTI image
            img = nib.load(filepath)
            img_data = img.get_fdata()
            
            # Resize the image
            resized_data = resize_image(img_data, new_shape)
            
            # Create a new NIfTI image
            resized_img = nib.Nifti1Image(resized_data, img.affine, img.header)
            
            # Save the resized image
            new_filepath = os.path.join(out, filename)
            nib.save(resized_img, new_filepath)
            print(f"Saved resized image: {new_filepath}")

# Define the directory containing the .nii images and the new shape
directory = "/Users/abharian/LDM_Project/StableDiffusion-PyTorch/Train_mat/3_nii"
out = '/Users/abharian/LDM_Project/StableDiffusion-PyTorch/Train_mat/3_64'
new_shape = (64, 64, 64)

# Process the directory
process_directory(directory, new_shape, out)

'''

'''
import os
import numpy as np
import scipy.io as sio
import nibabel as nib

# Define the target size
target_size = (511, 351, 119)

# Define the directories
input_directory  = '/Users/abharian/LDM_Project/StableDiffusion-PyTorch/Train_mat/3'
output_directory = '/Users/abharian/LDM_Project/StableDiffusion-PyTorch/Train_mat/3_nii'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

def pad_to_target_size(data, target_size):
    """Pad the data to the target size with zeros."""
    padded_data = np.zeros(target_size, dtype=data.dtype)
    slices = tuple(slice(0, min(sz, tsz)) for sz, tsz in zip(data.shape, target_size))
    padded_data[slices] = data[slices]
    return padded_data

def convert_mat_to_nii(mat_file_path, nii_file_path, target_size):
    """Convert a .mat file to a .nii file with zero padding."""
    # Load .mat file
    mat_data = sio.loadmat(mat_file_path)
    data = mat_data['Segmented_intensity']  # Adjust if needed

    # Pad data to target size
    padded_data = pad_to_target_size(data, target_size)

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(padded_data, np.eye(4))

    # Save NIfTI image
    nib.save(nifti_img, nii_file_path)

# Iterate through .mat files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.mat'):
        mat_file_path = os.path.join(input_directory, filename)
        nii_file_name = os.path.splitext(filename)[0] + '.nii'
        nii_file_path = os.path.join(output_directory, nii_file_name)
        
        convert_mat_to_nii(mat_file_path, nii_file_path, target_size)
        print(f'Converted {filename} to {nii_file_name}')
'''