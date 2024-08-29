import os
import glob
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Directory containing .mat files
data_dir = '/Users/abharian/Downloads/3D_Dataset/train_mat_files/1'

# Load all .mat files in the directory
mat_files = glob.glob(os.path.join(data_dir, '*.mat'))

data_list = []
label_list = []

for mat_file in mat_files:
    # Load the .mat file
    mat = scipy.io.loadmat(mat_file)
    
    print(f"Keys in {mat_file}: {mat.keys()}")
    # Assume the 3D image data is stored in a variable named 'data' within the .mat file
    # You might need to inspect the structure of the loaded .mat file to find the correct key
    if 'region_Labels' in mat:
        data = mat['region_Labels']
        output_data = {
            'region_Labels': data  # Correctly structure the dictionary
        }
    
        scipy.io.savemat(mat_file, output_data)
    else:
        print(f"'region_Labels' not found in {mat_file}")
