import glob
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from utils.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class Npz3DDataset(Dataset):
    r"""
    Dataset class for 3D images stored in .npz files.
    """
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_size: Size to which 3D images will be resized (as a tuple).
        :param im_channels: Number of channels in the 3D image.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        
        # Whether to load images and call VAE or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'Found {len(self.latent_maps)} latents')
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all .npz files from the path specified
        and loads the 3D images.
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), f"Images path {im_path} does not exist"
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('npz')))
            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_types:
                    labels.append(int(d_name))
        print(f'Found {len(ims)} images for split {self.split}')
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'class' in self.condition_types:
            cond_inputs['class'] = self.labels[index]
        #######################################
    
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Load the 3D image from the .npz file
            #npz_data = np.load(self.images[index], allow_pickle=True)
            #print("Arrays in the .npz file:", npz_data.files)
            data = np.load(self.images[index], allow_pickle=True)
 
            # List all the keys in the .npz file
            print("Keys in the .npz file:", data.files)
            # Initialize a dictionary to store all arrays and their corresponding names
            extracted_data = {}
 
            # Loop through all keys in the .npz file
            for key in data.files:
                array_data = data[key]
    
                # If the array is an object (like a dictionary), further extract its content
                if isinstance(array_data, np.ndarray) and array_data.dtype == object:
                    array_data = array_data.item()  # Convert to a dictionary if it's a stored object
        
                    # Store each sub-key and its corresponding array
                    for sub_key, sub_array in array_data.items():
                        extracted_data[sub_key] = sub_array
                        print(f"Extracted array '{sub_key}' from '{key}' with shape {sub_array.shape}")
                else:
                    # Directly store the array if it's not an object
                    extracted_data[key] = array_data
                    print(f"Extracted array '{key}' with shape {array_data.shape}")
 
 
 





            #im = np.asarray(im, dtype=np.float32)
            im_tensor = {}
            # Convert each array to a torch tensor and perform the operation

            for sub_key, array in extracted_data.items():
                tensor = torch.tensor(array, dtype=torch.float32)
                # Resize the image to the desired size (if necessary)
                if self.im_size is not None:
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0).unsqueeze(0), 
                        size=self.im_size, 
                        mode='trilinear'
                    ).squeeze(0).squeeze(0)                   
                # Convert input to -1 to 1 range
                tensor = (2 * tensor) - 1
                # Store the processed tensor back in the dictionary
                im_tensor[sub_key] = tensor
            '''
            # Convert to torch tensor
            hashable_sub_key = str(array_data.items())  # or tuple(sub_key)

            # Now use the hashable_sub_key
            im_tensor[hashable_sub_key] = torch.tensor(extracted_data[sub_key], dtype=torch.float32)
            #for sub_key in array_data.items():
             #   im_tensor[sub_key] = torch.tensor(extracted_data[sub_key], dtype=torch.float32)
        
            # Resize the image to the desired size (you might need to implement this resizing if necessary)
            #if self.im_size is not None:
            #    im_tensor = torch.nn.functional.interpolate(im_tensor.unsqueeze(0).unsqueeze(0), size=self.im_size, mode='trilinear').squeeze(0).squeeze(0)
        
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1'''
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs





'''
class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    
    def __init__(self, split, im_path, im_size, im_channels,
                 use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        
        # Conditioning for the dataset
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)
        
        # Whether to load images and call vae or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
        
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, d_name, '*.{}'.format('png')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpg')))
            fnames += glob.glob(os.path.join(im_path, d_name, '*.{}'.format('jpeg')))
            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_types:
                    labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'class' in self.condition_types:
            cond_inputs['class'] = self.labels[index]
        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.ToTensor()(im)
            
            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs'''
            
