import yaml
import argparse
import torch
import random
import torchvision
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)

    


    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    print(dataset_config['im_path'])
    
    data_loader_old = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)





    
    # Now modify the 2D images to 3D by repeating along the new dimension
    new_images = []
    new_labels = []
    for batch in data_loader_old:
        images = batch  # Assuming batch returns (images, labels)
        #images = F.interpolate(images, size=(dataset_config['im_size'], dataset_config['im_size']), mode='bilinear', align_corners=False)

        # Expand the 2D images (batch_size, channels, height, width) to (batch_size, channels, height, width, depth)
        # By repeating the image along the new (depth) dimension
        #images_3d = images.unsqueeze(-1).repeat(1, 1, 1, 1, 64)  # Repeating along the depth axis
        # Append to new lists
        new_images.append(images)
        # new_labels.append(labels)

    # new_labels = [label_dict['class'] for label_dict in new_labels]


    # Stack the new images and labels
    new_images = torch.cat(new_images, dim=0)  # Combine into a single tensor
    # new_labels = torch.cat(new_labels, dim=0)
    #print(new_images.shape)
    # Create a TensorDataset for the new 3D images
    new_dataset = TensorDataset(new_images)

    # Create a new DataLoader for the 3D images
    data_loader = DataLoader(new_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)

#     batch_size = train_config['autoencoder_batch_size']
#     im_channels = dataset_config['im_channels']
#     im_size = dataset_config['im_size']  # This should be a tuple (depth, height, width) for 3D
#     print(im_size)
#     im_size = (32, 32, 32)  # depth, height, width for 3D

#     # Create random images: (batch_size, channels, depth, height, width) for 3D data
# #       Adjust `im_size` and `im_channels` according to your needs
#     random_images = torch.randn(batch_size, im_channels, *im_size)

#     # Wrap it in a TensorDataset and create a DataLoader
#     random_dataset = TensorDataset(random_images)
#     data_loader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True)
    
    # summary(model, input_size=(3,256, 256,256))

    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    recon_losses_array = []
    perceptual_losses_array = []
    codebook_losses_array = []
    gen_losses_array = []
    disc_losses_array = []
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        #commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(data_loader):
            step_count += 1
            im = torch.stack(im).squeeze(0)
            #print("shape:", im.shape)
            #im= torch.randn(batch_size, im_channels, *im_size).float().to(device)

            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # Image Saving Logic
            # if step_count % image_save_steps == 0 or step_count == 1:
            #     sample_size = min(8, im.shape[0])
            #     save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
            #     save_output = ((save_output + 1) / 2)
            #     save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
            #     grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
            #     img = torchvision.transforms.ToPILImage()(grid)
            #     if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
            #         os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
            #     img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
            #                           'current_autoencoder_sample_{}.png'.format(img_save_count)))
            #     img_save_count += 1
            #     img.close()
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            
            #### Vahab
            output_v = output.mean(dim=2)
            im_v = im.mean(dim=2)

            lpips_loss = torch.mean(lpips_model(output_v, im_v)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
            recon_losses_array.append(np.mean(recon_losses))
            perceptual_losses_array.append(np.mean(perceptual_losses))
            codebook_losses_array.append(np.mean(codebook_losses))
            gen_losses_array.append(np.mean(gen_losses))
            disc_losses_array.append(np.mean(disc_losses))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
            recon_losses_array.append(np.mean(recon_losses))
            perceptual_losses_array.append(np.mean(perceptual_losses))
            codebook_losses_array.append(np.mean(codebook_losses))
            gen_losses_array.append(-1)
            disc_losses_array.append(-1)
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
    print('Done Training...')

    print(len(recon_losses_array), len(perceptual_losses_array), len(codebook_losses_array), len(gen_losses_array), len(disc_losses_array))

    plt.figure(figsize=(15, 10))

    # Plot reconstruction loss
    plt.subplot(3, 2, 1)
    plt.plot(recon_losses_array , label='Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss Over Epochs')
    plt.legend()

    # Plot Perceptual Loss
    plt.subplot(3, 2, 2)
    plt.plot(perceptual_losses_array , label='Perceptual Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Perceptual Loss Over Epochs')
    plt.legend()

    # Plot codebook Loss
    plt.subplot(3, 2, 3)
    plt.plot(codebook_losses_array , label='Codebook Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Codebook Loss Over Epochs')
    plt.legend()

    # Plot Generator Loss
    plt.subplot(3, 2, 4)
    plt.plot(gen_losses_array , label='Generator Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator Loss Over Epochs')
    plt.legend()

    # Plot Discriminator Loss
    plt.subplot(3, 2, 5)
    plt.plot(disc_losses_array  , label='Discriminator Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/abharian/LDM_Project/StableDiffusion-PyTorch/mnist/curves.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    train(args)
