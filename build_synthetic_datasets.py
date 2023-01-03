#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build synthetic datasets representing a T2 mapping series (10 evenly spaced TEs) using 
imagenet images as the true T2 and S0 maps
S(TE) = S0 * exp(-TE/T2)
    
@author: pbolan
"""
import numpy as np
import matplotlib.pyplot as plt
from os import path, getcwd, makedirs
from skimage import io
import cv2 as cv
import nibabel as nib
import sys
import pandas as pd
from utility_functions import *

#%% Generate dataset from image or noise
def generate_synthetic_dataset(
        dataset_dir, img_size, num_images, eta, use_urand_images=False, imagenet_offset=0, k_factor=2):

    # Make dirs if they don't exist
    if path.exists(dataset_dir):
        print(f'Dataset directory exists, skipping: {dataset_dir}')
        return

    print(f'Creating dataset of {num_images} samples in {dataset_dir}.')    
    print('Creating new directories')
    images_outdir = path.join(dataset_dir, 'images')
    labels_outdir = path.join(dataset_dir, 'labels')
    makedirs(dataset_dir)
    makedirs(images_outdir)
    makedirs(labels_outdir)
        
    Np = len(eta)
        
    # S0 Range: [0,1]
    S0_min = 0.
    S0_max = 1.
    
    # T is the normalized time (means T2 in prostate example)
    # Want Ts uniformly distributed
    # T2 range in [TEmin/k, TEmax*k].
    # T range in [eta_min/k, eta_max*k]
    #k_factor = 2 # k=2 is reasonable
    T_min = eta.min() / k_factor
    T_max = eta.max() * k_factor   
    
    # Noise sigma notation follows standard MRI convention. Sigma is the 
    # standard deviation of the thermal noise in both real and imaginary 
    # channels. If you then take the absolute value of that complex data you 
    # will get Rician noise, which has standard deviation of sigma*0.655 and 
    # expectation value of 1.25*sigma. See NEMA MS 1-2008 for details
    #
    # Noise range. Max signal is 1, will be lower in all images due to decay
    # For mean SNR from 0.5 to 50, noisesigma = 0.01 to 1.0
    # That might be a little noisy
    noise_min = 0.001
    noise_max = 0.1
    np.random.seed(6283)
    noisesigma = np.random.uniform(noise_min, noise_max, num_images)
        
    # Save out these noise levels to a text file
    df_noise_values = pd.DataFrame(noisesigma)
    df_noise_values.to_csv(path.join(dataset_dir, 'noisevalues.csv'))
    
    # Loop over all 2D image sets
    for idx in range(0, num_images):
        
        # Get S0 and R, either from imagenet or just uniform noise. 
        if use_urand_images:
            # If replacing S0 and R with pure noise
            S0 = np.random.uniform(low=S0_min, high=S0_max, size = [img_size, img_size])   
            T = np.random.uniform(low=T_min, high=T_max, size = [img_size, img_size])  
            
        else:
            # Select 2 images from the dataset. Offset by 22500 to get the 2nd
            # These are both returned normalied to [0,1]
            S0 = read_and_transform_imagenet_image(idx+1+imagenet_offset, img_size)
            T = read_and_transform_imagenet_image(idx+imagenet_offset+22500, img_size)

            # Scale them to the desired range
            S0 = S0 * (S0_max - S0_min) + S0_min
            T = T * (T_max - T_min) + T_min
                                                
        # Simulate the relaxometric series
        img_series = np.zeros([img_size, img_size, 1, Np])
        # Faster as a matrix multiply but easier to read as a for loop
        for jdx in range(Np):
            img_series[:,:,0, jdx] = S0[:,:] * np.exp(-eta[jdx] / T)
             
        # Add COMPLEX Gaussian noise to the full series. 
        # Values are indpendent across time and between real/imag
        noiseimg_real = np.random.standard_normal(img_series.shape)
        noiseimg_imag = np.random.standard_normal(img_series.shape)
        noiseimg_cplx = (noiseimg_real + 1j * noiseimg_imag) * noisesigma[idx]        
        img_series = img_series + noiseimg_cplx
        
        # Convert to magnitude, and Rician noise
        img_series = np.abs(img_series)
        noise_series = np.abs(noiseimg_cplx)
                
        # Display for debugging
        if False:
            plt.subplot(1,2,1)
            plt.imshow(S0)
            plt.title('S0')
            plt.subplot(1,2,2)
            plt.imshow(T)
            plt.title('T')
            plt.show()
            
            imshow_montage(img_series[:,:,0,:].squeeze(), [0, .2])
    
        # Save images as nifti. The eta direction is 3rd, will look like a slice
        ni_img = nib.Nifti1Image(img_series, affine=np.eye(4)) 
        fname = f'synth_{idx:06d}.nii.gz'
        nib.save(ni_img, path.join(images_outdir, fname))
        
        # Also save the labels (S0, T, noiseimages) in the labels dir, with matching name
        #label_series = np.zeros([img_size, img_size, 1, Np+2])
        label_series = np.zeros([img_size, img_size, 1, 2])
        label_series[:,:,0,0] = S0
        label_series[:,:,0,1] = T
        # During development I also saved out the full noise series for testing
        #label_series[:,:,0,2:] = noise_series.squeeze()
        ni_img = nib.Nifti1Image(label_series, affine=np.eye(4)) 
        fname = f'synth_{idx:06d}.nii.gz'
        nib.save(ni_img, path.join(labels_outdir, fname))


#%%
def main():
    
    # Where to create these datasets
    datasets_dir = get_datasets_dir()
    
    # Image size 
    Nsize = 128
    
    # Eta is normalized TE, so that that max eta is 1
    eta = get_eta()
    
    # Training datasets are built with a few sizes (100, 1k, 10k)    
    # The smaller ones are subsets of the larger training sets
    # Test datasets are smaller and do NOT overlap with training data (ie, 
    # they start at index=20k)
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_imagenet_10k_train'), 
            img_size=Nsize, num_images=10000, eta=eta, 
            use_urand_images=False, imagenet_offset=0, k_factor=4)        
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_imagenet_1k_train'), 
            img_size=Nsize, num_images=1000, eta=eta, 
            use_urand_images=False, imagenet_offset=0, k_factor=4)     
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_imagenet_1k_test'), 
            img_size=Nsize, num_images=1000, eta=eta, 
            use_urand_images=False, imagenet_offset=20000, k_factor=4)    
    
    
    # Random datasets are the same size     
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_urand_10k_train'), 
            img_size=Nsize, num_images=10000, eta=eta, 
            use_urand_images=True, k_factor=4)          
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_urand_1k_train'), 
            img_size=Nsize, num_images=1000, eta=eta, 
            use_urand_images=True, k_factor=4) 
    generate_synthetic_dataset(dataset_dir=path.join(datasets_dir, 'synth_urand_1k_test'), 
            img_size=Nsize, num_images=1000, eta=eta, 
            use_urand_images=True, k_factor=4)      
          
    
    print('Done.')
    
    
#%% Run the main function
if __name__=="__main__":
    main()    
    
    
    
