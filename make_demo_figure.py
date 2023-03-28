#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:44:12 2022

Used to create the first figure in the paper

@author: pbolan
"""

from utility_functions import *
from os import path
import glob 
import nibabel as nib
import matplotlib.pyplot as plt


#%%
vmin = 0
vmax_T = 4
vmax_S0 = 1
vmax_img = 1
rot90s = 0

# Uncomment out the different sections to produce the sample images
datasets_dir = get_datasets_dir()
predictions_dir = get_predictions_dir()

example_name = 'imagenet_train'
example_name = 'urand_train'
example_name = 'imagenet_test'
example_name = 'invivo_train'
match example_name:

    case 'imagenet_train':
        # IMAGENET example
        image_file = path.join(datasets_dir, 'synth_imagenet_1k_train/images/synth_000066.nii.gz')
        label_file = path.join(datasets_dir, 'synth_imagenet_1k_train/labels/synth_000066.nii.gz')

    case 'urand_train':
        #URAND example
        image_file = path.join(datasets_dir, 'synth_urand_1k_train/images/synth_000066.nii.gz')
        label_file = path.join(datasets_dir, 'synth_urand_1k_train/labels/synth_000066.nii.gz')

    case 'invivo_train':
        # INVIVO example, from test set
        image_file = path.join(datasets_dir, 'invivo2D_set1/images/invivo_000198.nii.gz')
        label_file = path.join(predictions_dir, 'INVIVO2D_SET1/FIT_NLLS/preds_000198.nii.gz' )
        vmax_T = 2
        vmax_S0 = 1
        vmax_img = .25
        rot90s = 1

    case 'imagenet_test':
        # This one is for the prediction example, figure 3
        sample_number = 1
        image_file = path.join(datasets_dir, f'synth_imagenet_1k_test/images/synth_{sample_number:06d}.nii.gz')
        label_file = path.join(datasets_dir, f'synth_imagenet_1k_test/labels/synth_{sample_number:06d}.nii.gz')
        vmax_T = 4
        vmax_S0 = 1
        vmax_img = 1
        rot90s = 0


# First the image

img3d = nib.load(image_file).get_fdata()
img3d = img3d[:,:,0,:].squeeze()
img3d = img3d / img3d.max() # normalize

fig, axs = plt.subplots(1, img3d.shape[2], figsize=[20,2])



for idx in range(img3d.shape[2]):
    axs[idx].imshow(np.rot90(img3d[:,:,idx], rot90s), vmin=vmin, vmax=vmax_img, cmap='gray', interpolation='none')
    axs[idx].set_axis_off()

#axs[0].imshow(img3d[:,:,0])

plt.show()
    

if label_file is not None:
    
    img3d = nib.load(label_file).get_fdata().squeeze()
    #img3d = img3d[:,:,sl,:].squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=[10,4])
    im = ax.imshow(np.rot90(img3d[:,:,0], rot90s), vmin=vmin, vmax=vmax_S0, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.colorbar(im)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=[10,4])
    im = ax.imshow(np.rot90(img3d[:,:,1], rot90s), vmin=vmin, vmax=vmax_T, cmap='viridis',interpolation='none')
    plt.axis('off')
    plt.colorbar(im)    
    plt.show()
        


