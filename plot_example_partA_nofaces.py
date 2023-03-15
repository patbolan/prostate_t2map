#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 09:37:30 2022
20230110 Needed to select an example without human faces for the medarXiv 
version

@author: pbolan
"""

import numpy as np
import matplotlib.pyplot as plt
from os import path, getcwd, makedirs
import glob
from skimage import io
import cv2 as cv # Note some issues with opencv-python versions. U
import nibabel as nib
import sys
import time
from utility_functions import *
from PIL import Image

import cmasher as cmr



#%%  Code to marge a bunch of niftis into one big one, for review
# dirname = '/home/pbolan/prj/prostate_t2map/predictions/INVIVO2D_SET3/FIT_NLLS'
# dirname = '/home/pbolan/prj/prostate_t2map/predictions/INVIVO2D_SET3/CNN_IMAGENET'
# dirname = '/home/pbolan/prj/prostate_t2map/predictions/INVIVO2D_SET3/CNN_SS_INVIVO'

# dirname = '/home/pbolan/prj/prostate_t2map/datasets/synth_imagenet_1k_test/labels'
# #combine_files(dirname, path.join(dirname, 'allfiles.nii.gz'))
# combine_files(dirname)

# sys.exit()



#%%
# Lets do A-B comparison
# A will be reference
dirnameA = '/home/pbolan/prj/prostate_t2map/datasets/synth_imagenet_1k_test/labels'

method_names = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE', 
                'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_INVIVO', 
                'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO']

# TEMP Just a few to help pick out new example
#method_names = ['FIT_LOGLIN', 'CNN_IMAGENET',]
#method_names = ['FIT_LOGLIN',]


for method in method_names:
    
    dirnameB = f'/home/pbolan/prj/prostate_t2map/predictions/IMAGENET_TEST_1k/{method}'
    #dirnameB = '/home/pbolan/prj/prostate_t2map/predictions/IMAGENET_TEST_1k/CNN_IMAGENET'
    #dirnameB = '/home/pbolan/prj/prostate_t2map/predictions/IMAGENET_TEST_1k/NN1D_URAND'
    
    # 10, 105 are nice examples. Detail, range of T2 values, noise regions
    # Also OK: 170,346,348,388
    # More selections: 1,47,70,97, 405
    sample_number = 1
    fnameA = f'synth_{sample_number:06d}.nii.gz'
    fnameB = f'preds_{sample_number:06d}.nii.gz'
    
    tmpA = nib.load(path.join(dirnameA, fnameA)).get_fdata()    
    tmpB = nib.load(path.join(dirnameB, fnameB)).get_fdata()    
    T_A = tmpA[:,:,0,1]
    T_B = tmpB[:,:,0,1]
    
    # Transform so they match itksnaps view
    #T_A = np.rot90(T_A[::-1, :])
    #T_B = np.rot90(T_B[::-1, :])
    diff = T_B - T_A
    
    # Show the S0 too, quickly
    # fig,ax = plt.subplots(1,1,figsize=[8,8])
    # ax.imshow(tmpA[:,:,0,0], vmin=0, vmax=1, cmap='gray')
    # plt.show()
    
    # # Show the Ttrue too, 
    # fig,ax = plt.subplots(1,1,figsize=[8,8])
    # ax.imshow(tmpA[:,:,0,1], vmin=0, vmax=4, cmap='viridis', interpolation='none')
    # plt.show()
    
    
    figsize=[12,8]
    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    im = ax.imshow(T_B, vmin=0, vmax=4, interpolation='none')
    plt.colorbar(im)
    plt.title(method)
    plt.axis('off')
    plt.show()


    fig, ax = plt.subplots(1,1,figsize=figsize)
    im = ax.imshow(diff, vmin=-2, vmax=2, cmap=cmr.guppy_r, interpolation='none')
    plt.colorbar(im)
    plt.title(method)
    plt.axis('off')
    plt.show()



    # Print stats underneath
    print(f'median absolute error {method}: {np.median(np.abs(diff))}, median error {np.median(diff)}')
    





