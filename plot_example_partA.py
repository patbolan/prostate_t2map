#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 09:37:30 2022

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


#%%
# Compare predictions with true T values for all methods
dirnameA = path.join(get_datasets_dir(), 'synth_imagenet_1k_test/labels')

# Select an example case
# Nicest example is 105, but it include a face, which is a problem for 
# medarxiv and some publications. Other examples that show detail, range 
# of T2 values, noise regions, etc include:
# 10, 170, 346, 348, 388, 1,47,70,97, 405
# Sample #1 is good and does not include faces
sample_number = 1
fnameA = f'synth_{sample_number:06d}.nii.gz'
fnameB = f'preds_{sample_number:06d}.nii.gz'

# Note - use make_demo_figure.py to generate the images at the top

# Select methods
# For the paper figure, there are 14 predictions. 
# To lay it out in 3 rows by 5 columns, I will double up the FIT_NLLS_RICE
method_names = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE', 'FIT_NLLS_RICE',
                'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_INVIVO', 
                'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO']

# Settings for this figure
print_stats = False
plot_figure = True
plot_figure_titles = False
plot_diff = True

if plot_figure: 
    figsize = [12,8]
    figsize = [18,12]
    fig, axs = plt.subplots(3,5, figsize=figsize)

# loop over rows and columns
for irow in range(0,3):
    for icol in range(0,5):
        idx = icol + irow * 5
        print(f'({idx}) {method_names[idx]}: {irow},{icol}')
        method = method_names[idx]        

        # Get the images, calculate difference    
        dirnameB = path.join(get_predictions_dir(), f'IMAGENET_TEST_1k/{method}')        
        tmpA = nib.load(path.join(dirnameA, fnameA)).get_fdata()    
        tmpB = nib.load(path.join(dirnameB, fnameB)).get_fdata()    
        T_A = tmpA[:,:,0,1]
        T_B = tmpB[:,:,0,1]
        diff = T_B - T_A
    
        if plot_figure:    
            ax = axs[irow,icol]
            
            if plot_diff: 
                im = ax.imshow(diff, vmin=-2, vmax=2, cmap=cmr.guppy_r, interpolation='none')
            else:
                im = ax.imshow(T_B, vmin=0, vmax=4, interpolation='none')
            if plot_figure_titles:
                ax.set_title(method, fontsize=8) 
            ax.set_axis_off()
        
    
        if print_stats:
            print(f'median absolute error {np.median(np.abs(diff))}, median error {np.median(diff)}')

# To manually make the fig you may need a colorbar. It is not in the correct 
# layout, but you can uncomment this and arrange it in the figure later
#plt.colorbar(im)

# Some manual tweaks to get the montage right
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.0)
plt.show()
    





