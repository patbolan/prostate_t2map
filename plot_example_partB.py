#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 09:37:30 2022

To create the paper figure from here, run several times with different values
of dirnameB and the ROI file (muscle/prostate)

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

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score




#%% de
def make_BA_plot(A, B):
    ba_xaxis = ((A + B)/2).flatten()
    diff = B-A
    
    plt.plot(ba_xaxis, diff.flatten(), '.', alpha = 0.1)
    plt.plot([0,2], np.median(diff)*np.array([1,1]), '-r', alpha = 0.5)
    plt.plot([0,2], np.percentile(diff, 97.5)*np.array([1,1]), '--r', alpha = 0.5)
    plt.plot([0,2], np.percentile(diff, 2.5)*np.array([1,1]), '--r', alpha = 0.5)
    plt.xlim([0,2])

    plt.ylim([-.1,.1])
    plt.ylim([-1,1])

    plt.ylabel('B - A')
    plt.show()

#%%
# Lets do A-B comparison
# A will be reference
predictions_dir = get_predictions_dir()
dirnameA = path.join(predictions_dir, 'INVIVO2D_SET3/FIT_NLLS')
dirnameB = path.join(predictions_dir, 'INVIVO2D_SET3/CNN_SS_INVIVO')
fname = 'preds_000277.nii.gz'

tmpA = nib.load(path.join(dirnameA, fname)).get_fdata()    
tmpB = nib.load(path.join(dirnameB, fname)).get_fdata()    
T_A = tmpA[:,:,0,1]
T_B = tmpB[:,:,0,1]

# Transform so they match itksnaps view
T_A = np.rot90(T_A[::-1, :])
T_B = np.rot90(T_B[::-1, :])
diff = T_B - T_A

# Plot A, B, difference. Separate figure
figsize = [7.5,6]
fig, ax = plt.subplots(1,1,figsize=figsize)
im = ax.imshow(T_A, vmin=0, vmax=1.5, interpolation='none')
plt.axis('off')
plt.colorbar(im)
plt.show()

fig, ax = plt.subplots(1,1,figsize=figsize)
im = ax.imshow(T_B, vmin=0, vmax=1.5, interpolation='none')
plt.axis('off')
plt.colorbar(im)
plt.show()

fig, ax = plt.subplots(1,1,figsize=figsize)
im = ax.imshow(diff, vmin=-0.2, vmax=0.2, cmap=cmr.guppy_r, interpolation='none')
plt.axis('off')
plt.colorbar(im)
plt.show()

# axs[1].imshow(T_B, vmin=0, vmax=1.5)
# plt.axis('off')
# im = axs[2].imshow(diff, vmin=-.1, vmax=0.1)
# plt.axis('off')
# plt.colorbar(im)
# plt.show()

# These are nice, but not for paper
#make_BA_plot(T_A, T_B)
#plt.show() 

#%%
fig = plt.figure(figsize=[6,6])
plt.plot(T_A.flatten(), T_B.flatten(), 'ob', alpha=0.1,mec='b', mfc='b',mew=1,ms=4)
plt.plot([0,2], [0,2], '-r')
plt.xlim([0,2])
plt.ylim([0,2])
plt.show()


# STats
pearson_stat, pearson_pvalue = pearsonr(T_A.flatten(), T_B.flatten())
spearman_rho, spearman_pvalue = spearmanr(T_A.flatten(), T_B.flatten())
print(f'R^2 {r2_score(T_A, T_B)}')
print(f'Pearson R {pearson_stat}, p={pearson_pvalue:.6f}')
print(f'Spearman rho {spearman_rho}, p={spearman_pvalue:.6f}')
#%%
# Load the roi
roi_file = path.join(get_base_dir(), 'example_partB/invivo_set3_000277/roi_prostate.tif')
roi_file = path.join(get_base_dir(), 'example_partB/invivo_set3_000277/roi_muscle.tif')
roi_img = np.array(Image.open(roi_file))

# Normalize, flip both directions
roi_img[roi_img>0] = 1
roi_img = roi_img[::-1,::-1]

fig, ax = plt.subplots(1,1,figsize=figsize)
#im = ax.imshow(T_A*roi_img, vmin=0, vmax=1.5)
im = ax.imshow(roi_img, vmin=0, vmax=1, cmap='gray')
#im = ax.imshow(roi_img, vmin=-1, vmax=-.1, cmap='gray')
plt.axis('off')
#plt.colorbar(im)
plt.show()


valsA = T_A*roi_img
valsB = T_B*roi_img
# make_BA_plot(valsA, valsB)
# plt.show()

# plt.plot(valsA, valsB, '.b', alpha=0.2)
# plt.plot([0,2], [0,2], '-r')
# plt.xlim([0,2])
# plt.ylim([0,2])
# plt.show()

fig = plt.figure(figsize=[6,6])
plt.plot(valsA, valsB, 'ob', alpha=0.1,mec='b', mfc='b',mew=1,ms=4)
plt.plot([0,2], [0,2], '-r')
plt.xlim([0,2])
plt.ylim([0,2])
plt.show()


# STats
pearson_stat, pearson_pvalue = pearsonr(valsA.flatten(), valsB.flatten())
spearman_rho, spearman_pvalue = spearmanr(valsA.flatten(), valsB.flatten())
print(f'R^2 {r2_score(valsA, valsB)}')
print(f'Pearson R {pearson_stat}, p={pearson_pvalue:.6f}')
print(f'Spearman rho {spearman_rho}, p={spearman_pvalue:.6f}')

#%%
# This is a cool histogram overlay. Not using for paper
# _ = plt.hist(T_A.flatten(), bins=100, range=[0.001, 2], alpha=0.5)
# _ = plt.hist(T_B.flatten(), bins=100, range=[0.001, 2], alpha=0.5)
# plt.show()







