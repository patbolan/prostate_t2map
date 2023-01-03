#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:47:39 2022

@author: pbolan
"""

import numpy as np
import glob
from os import path
import nibabel as nib
import matplotlib.pyplot as plt

from utility_functions import get_evaluation_paths, image_threshold_percentage, get_plot_dir, imshow_montage



    
#%% THis is development, reviewing iamges
def review_files(images_file, file_true, file_pred):
    img_src = nib.load(images_file).get_fdata()
    img_true = nib.load(file_true).get_fdata()
    img_pred = nib.load(file_pred).get_fdata()
    diff = img_true-img_pred
    
    # Source data
    imshow_montage(img_src.squeeze())

    # For now, let's just make a pretty picture!
    sl = int(img_true.shape[2]/2) # Take middle slice
    fig, axs = plt.subplots(2,3, figsize=[6,4])
    axs[0,0].imshow(img_true[:,:,sl,0], vmin=0, vmax=1)
    axs[0,1].imshow(img_pred[:,:,sl,0], vmin=0, vmax=1)
    axs[0,2].imshow(diff[:,:,sl,0], vmin=-0.05, vmax=0.05)
    axs[0,0].set_ylabel('S0')
    
    axs[1,0].imshow(img_true[:,:,sl,1], vmin=0, vmax=2)
    axs[1,1].imshow(img_pred[:,:,sl,1], vmin=0, vmax=2)
    axs[1,2].imshow(diff[:,:,sl,1], vmin=-0.05, vmax=0.05)
    axs[1,0].set_ylabel('T')
    axs[1,0].set_xlabel('true')
    axs[1,1].set_xlabel('pred')
    axs[1,2].set_xlabel('diff')
    plt.show()
  
    
    # Mask to only evaluate in "foreground"
    # Consider background the lowest 5% of the image. Arbitrary
    #thr = filters.threshold_otsu(img_true[:,:,:,0])
    thr = image_threshold_percentage(img_true[:,:,:,0], 1)
    fgmask = img_true[:,:,sl,0] > thr
    #plt.imshow(fgmask)
    #plt.show()
    #plt.imshow(fgmask  * img_true[:,:,:,0])
    #plt.imshow(fgmask)
    
    # Error statistics, after masking
    S0_true = img_true[:,:,sl,0]                
    S0_pred = img_pred[:,:,sl,0]                
    T_true = img_true[:,:,sl,1]                
    T_pred = img_pred[:,:,sl,1]
    S0_err = (S0_true - S0_pred)                 
    T_err = (T_true - T_pred)            
        

    # Diagonal plots
    fig,axs = plt.subplots(1,2,figsize=[6,3])
    axs[0].plot(S0_true[fgmask], S0_pred[fgmask], '.')    
    axs[0].set_ylabel('Pred')
    axs[0].set_xlabel('S0')
    axs[0].set_ylim([0, 1])  
    axs[1].plot(T_true[fgmask], T_pred[fgmask], '.')
    axs[1].set_xlabel('T')
    axs[1].set_ylim([0, 2])

    # Bland-Altman style plots
    fig,axs = plt.subplots(1,2,figsize=[6,3])
    axs[0].plot(S0_true[fgmask], S0_err[fgmask], '.')    
    axs[0].set_ylabel('Error')
    axs[0].set_xlabel('S0')
    axs[0].set_ylim([-.5, .5])  
    axs[1].plot(T_true[fgmask], T_err[fgmask], '.')
    axs[1].set_xlabel('T')
    axs[1].set_ylim([-1, 1])  
    
    plt.show()  
    
    # PRint some summary stats
    print('S0 true: '+ get_summary_stats_string(img_true[:,:,:,0]))
    print('S0 pred: '+ get_summary_stats_string(img_pred[:,:,:,0]))
    print('T true:  '+ get_summary_stats_string(img_true[:,:,:,1]))
    print('T pred:  '+ get_summary_stats_string(img_pred[:,:,:,1]))
    
    print(f'MAE S0 = {np.mean(np.abs(S0_err[fgmask])):.4f}')
    print(f'MAE T  = {np.mean(np.abs(T_err[fgmask])):.4f} MASKED')
    
   
    
    
#%%
def review_one_case(ds_name, model_name, file_num_str):
        output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths(model_name, ds_name)
        
        images_file = glob.glob(path.join(ds_images_dir, f'*{file_num_str}.nii.gz'))[0]
        pred_file = glob.glob(path.join(output_dir, f'*{file_num_str}.nii.gz'))[0]
        label_file = glob.glob(path.join(ds_labels_dir, f'*{file_num_str}.nii.gz'))[0]
        
        # Calculate metrics
        review_files(images_file, label_file, pred_file)   

def review_cases(ds_name, model_name):

    # Get all the output files
    output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths(model_name, ds_name)
    pred_files = sorted(glob.glob(path.join(output_dir, '*.nii.gz')))
    for idx, pred_file in enumerate(pred_files):
        # extract the case number, find matching file in ds_labels_dir
        file_num_str = get_file_numstr_from_file_name(pred_file)
    
        print('****************************** ')
        print(f'{ds_name}, {model_name}, {idx}, {file_num_str}')
        review_one_case(ds_name, model_name, file_num_str)
        
        
    return


#%%
#review_cases('IMAGENET_VALIDATION', 'FIT_LOGLIN')
#review_cases('IMAGENET_VALIDATION', 'FIT_NLLS')
#review_cases('IMAGENET_VALIDATION', 'NN1D_URAND')

casestr = '000092'
review_one_case('IMAGENET_VALIDATION', 'FIT_NLLS', casestr)
review_one_case('IMAGENET_VALIDATION', 'NN1D_URAND', casestr)
    
    
    
    