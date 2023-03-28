#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 11:17:33 2022

@author: pbolan
"""

import os
import sys
import glob

import nibabel as nib
from utility_functions import save_nifti, get_datasets_dir

#%%
def extract_2D_from_3D(source_dataset, target_dataset):
    
    if os.path.exists(target_dataset):
        print(f'Target already exists. Remove manually: {target_dataset}')
        return
    else:
        os.makedirs(target_dataset)    
        print(f'Creating {target_dataset}')
        
    source_files = glob.glob(os.path.join(source_dataset, '*.nii.gz'))
    source_files.sort()
    
    slicenum = 0
    for image_file in source_files:
        # load the nifti file 
        img = nib.load(image_file).get_fdata()
        print(f'Extracting {img.shape[2]} slices from {image_file} starting at {slicenum}')
        
        for idx in range(0,img.shape[2]):
            
            fname = os.path.join(target_dataset, f'invivo_{slicenum:06d}.nii.gz')
            tmp = img[:,:,[idx],:] # Just pick the slice dim, don't collapse it
            save_nifti(tmp, fname)
            
            slicenum += 1


#%%
def extract_all():
    dataset_root = get_datasets_dir()
    
    
    # Run this for datasets 1-3 inclusive
    for idx in range(0,3):    
        source_dataset = os.path.join(dataset_root, f'invivo_set{idx+1}','images')
        target_dataset = os.path.join(dataset_root, f'invivo2D_set{idx+1}', 'images')
           
        extract_2D_from_3D(source_dataset, target_dataset)

if __name__=='__main__':
    extract_all()
