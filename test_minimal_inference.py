#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 12:15:43 2022

Simple as possible file to do inference

@author: pbolan
"""

import torch
from os import path
import nibabel as nib
import glob
import numpy as np
from matplotlib import pyplot as plt

from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)

from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from utility_functions import *

#%%
device = torch.device("cpu")
data_dir = '/home/pbolan/prj/prostate_t2map/datasets/synth_imagenet_100_train/images'

model_scripted_filename ='/home/pbolan/prj/prostate_t2map/models/cnn_ss_imagenet_XXX.pt'
model = torch.jit.load(model_scripted_filename)
model = model.to(device)

# Use MONAI dataset
train_images = sorted(glob.glob(path.join(data_dir, "*.nii.gz")))
# Make this with a dummy labels entry too
data_dicts = [{"image": image_name, "label": image_name}
    for image_name in train_images ]

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(
            1., 1.), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"]),
    ]
)

train_ds = CacheDataset(
    data=data_dicts, transform=transforms, cache_rate=1.0, num_workers=4 )
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

#%%
model.eval()
with torch.no_grad():
    
    for data in train_loader:
        #val_data = first(val_loader)
        
        inputs, labels = (
            data["image"].to(device),
            data["label"].to(device),
        )
           
        outputs = model(inputs[:,:,:,:,0])
        
        plt.imshow(outputs.detach().cpu()[0,0,:,:].squeeze())
        plt.show()

        # For each thing in the batch    
        for idx in range(inputs.shape[0]):   
            #idx = 0
            
            # Synthesize the image series! Do this in NP
            sig_true = inputs.detach().cpu()[idx,:,:,:,0].numpy()
            
            S0_pred = np.abs(outputs.detach().cpu()[idx,0, :,:].numpy())
            T_pred = np.abs(outputs.detach().cpu()[idx,1, :,:].numpy())
            
            eta = get_eta()
            sig_pred = np.zeros([len(eta), 128, 128])
            for idx, e in enumerate(eta):
                sig_pred[idx,:,:] = np.abs(S0_pred * np.exp(-e/(T_pred+1e-16)))
                              
            
            # NP doesn't have permute
            sig_pred = np.moveaxis(sig_pred, 0, 2)
            sig_true = np.moveaxis(sig_true, 0, 2)
            diff = sig_true - sig_pred
            
            #imshow_montage(sig_true, limits=[0,1])
            #imshow_montage(sig_pred, limits=[0,1])
            #imshow_montage(diff, limits=[-.2,.2])
            
            mse = np.mean(diff * diff)
            print(f'mse {mse}')
            
            fig, axs = plt.subplots(2,2, figsize=[6,6])
            axs[1,0].imshow(S0_pred, vmin=0, vmax=1)
            axs[1,1].imshow(T_pred, vmin=0, vmax=2)
            for x in range(2):
                for y in range(2):
                    axs[x,y].set_xticks([])
                    axs[x,y].set_yticks([])
            axs[0,0].set_ylabel('True')
            axs[1,0].set_ylabel('Pred')
            axs[1,0].set_xlabel('S0')
            axs[1,1].set_xlabel('T')
            plt.show()












