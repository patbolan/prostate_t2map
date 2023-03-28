#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:08:00 2022

Make an example figure comparing the predictions for multiple methods at 
different noise levels

To reproduce the paper figure, run this 2x, changing the boolean in the if statement


@author: pbolan
"""

from utility_functions import *
from os import path
import glob 
import nibabel as nib
from inference import get_inference_dataloader, estimate_exp_decay
import matplotlib.pyplot as plt
import torch
from monai.utils import first

#%%
# datasets will be columns, methods will be rows
datasets = ['INVIVO2D_SET3', 'INVIVO2D_SET3_NOISE_2', 'INVIVO2D_SET3_NOISE_3', 'INVIVO2D_SET3_NOISE_4']
methods = ['FIT_NLLS_BOUND', 'CNN_IMAGENET', 'CNN_SS_INVIVO', 'NN1D_URAND', 'NN1D_SS_INVIVO']

methods = ['NN1D_URAND', 'CNN_IMAGENET', 'CNN_SS_INVIVO', ]
#methods = ['FIT_NLLS', 'NN1D_URAND', 'CNN_SS_INVIVO', ]



fig, axs = plt.subplots(len(datasets), len(methods))

# Save a copy of each for interactive analysis
mat_T = np.zeros([128,128,len(datasets), len(methods)])

for cidx, ds_name in enumerate(datasets):
    for midx, method in enumerate(methods):

        #loader = get_inference_dataloader(ds_name)
        loader, modification_code, modification_number = get_inference_dataloader(ds_name)

        # Grab the first
        #sample = first(loader)
        # This is a very slow way to pick one, but lets me use the dataloader
        # Nice examples are 601, 409, 236, 32
        sample = nth(loader, 400)
          
        imgseries = sample['image']
        
        # NOrmalization
        imgseries = imgseries / imgseries.max()
        
        # Optional dynamic modifications (not in loader)
        if modification_code == "NOISE":
            np.random.seed(6283)
            # Add COMPLEX Gaussian noise to the full series. 
            # Values are indpendent across time and between real/imag
            noiseimg_real = np.random.standard_normal(imgseries.shape)
            noiseimg_imag = np.random.standard_normal(imgseries.shape)
            noiseimg_cplx = torch.from_numpy(noiseimg_real + 1j * noiseimg_imag)  
            # TODO: figoure out how to scale noise so its in a realistic range
            #imgseries = np.abs(imgseries + noiseimg_cplx * 0.01*int(modification_number))
            imgseries = torch.abs(imgseries + noiseimg_cplx * 0.01*int(modification_number))
            
            # Re-normalize
            imgseries = (imgseries / imgseries.max() ).float()      
            
        elif modification_code == "LENGTH":
            # Shorten by discarding later echos
            imgseries = imgseries[:,:int(modification_number), :,:,:]
            
        elif modification_code == None:
            # That's fine, do nothing
            pass
        
        else:
            raise Exception(f"Unknown modification code {modification_code}")     
            
        # Pick a center slice before inference
        sl = np.round(imgseries.shape[4]/2).astype(np.int32) -1
        #sl = 16
        imgseries = imgseries[:,:,:,:,[sl]]       
        
        # imgseries has [Nbatch=1, Np, Nx, Ny, Nz]
        # convert it to [Nx, Ny, Nz, Np]        
        S0, T = estimate_exp_decay(imgseries[0,:,:,:,:].permute([1,2,3,0]), method) 
        
        # Bring it down to 2D
        S0 = S0[:,:,0]
        T = T[:,:,0]
        
        # Plot either results, or source images
        if True:
            im = axs[cidx, midx].imshow(np.fliplr(np.rot90(T, 1)), vmin=0, vmax=1.5, cmap='viridis', interpolation='none')
            axs[cidx, midx].set_axis_off()
        else:
            # This is a hack. Instead of plotting the result, show the first and
            # last source image. Use this for the paper figure, commenting ou
            if midx==0:
                tmp_img = np.fliplr(np.rot90(imgseries[0,0,:,:,0], 1))
            else:
                tmp_img = np.fliplr(np.rot90(imgseries[0,0-1,:,:,0], 1))
                
            im = axs[cidx, midx].imshow(tmp_img, vmin=0, vmax=0.4, cmap='gray', interpolation='none')
            axs[cidx, midx].set_axis_off()        
        
    
# Uncomment this to get a colorbar you can later manipulate
#fig.colorbar(im, cax=axs[0,0], orientation='vertical')

fig.tight_layout()
plt.show()
   



    