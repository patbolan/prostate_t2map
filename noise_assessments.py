#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:40:51 2022

Development.
Trying to figure out the SNR of the invivo data
Adding noise and re-evaluating

@author: pbolan
"""
from utility_functions import *
from inference import get_inference_dataloader

#%%
ds_name = 'INVIVO_SET1_NOISE_3'
loader, modification_code, number = get_inference_dataloader(ds_name)


#%%
all_pix_snr = np.array([])

# Iterate over all elements in the loader, perform inference, save
for idx, sample in enumerate(loader):
    
    imgseries = sample['image']
    
    fullfilename = sample['image_meta_dict']['filename_or_obj'][0]
    file_num_str = get_file_numstr_from_file_name(fullfilename)
    print(f'{file_num_str}, slices {imgseries.shape[4]}')
    

    # imgseries has [Nbatch=1, Np, Nx, Ny, Nz]
    # convert it to [Nx, Ny, Nz, Np]        
    #S0_pred, T_pred = estimate_exp_decay(imgseries[0,:,:,:,:].permute([1,2,3,0]), model_name)
    
    # Stopping here. Looks set up correctly, maybe its expecting 1 slice
    snr_est = estimate_snr(ds_name, file_num_str)
    
    all_pix_snr = np.append(all_pix_snr, snr_est.flatten())

#%%
plt.hist(all_pix_snr, range=[0, 50], bins=100)
plt.title('SNR histogram')
plt.show()
print(f'SNR 25/50/75% = {np.percentile(all_pix_snr,25):.1f} / {np.percentile(all_pix_snr,50):.1f} / {np.percentile(all_pix_snr,75):.1f}')

