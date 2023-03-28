#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:46:44 2022

@author: pbolan
"""
import numpy as np
from os import path, makedirs
import glob
import nibabel as nib
import torch
import time

from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    IntensityStatsd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRange,
    ScaleIntensityd,
    ScaleIntensity,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.data import CacheDataset, DataLoader, Dataset

from monai.transforms.transform import MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


from exp_fitting import fit_exp_linear, fit_exp_nlls_2p, fit_exp_nlls_2p_bound, fit_exp_rician
from utility_functions import *

import multiprocessing as mp

#%%
# Start simply: loop over the images
# eta and imgseries can be either numpy or torch arrays
# Image series needs to be a numpy array in the form [Nx, Ny, Nz, Ntimepts]
def fit_1d(eta, imgseries, model_name):
    
    [Nx, Ny, Nz, Ntimepts] = imgseries.shape    
    T = np.zeros([Nx, Ny, Nz])
    S0 = np.zeros([Nx, Ny, Nz])
    
    # Supports a few different types of 1D fits
    if model_name=='FIT_LOGLIN':
        fitfun = fit_exp_linear
    elif model_name=='FIT_NLLS':
        fitfun = fit_exp_nlls_2p
    elif model_name=='FIT_NLLS_BOUND':
        fitfun = fit_exp_nlls_2p_bound
    elif model_name=='FIT_NLLS_RICE':
        fitfun = fit_exp_rician
    else:
        raise Exception(f'ERROR: method <{model_name}> not recognized')

    # For loops! Parallelize for more speed
    for xdx in range(Nx):
        for ydx in range(Ny):
            for zdx in range(Nz):
                timeseries = imgseries[xdx,ydx,zdx,:]
                if timeseries[0] > 0:
                    fitstruct = fitfun(eta, timeseries)   
                    T[xdx, ydx, zdx] = fitstruct['T']
                    S0[xdx, ydx, zdx] = fitstruct['S0']

    return S0, T

#%%
# Start simply: loop over the images
# eta and imgseries can be either numpy or torch arrays
# Image series needs to be a numpy array in the form [Nx, Ny, Nz, Ntimepts]
def fit_1d_parallel(eta, imgseries, model_name):
    
    [Nx, Ny, Nz, Ntimepts] = imgseries.shape    
    T = np.zeros([Nx, Ny, Nz])
    S0 = np.zeros([Nx, Ny, Nz])
    
    # Supports a few different types of 1D fits
    if model_name=='FIT_LOGLIN':
        fitfun = fit_exp_linear
    elif model_name=='FIT_NLLS':
        fitfun = fit_exp_nlls_2p
    elif model_name=='FIT_NLLS_BOUND':
        fitfun = fit_exp_nlls_2p_bound
    elif model_name=='FIT_NLLS_RICE':
        fitfun = fit_exp_rician
    else:
        raise Exception(f'ERROR: model <{model_name}> not recognized')


    # First, flatten the first 3 dims
    data = imgseries.reshape([Nx*Ny*Nz, Ntimepts]).numpy()    
    
    pool = mp.Pool(12)
    results = pool.starmap(fitfun, [(eta, row) for row in data])
    pool.close()

    # Extract results as list, convert to numpy, reshape    
    tmp = [result['S0'] for result in results]
    S0 = np.reshape(np.asarray(tmp), [Nx, Ny, Nz]).astype(np.float32)
    tmp = [result['T'] for result in results]
    T = np.reshape(np.asarray(tmp), [Nx, Ny, Nz]).astype(np.float32)

    return S0, T

#%% 
# Inference, 1D.
# Lke fit_1d, but we'll load up the model once for efficiency
# Could also do it on GPU

def infer_1d(eta, imgseries, model_name):
    
    model_root = get_models_dir()

    if model_name=='NN1D_URAND':
        model_scripted_filename = path.join(model_root, 'nn1d_urand.pt')
    elif model_name=='NN1D_IMAGENET':
        model_scripted_filename = path.join(model_root, 'nn1d_imagenet.pt')
    elif model_name=='NN1D_SS_URAND':
        model_scripted_filename = path.join(model_root, 'nn1d_ss_urand.pt')
    elif model_name=='NN1D_SS_IMAGENET':
        model_scripted_filename = path.join(model_root, 'nn1d_ss_imagenet.pt')
    elif model_name=='NN1D_SS_INVIVO':
        model_scripted_filename = path.join(model_root, 'nn1d_ss_invivo.pt')
    else:
        raise Exception(f'ERROR: model <{model_name}> not recognized')
    
    model = torch.jit.load(model_scripted_filename)
    model.eval()

    # CPU is just as fast for one at a time
    #device_cpu = torch.device('cuda:0')
    device_cpu = torch.device('cpu')
    model.to(device_cpu)
    
    # Can run the multiple spatial positions like a batch inference
    [Nx, Ny, Nz, Ntimepts] = imgseries.shape
    outputs = model(imgseries.reshape([Nx*Ny*Nz, Ntimepts]))
    
    S0 = outputs[:,0].reshape([Nx, Ny, Nz]).detach().numpy()
    T = outputs[:,1].reshape([Nx, Ny, Nz]).detach().numpy()
    return S0, T

#%%
def infer_2d(eta, imgseries, model_name='CNN_IMAGENET'):

    model_root = get_models_dir()
    
    if model_name=='CNN_IMAGENET':
        model_scripted_filename = path.join(model_root, 'cnn_imagenet.pt')
    elif model_name=='CNN_URAND':
        model_scripted_filename = path.join(model_root, 'cnn_urand.pt')
    elif model_name=='CNN_SS_IMAGENET':
        model_scripted_filename = path.join(model_root, 'cnn_ss_imagenet.pt')
    elif model_name=='CNN_SS_URAND':
        model_scripted_filename = path.join(model_root, 'cnn_ss_urand.pt')
    elif model_name=='CNN_SS_INVIVO':
        model_scripted_filename = path.join(model_root, 'cnn_ss_invivo.pt')
    elif model_name=='CNN_IMAGENET':
        model_scripted_filename = path.join(model_root, 'cnn_imagenet.pt')
    else:
        raise Exception(f'ERROR: model <{model_name}> not recognized')

    model = torch.jit.load(model_scripted_filename)
    model.eval()
    
    #device = torch.device('cuda:0')
    device = torch.device('cpu')
    model.to(device)
    
    # Can run the multiple spatial positions like a batch inference
    # Model operates on 2D inputs, expects [Nbatch, Np, Nx, Ny]
    # Here there is no batch, but we'll put the z-dim in front to process the
    # slices as different memberes of a batch
    [Nx, Ny, Nz, Np] = imgseries.shape
    inputs = imgseries.permute([2,3,0,1])
    inputs = inputs.to(device)
    outputs = model(inputs)
    
    # Flip the batch dimension back to Nz (slice)
    outputs = outputs.permute([2, 3, 0, 1]).detach().cpu().numpy()
    S0 = outputs[:,:,:,0]
    T = outputs[:,:,:,1]
    
    return S0, T
    

#%% Inference/Estimation/prediction code
# Takes a time series as input, predicts S0 and T
# imgseries must be [Nx, Ny, Nz, Np], torch array, on cpu
# Returns S0 and T in [Nx, Ny, Nz], as numpy array
# All operations are 2D, and will look over the 3rd (Nz) direction
def estimate_exp_decay(imgseries, model_name):
    
    eta = get_eta()
    eta = eta[:imgseries.shape[3]] # shorten eta to match imgseries length
    
    fit_1d_models = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE']
    infer_1d_models = ['NN1D_URAND', 'NN1D_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_INVIVO']
    infer_2d_models = ['CNN_URAND', 'CNN_IMAGENET', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO', ]

    if model_name in fit_1d_models:    
        #S0, T = fit_1d(eta, imgseries, model_name=model_name)
        S0, T = fit_1d_parallel(eta, imgseries, model_name=model_name)
    elif model_name in infer_1d_models:    
         S0, T = infer_1d(eta, imgseries, model_name=model_name)
    elif model_name in infer_2d_models:    
         S0, T = infer_2d(eta, imgseries, model_name=model_name)
    else:
        raise Exception(f'ERROR: model <{model_name}> not recognized')

         
    # Abs. Training can create negative estimates. Apply this to all methods
    S0 = np.abs(S0)
    T = np.abs(T)
              
    return S0, T



#%%
# For scaling, we want to normalize the image (ie divide by max)
# AND perform that same scaling on the S0 label. Not the T label!
class ScaleTransformSpecial:
    def __call__(self, d):
        
        # Scale the image
        max_val = d['image'].max()
        d['image'] = d['image'] / max_val        
        
        if 'label' in d:
            # Scale the S0 (first component) label
            d['label'][0] = d['label'][0] / max_val
            
        return d

#%%
# TODO: I think we never need labels for this
def get_inference_dataloader(ds_name):
    
    # Find source images
    source_root = get_datasets_dir() 
    #dataset_dir, is_validation_split, is_synthetic = parse_dataset_name(ds_name)
    dataset_dir, is_validation_split, is_synthetic, modification_code, number = parse_dataset_name(ds_name)

    
    # First make lists of all images and labels needed, put them in a dictionary
    all_images = sorted(glob.glob(path.join(source_root, dataset_dir, 'images', '*nii.gz')))
    if is_synthetic:
        all_labels = sorted(glob.glob(path.join(source_root, dataset_dir, 'labels', '*nii.gz')))
        test_dict = [{"image": image_name, "label": label_name} 
                     for image_name, label_name in zip(all_images, all_labels)]
    else:        
        test_dict = [{"image": image_name} for image_name in all_images]
      
    # Only take the last 20% if this is from a "training" dataset    
    if is_validation_split:
        split_index = int(0.8 * len(test_dict))
        #_, test_files = test_dict[-split_index:], test_dict[:-split_index]
        _, test_files = test_dict[:split_index], test_dict[split_index:]
    else:
        # Use all files. This is for test datasets
        test_files = test_dict        

    print(f'Inferring on {len(test_files)} files from {dataset_dir}')

    # Prepare a loader for the dataset, using MONAI's infrastructure
    # Different if the labels are needed
    # HACK Adding intensity scaling here! Defaults to [0,1]
    if is_synthetic:
        # Note we don't scale if synthetic, because we'd need to scale S0  equally.
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(
                    1., 1.), mode=("bilinear", "bilinear")),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
                EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            ]
        )
    else: 
        # Since we have no labels we can Scale to [0,1]
        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1., 1.), mode=("bilinear")),
                EnsureTyped(keys=["image"], dtype=torch.float32),
            ]
        )
    num_workers=4
    #test_ds = CacheDataset( data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=num_workers)
    test_ds = Dataset( data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)
    return test_loader, modification_code, number



#%%
def infer_parameters(method, ds_name):
    
    # Prepare output area
    output_dir = path.join(get_predictions_dir(), ds_name, method)
    if path.exists(output_dir):
        print(f"Inference already exists for: \n  {output_dir}\nSkipping.")
        return
    else:
        print(f'Creating output directory {output_dir}')
        makedirs(output_dir)
        
    test_loader, modification_code, modification_number = get_inference_dataloader(ds_name)

    # Iterate over all elements in the loader, perform inference, save
    for idx, sample in enumerate(test_loader):
        
        imgseries = sample['image']
        # imgseries has [Nbatch=1, Np, Nx, Ny, Nz]
        # convert it to [Nx, Ny, Nz, Np]        
        imgseries = imgseries[0,:,:,:,:].permute([1,2,3,0])
        
        # Normalize on a per series level
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
            imgseries = torch.abs(imgseries + noiseimg_cplx * 0.01*int(modification_number))
            
            # Re-normalize
            imgseries = (imgseries / imgseries.max()).float()  
            
        elif modification_code == "LENGTH":
            # Shorten by discarding later echos
            imgseries = imgseries[:,:,:,:int(modification_number)]
            
        elif modification_code == None:
            # That's fine, doe nothing
            pass
        
        else:
            raise Exception(f"Unknown modification code {modification_code}")            
            
        
        S0_pred, T_pred = estimate_exp_decay(imgseries, method)
                
        # Get output filenames
        srcimage_fname = path.basename(sample['image_meta_dict']['filename_or_obj'][0])
        base, file_num_str = split_filename(srcimage_fname)
        
        # Recombine S0 and T into a nifti file
        preds = np.stack([S0_pred, T_pred], 3)
        fname = path.join(output_dir, f'preds_{file_num_str}.nii.gz')
        save_nifti(preds, fname)
        
        # Write the output files
        print(f'Writing out {fname}')


#%%e
def perform_partA_inference():
    # Time the probolem
    start = time.time()
    datasets = get_partA_datasets()
    methods = get_partA_methods()
    
    for dataset in datasets:
        for method in methods:
            infer_parameters(method = method, ds_name = dataset)


    end = time.time()
    print(f'Part A Inference in {end-start:.2f} seconds')

def perform_partB_inference():
    # Same as A, different dataset
    start = time.time()
    ds_names = ['INVIVO2D_SET3', 'INVIVO2D_SET1'] # Set 1 is only for demo/testing
    methods = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE',
               'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_INVIVO',
               'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO']
    # For time's sake, reduce the dataset a bit
    methods = ['FIT_NLLS', 
               'NN1D_URAND', 'NN1D_SS_INVIVO',
               'CNN_IMAGENET', 'CNN_SS_INVIVO']


    for dataset in ds_names:
        for method in methods:
            infer_parameters(method = method, ds_name = dataset)


    end = time.time()
    print(f'Part B Inference in {end-start:.2f} seconds')

def perform_partC_inference():
    # This will do the noise variations 
    start = time.time()
    methods = ['FIT_NLLS', 
               'NN1D_URAND', 'NN1D_SS_INVIVO',
               'CNN_IMAGENET', 'CNN_SS_INVIVO']
    for method in methods:
        infer_parameters(method=method, ds_name='INVIVO2D_SET3')
        for idx in np.arange(1,10):
            print(f'Inferring for method {method} on ds INVIVO2D_SET3_NOISE_{idx}')
            infer_parameters(method=method, ds_name=f'INVIVO2D_SET3_NOISE_{idx}')    
        
    end = time.time()
    print(f'Part C Inference in {end-start:.2f} seconds')


if __name__=='__main__':
    perform_partA_inference()
    perform_partB_inference()
    perform_partC_inference()
    
    # For figure 1, I need to infer FIT_NLLS for set 1    
    #infer_parameters(method='FIT_NLLS', ds_name='INVIVO2D_SET1')
    
    # One at a time, for testing
    #infer_parameters(method='CNN_IMAGENET', ds_name='INVIVO2D_SET1')
    #infer_parameters(method='CNN_IMAGENETK4', ds_name='INVIVO2D_SET1')

