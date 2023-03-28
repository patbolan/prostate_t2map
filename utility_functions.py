#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build synthetic datasets representing a T2 mapping series (10 evenly spaced TEs) using 
imagenet images as the true T2 and S0 maps
S(TE) = S0 * exp(-TE/T2)
    
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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi
from skimage.measure import blur_effect


#%% Project paths are defined here, only once
def get_base_dir(): 
    return '/home/pbolan/dev/prostate_t2map'

def get_plot_dir():
    return path.join(get_base_dir(), 'plots')

def get_datasets_dir():
    return path.join(get_base_dir(), 'datasets')

def get_models_dir():
    return path.join(get_base_dir(), 'models')

def get_predictions_dir():
    return path.join(get_base_dir(), 'predictions')


#%%
def get_eta():
    Np = 10 # number of TE/eta values
    TE = np.linspace(26.4, 145.2, Np) # ms
    eta = TE / TE.max() 
    return eta


def get_summary_stats_string(arr):
    tmp = arr.flatten()
    str = f'range=[{tmp.min():.2f}, {tmp.max():.2f}]'
    str += f', mean/med={np.mean(arr):.2f}/{np.median(arr):.2f} '
    return str

#%%
def save_nifti(img3d, fname):
    img = nib.Nifti1Image(img3d, np.eye(4))
    nib.save(img, fname)

#%% Takes all the files in a folder, assuming they're 2D, and make a single 3D
# file called all.nii.gz. Or pass in a filename. 
# Useful for combining results for review
def combine_files(dirname, outfile=None):
    
    # Load all the files into a single thing
    filelist = sorted(glob.glob(path.join(dirname, '*_*nii.gz')))
    
    # Load the first one to get sizes
    tmp = nib.load(path.join(dirname, filelist[0])).get_fdata()      
    img = np.zeros([tmp.shape[0], tmp.shape[1], len(filelist), tmp.shape[3]])

    for idx, fname in enumerate(filelist):
        tmp = nib.load(path.join(dirname, fname)).get_fdata()      
        img[:,:,idx,:] = tmp[:,:,0,:]

    # Cap the largest images
    img[img>10] = 10
    
    if outfile==None:
        outfile = path.join(dirname, 'allfiles.nii.gz')
    
    save_nifti(img, outfile)
    print(f'saved {outfile}')


#%%
def split_filename(fname):
    # filenames are BLAHBLAH_000000.nii.gz
    # first take off the .nii.gz
    basename, _, _ = fname.split('.')
    typename, file_num_str = basename.split('_')
    return typename, file_num_str
    


#%%
def get_file_numstr_from_file_name(fname):
    # There are two conventions. 
    # in vivo has only 3 digits (001), but synthetics have way more 
    if fname[-11] == '_':
        # invivo
        return fname[-10:-7]
    else:
        # synthetic
        return fname[-13:-7] 
                
#%% Imagenet access
def get_imagenet_root():
    # TODO: change this location for your system!
    # Inside this folder you should find ILSVRC2012_val_00000001.JPEG, etc
    # Download from https://www.kaggle.com/datasets/samfc10/ilsvrc2012-validation-set
    return '/mnt/data/ReferenceDatasets/imagenet_sm/Images/imagenet'


# Reads in an image from the imagenet folder, normalizes, crops
# Valid images are 1 to 50000
def read_and_transform_imagenet_image(img_number, Nsize=128):
    imagenet_root = get_imagenet_root()
    fname = path.join(imagenet_root, f'ILSVRC2012_val_{img_number:08d}.JPEG')
    
    # Read and convert to grayscale
    img = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) # convert to float
    
    # Add small randominzation to de-quantize. Images are [0,255], so std is 1
    rng = np.random.default_rng()
    img = img + rng.standard_normal(size=img.shape)
    
    # Crop to a Nsize x Nsize image.
    out = np.zeros((Nsize, Nsize))
    N_smallest = np.min(img.shape)
    if N_smallest<Nsize:
        # Copy what you can, leave the rest as zeros
        
        Nx = np.min([Nsize, img.shape[0]])
        Ny = np.min([Nsize, img.shape[1]])

        # Place it in center        
        x_start = int(Nsize/2 - Ny/2)
        y_start = int(Nsize/2 - Nx/2)
        out[y_start:y_start+Nx, x_start:x_start+Ny] = img[:Nx, :Ny]
        
    else:
        # Center crop
        x_start = int(img.shape[1]/2 - Nsize/2)
        y_start = int(img.shape[0]/2 - Nsize/2)
        
        out = img[y_start:y_start+Nsize, x_start:x_start+Nsize]

    # Normalize  
    out = out - out.min()      
    if out.max() <= 0:
        # Have an image that is just zero
        print(f'BAD IMAGE {img_number}, {fname}')
        out = out * 0 + 1 # Don't know if this will work!?
    else:
        out = out / out.max() # normalize

        
    return out


#%% 
# From the dataset name convention, finds the location on disk
# Synthetic dataset names have labels, are like: URAND_VALIDATION_1k
# the source directory is different, like synth_urand_1k_train
# 
# Measured datsets do not have labels
def parse_dataset_name(ds_name):
    # Parse the ds name
    parts = ds_name.split('_')
    part_a = parts[0]

    # Defaults
    modification_code = None
    number = 0
    
    if part_a == 'URAND':
        root = 'synth_urand'
        is_synthetic = True
    elif part_a == "IMAGENET":
        root = 'synth_imagenet'
        is_synthetic = True
    elif part_a == "PHANTOM":
        root = 'phantom'
        is_synthetic = False 
    elif part_a == "INVIVO2D":
        root = 'invivo2D' 
        is_synthetic = False 
    else:
        print('dataset name not recognized.')
        return

    # The naming conventions are different for measured and synthetic
    if is_synthetic:
        usage_code = parts[1]
        size_code = parts[2]   

        # Have form type_source_number_usage, eg synth_urand_10k_test
        if usage_code == 'VALIDATION':
            suffix = "train"
            is_validation_split = True
        elif usage_code == "TEST":
            suffix = "test"
            is_validation_split = False

        dataset_dir = f'{root}_{size_code}_{suffix}'  
                
    else:
        # Have form source_set_variant_number. 
        # eg, INVIVO_SET1_NOISE_3, PHANTOM_SET1_ETAS_4
        
        set_number = parts[1][-1] # extract the number from the set
        
        is_validation_split = False
        dataset_dir = f'{root}_set{set_number}' # 
        if len(parts)>2:
            modification_code = parts[2]
            number = parts[3]
    
    return dataset_dir, is_validation_split, is_synthetic, modification_code, number

#%% 
def parse_dataset_name_OLDER(ds_name):
    # Parse the ds name
    parts = ds_name.split('_')
    part_a = parts[0]
    part_b = parts[1]
    
    is_synthetic = True # Synthetic data has labels
    if part_a == 'URAND':
        root = 'synth_urand'
    elif part_a == "IMAGENET":
        root = 'synth_imagenet'
    elif part_a == "PHANTOM":
        root = 'phantom'
        is_synthetic = False # invivo data has no label
    elif part_a == "INVIVO":
        root = 'invivo'
        is_synthetic = False # invivo data has no label

    if part_b == 'VALIDATION':
        suffix = "train"
        is_validation_split = True
    elif part_b == "TEST":
        suffix = "test"
        is_validation_split = False
          
    if len(parts)>2:
        size_code = parts[2]    
        dataset_dir = f'{root}_{size_code}_{suffix}'  
    else:
        dataset_dir = f'{root}_{suffix}'  
    
    return dataset_dir, is_validation_split, is_synthetic

#%%
def get_evaluation_paths(model_name, ds_name): 
       
    output_dir = path.join(get_base_dir(), 'predictions', ds_name, model_name)
    
    dataset_dir, _, _, _, _= parse_dataset_name(ds_name)

    ds_labels_dir = path.join(get_datasets_dir(), f'{dataset_dir}', 'labels')   
    ds_images_dir = path.join(get_datasets_dir(), f'{dataset_dir}', 'images')   
    
    return output_dir, ds_labels_dir, ds_images_dir

#%% 
def get_noisefile(ds_name):
    dataset_dir, _, _, _, _= parse_dataset_name(ds_name)
    return path.join(get_datasets_dir(), f'{dataset_dir}', 'noisevalues.csv')   
    

#%%
# Helper function to plot a montage of a 3D image set
# TODO: have it return a 2D image and plot it yourself
def imshow_montage(img3d, limits=None):
    n_images = img3d.shape[2]
    n_cols = int(np.ceil(np.sqrt(n_images)))
    n_rows = int(np.ceil(n_images/n_cols)) 
    
    fig = plt.figure()
    for idx in range(n_images):
        plt.subplot(n_rows, n_cols, idx+1)
        if limits is None:
            plt.imshow(np.squeeze(img3d[:,:,idx]))
        else: 
            plt.imshow(np.squeeze(img3d[:,:,idx]), vmin=limits[0], vmax=limits[1])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


#%%
def show_model_details(model):
    print(model)
    num_model_params = sum([param.nelement() for param in model.parameters()])
    print(f'Total parameters in model: {num_model_params}')
    
#%%
# Calculates a treshold based on the iamge histogram
def image_threshold_percentage(img, percentage):
    vals = np.sort(img.flatten())
    
    # Discard the zeros from the calculation
    # This is only relevant in synthetic cases, otherwise noise is non-zero
    vals = vals[vals>0]
    
    threshold_index = int(vals.shape[0]* percentage/100)
    return vals[threshold_index]


#%% Calculates the SNR using tradiational EE definition. 
# If the true noise sigma is not provided, it can estimate using std(residual)
# SNRdB = 10 log10 (sum(S(TE)^2) / (sigma^2 * #TE))
def estimate_snr_db(ds_name, file_num_str, sigma=None):
    eta = get_eta()
    Np = len(eta)
    
    # Get all the output files for FIT_NLLS
    output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths('FIT_NLLS', ds_name)

    # Get the prediction image    
    files_found = glob.glob(path.join(output_dir, f'*{file_num_str}.nii.gz'))
    if len(files_found) < 1:
        raise Exception('ERROR: cannot find expected file. Perhaps inference incomplete?')     
    pred_file = files_found[0]
    
    # If we could find a prediction, we probably have the label and images.
    image_file = glob.glob(path.join(ds_images_dir, f'*{file_num_str}.nii.gz'))[0]
    
    # Load files
    img_pred = nib.load(pred_file).get_fdata()
    img_orig = nib.load(image_file).get_fdata()
    
    # If you have a sigma, use it to calculate without scaling
    if sigma is not None:
        sum_square = np.sum(np.power(img_orig,2),3)        
        snrdb = 10 * np.log10( sum_square / (sigma*sigma * Np))    
        
    else:
        # If we don't actually know sigma, fit with NLLS, use the signal model 
        # calculate S and the residual to estimate sigma
        
        # Normalize, because predictions were based on normalized data
        img_orig = img_orig / img_orig.max()
        
        # Simulate the data from the fit S0, T
        S0_pred = img_pred[:,:,:,0]
        T_pred = img_pred[:,:,:,1]
        [Nx, Ny, Nz] = S0_pred.shape
        sim_series = np.zeros([Nx, Ny, Nz, Np])
        for jdx in range(Np):
            sim_series[:,:,:, jdx] = S0_pred * np.exp(-eta[jdx] / T_pred)
         
        # Calculate residual, take abs to make it Rician 
        residual_series = sim_series - img_orig
        residual_series = np.abs(residual_series)
        
        res_std = np.std(residual_series, axis=3, ddof=1.5)
        signal_ave_est = np.mean(sim_series[:,:,:,:], 3)
        
        sigma = res_std
        sum_square = np.sum(np.power(sim_series,2),3)        
        snrdb = 10 * np.log10( sum_square / (sigma*sigma * Np))         
    
    return snrdb


#%% 
def estimate_snr(ds_name, file_num_str):
    eta = get_eta()
    
    # We're going to need the FIT_NLLS data for this estmation
    model_name = 'FIT_NLLS'
    
    # Get all the output files
    output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths(model_name, ds_name)

    # Get the prediction image    
    files_found = glob.glob(path.join(output_dir, f'*{file_num_str}.nii.gz'))
    if len(files_found) < 1:
        raise Exception('ERROR: cannot find expected file. Perhaps inference incomplete?')     
    pred_file = files_found[0]
    
    # If we could find a prediction, we probably have the label and images.
    image_file = glob.glob(path.join(ds_images_dir, f'*{file_num_str}.nii.gz'))[0]
    
    # Load files
    img_pred = nib.load(pred_file).get_fdata()
    img_orig = nib.load(image_file).get_fdata()
    
    # Normalize, because predictions were based on normalized data
    img_orig = img_orig / img_orig.max()
    
    # Simulate the data from the fit S0, T
    Np = len(eta)
    S0_pred = img_pred[:,:,:,0]
    T_pred = img_pred[:,:,:,1]
    [Nx, Ny, Nz] = S0_pred.shape
    sim_series = np.zeros([Nx, Ny, Nz, Np])
    for jdx in range(Np):
        sim_series[:,:,:, jdx] = S0_pred * np.exp(-eta[jdx] / T_pred)
     
    # Calculate residual, take abs to make it Rician 
    residual_series = sim_series - img_orig
    residual_series = np.abs(residual_series)
    
    res_std = np.std(residual_series, axis=3, ddof=1.5)
    signal_ave_est = np.mean(sim_series[:,:,:,:], 3)


    # HACK
    # Playing with my definition of SNR
    #snr_est = signal_ave_est / (res_std*0.655)
    snr_est = S0_pred / (res_std*0.655)
    
    return snr_est

#%%    
def plot_training_history(history):
    # Pull out the data you want
    train_loss = history['train_loss']
    validation_loss = history['validation_loss']
    val_interval = 1 # Have to assume this
    
    train_epochs = [i + 1 for i in range(len(train_loss))]
    val_epochs = [val_interval * (i + 1)  for i in range(len(validation_loss))]
    
    # Plot
    plt.plot(train_epochs, train_loss, '-b', label = 'training loss')
    plt.plot(val_epochs, validation_loss, '-r', label = 'validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, np.min(validation_loss)*4) # Sometimes first training losses bad
    plt.show()
    print(f'Min training loss {np.min(train_loss)}')
    print(f'Min validation loss {np.min(validation_loss)}')

# LIke Monai's first, but gets the nth
def nth(iterable, nth=0, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for idx, i in enumerate(iterable):
        if idx<nth:
            continue
        return i
    return default


#%%
def compare_images(img_test, img_ref, mask=None):

    # Also mask the images to use skimage methods
    if not isinstance(mask, np.ndarray):   
        mask = np.full(img_test.shape, True)
 
    # These are flattened arrays, not images. 
    # mask=False values do not contribute
    # Makes a difference when averaging.
    img_vals = img_test[mask]
    ref_vals = img_ref[mask] 
    diff_vals = img_vals - ref_vals
    median_relative_diff = np.median(diff_vals / (ref_vals+1e-16))
    median_relative_err = np.median( np.abs(diff_vals)/ (ref_vals+1e-16))
    
    mse = np.mean( diff_vals * diff_vals)
    mae = np.mean(np.abs(diff_vals))
    # NMRSE with Euclidean norm. See skimage ref for details
    nrmse = np.sqrt(mse) / np.sqrt(np.mean(ref_vals * ref_vals))   
    
    # Crete-Roffet Blur metric
    test_blur = blur_effect(img_test)
    ref_blur = blur_effect(img_ref)
    # This is the relative blur metric
    rcrbm = test_blur - ref_blur
      
    metrics = {'mae'       : mae, 
               'medae'     : np.median(np.abs(diff_vals)),
               'medbias'   : np.median(diff_vals),
               'mse'       : mse, 
               'nrmse'     : nrmse,
               'nmi'       : nmi(img_test, img_ref), 
               'ssim'      : ssim(img_test, img_ref), 
               'medreldif' : median_relative_diff, 
               'medrelerr' : median_relative_err,
               'rcrbm'     : rcrbm }
    return metrics

#%% Use these to adjust for the train/infer/eval cycle

def get_partA_datasets():
    # Used validation for development phases
    #datasets = ['IMAGENET_VALIDATION_1k', 'URAND_VALIDATION_1k']
    datasets = ['IMAGENET_TEST_1k', 'URAND_TEST_1k']

    return datasets

def get_partA_methods():

    methods = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE',
               'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_INVIVO',
               'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO', ]
               #'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO']

    # # For figure 3 
    # methods = ['FIT_NLLS', 
    #            'NN1D_URAND', 
    #            'CNN_IMAGENET','CNN_SS_INVIVO']
    return methods
    
def get_partC_datasets():
    datasets = ['INVIVO_VALIDATION']
    return datasets

def get_partC_methods():
    methods = ['FIT_NLLS_BOUND', 'NN1D_URAND', 'CNN_IMAGENET', ]
    return methods
        
    
    

