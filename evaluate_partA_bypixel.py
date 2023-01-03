#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:05:00 2022

Averaging over a slice may hide important biases.
I want to see accuracy (B/P) as a function of T2 and SNR
Do this by gather a structure of all pixel data
Bin them, by T2 and SNR groups
Plot with mean and std, for bias/precision

May have to go to patches
May have to subsample, the datasize is large
Not trying to load all the data at once, there are tens of millions of pixels

@author: pbolan
"""

import numpy as np
from os import path, makedirs
import sys
import glob
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm
from utility_functions import *

import seaborn as sns
import palettable.scientific.diverging as psgcm
import cmasher as cmr

#%%
# If ref_type=='labels', it will use the labels to find the "true" or reference values
# if ref_type=='FIT_NLLS' (or some other method) it will load those data as reference
def extract_all_pixels(ds_name, method_name, ref_type='labels'):
    # Make a big array, with columns [S0_ref, S0_pred, T_ref, T_pred, SNR]
    biga = np.zeros([0,5])
    
    # Get all the output files
    output_dir, ds_labels_dir, _ = get_evaluation_paths(method_name, ds_name)
    if not ref_type == 'labels':
        ref_output_dir, _, _ = get_evaluation_paths(ref_type, ds_name)
    
    # Load up the csv file with the corresponding noise levels
    noise_level_file = get_noisefile(ds_name)
    if path.isfile(noise_level_file):
        true_noise_value_available = True
        df_noise = pd.read_csv(noise_level_file)
        
        
    
    pred_files = sorted(glob.glob(path.join(output_dir, '*.nii.gz')))
    for idx, pred_file in enumerate(pred_files):
                        
        # extract the case number, find matching file in ds_labels_dir
        file_num_str = get_file_numstr_from_file_name(pred_file)        
        print(f'{ds_name}, {method_name}, {idx}, {file_num_str}')
        
        # Get the reference data from labels or a 
        if ref_type == 'labels':
            files_found = glob.glob(path.join(ds_labels_dir, f'*{file_num_str}.nii.gz'))
            if len(files_found) < 1:
                print('ERROR: cannot find expecteded file. Perhaps inference incomplete?')               
            label_file = files_found[0]
        
        else:
            label_file = path.join(ref_output_dir, f'preds_{file_num_str}.nii.gz')
        
        # Load files
        img_ref = nib.load(label_file).get_fdata()
        img_pred = nib.load(pred_file).get_fdata()
        
        # Add some foreground masking. Gets rid of zeros, and bottom 1%
        thr = image_threshold_percentage(img_ref[:,:,:,0], 1)
        
        # For the paper I've decided the masking is not necessary
        thr = 0 # Turns off masking 
        fgmask = img_ref[:,:,:,0] >= thr
        
        # Don't want to set them to zero, just remove these points from analysis
        S0_ref = img_ref[:,:,:,0]
        S0_pred = img_pred[:,:,:,0]
        T_ref = img_ref[:,:,:,1]
        T_pred = img_pred[:,:,:,1]    
    
        # Here's the old, non dB metric for SNR
        #snr = estimate_snr(ds_name, file_num_str)

        # Estimate snr. 
        # A quick evaluation shows that using true vs estimated sigma is pretty close
        if true_noise_value_available:
            sigma_known = df_noise['0'][idx]
            snr = estimate_snr_db(ds_name, file_num_str, sigma_known)
        else:
            snr = estimate_snr_db(ds_name, file_num_str)
        
        # [S0_ref, S0_pred, T_ref, T_pred, SNR]
        tmpa = np.array([S0_ref[fgmask].flatten(), S0_pred[fgmask].flatten(), \
                         T_ref[fgmask].flatten(), T_pred[fgmask].flatten(), 
                         snr[fgmask].flatten()])
        tmpa = tmpa.transpose()
        if biga.shape[0] == 0:
            biga = tmpa
        else:
            biga = np.concatenate([biga, tmpa])
    
    print(f'Gathered {biga.shape[0]} pixels from {len(pred_files)} files')          
    return biga 
        

#%%
# Expects the array [S0_ref, S0_pred, T_ref, T_pred, SNR]
def plot_error_by_snr_binned(biga, plot_fileroot, show_text=True):
          
    # Sort array on SNR
    sort_indices = biga[:,4].argsort()
    biga = biga[sort_indices,:] 
    
    # Only pick SNR in the range of interest
    max_val = 50
    smalla = biga[biga[:,4]<=max_val]
    num_small = smalla.shape[0]
    num_total = biga.shape[0]
    #print(f'{num_small}/{num_total} ({num_small/num_total*100:.2f}%) pixels have snr <={max_val}')

    # Calculate bin edges
    Nbins = 100
    bin_size = int(biga.shape[0] / (Nbins-1))
    
    ref_mean = np.zeros([Nbins])
    ref_median = np.zeros([Nbins])
    pred_mean = np.zeros([Nbins])
    pred_median = np.zeros([Nbins])
    pred_std = np.zeros([Nbins])
    pred_ub = np.zeros([Nbins])
    pred_lb = np.zeros([Nbins])
    
    err_mean = np.zeros([Nbins])
    err_median = np.zeros([Nbins])
    err_std = np.zeros([Nbins])
    err_ub = np.zeros([Nbins])
    err_lb = np.zeros([Nbins])
    
    snr_mean = np.zeros([Nbins])
    
    # For each bin, take means/medians
    for idx, bstart in enumerate(range(0, biga.shape[0], bin_size)):
        bend = np.min([biga.shape[0], bstart+bin_size])
        #print(f'{idx}: Bin from {bstart} to {bend}')
        T_ref = biga[bstart:bend, 2]
        T_pred = biga[bstart:bend, 3]
        err = T_pred - T_ref
        snr = biga[bstart:bend, 4]
    
        err_mean[idx] = np.mean(err)
        err_median[idx] = np.median(err)
        err_std[idx] = np.std(err)           
        err_ub[idx] = np.percentile(err, 75)
        err_lb[idx] = np.percentile(err, 25)
        
        pred_mean[idx] = np.mean(T_pred)
        pred_median[idx] = np.median(T_pred)
        pred_std[idx] = np.std(T_pred)  

        pred_ub[idx] = np.percentile(T_pred, 75)
        pred_lb[idx] = np.percentile(T_pred, 25)
        
        ref_mean[idx] = np.mean(T_ref)
        ref_median[idx] = np.median(T_ref)
        
        snr_mean[idx] = np.mean(snr)
                
    # Binned, with 95% bounds
    fig,ax = plt.subplots(figsize=[6,4])
    plt.plot(snr_mean, err_median, '-')
    plt.fill_between(snr_mean, err_lb, err_ub, alpha=0.4)
    plt.plot(snr_mean, snr_mean*0, '--r', linewidth=.5)
    plt.ylim([-2, 2])
    plt.xlim([0, max_val]) # Only interested in the low-SNR
    

    if show_text:
        plt.xlabel(f'SNR (dB) ({num_small/num_total*100:.2f}% of data)')
        plt.ylabel('T err')
        plt.legend(['median', 'IQR'])
    else:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        
    plt.show()
    if plot_fileroot is not None:
        plt.savefig(f'{plot_fileroot}_Terr_x_snr_binned.png')
        plt.savefig(f'{plot_fileroot}_Terr_x_snr_binned.svg')
    plt.close()


#%% 

# Expects the array [S0_ref, S0_pred, T_ref, T_pred, SNR]
def plot_error_by_T_binned(biga, plot_fileroot, show_text=True):           
    
    # Sort on T_ref
    sort_indices = biga[:,2].argsort()
    biga = biga[sort_indices,:]      

    # Calculate bin edges
    Nbins = 100
    bin_size = int(biga.shape[0] / (Nbins-1))
    
    ref_mean = np.zeros([Nbins])
    ref_median = np.zeros([Nbins])
    pred_mean = np.zeros([Nbins])
    pred_median = np.zeros([Nbins])
    pred_std = np.zeros([Nbins])
    pred_ub = np.zeros([Nbins])
    pred_lb = np.zeros([Nbins])
    
    err_mean = np.zeros([Nbins])
    err_median = np.zeros([Nbins])
    err_std = np.zeros([Nbins])
    err_ub = np.zeros([Nbins])
    err_lb = np.zeros([Nbins])
    
    for idx, bstart in enumerate(range(0, biga.shape[0], bin_size)):
        bend = np.min([biga.shape[0], bstart+bin_size])
        #print(f'{idx}: Bin from {bstart} to {bend}')
        T_ref = biga[bstart:bend, 2]
        T_pred = biga[bstart:bend, 3]
        T_err = T_pred - T_ref
    
        err_mean[idx] = np.mean(T_err)
        err_median[idx] = np.median(T_err)
        err_std[idx] = np.std(T_err)           
        err_ub[idx] = np.percentile(T_err, 75)
        err_lb[idx] = np.percentile(T_err, 25)      
    
        pred_mean[idx] = np.mean(T_pred)
        pred_median[idx] = np.median(T_pred)
        pred_std[idx] = np.std(T_pred)  
        pred_ub[idx] = np.percentile(T_pred, 75)
        pred_lb[idx] = np.percentile(T_pred, 25)
       
        ref_mean[idx] = np.mean(T_ref)
        ref_median[idx] = np.median(T_ref)


    # Binned versions, showing median and 95% of the range 
    # Also optional masking above
    # Also, keeping end bins or using all
    fig,ax = plt.subplots(figsize=[6,4])
    plt.plot(ref_median, err_median, '-')
    plt.fill_between(ref_median, err_lb, err_ub, alpha=0.4)
    plt.plot(ref_median, ref_median*0, '--r', linewidth=0.5)
    plt.ylim([-2,2])
    plt.xlim([-.1, 2.1])

    # I can see arguments extending this to 4, but I want to focus on 2
    #plt.xlim([-.1, 4.1])


    if show_text:
        plt.ylabel('T Err')
        plt.xlabel('T reference')
        plt.legend(['median', 'IQR'])
    else:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])    

    plt.show()  
    
    
    if plot_fileroot is not None:
        plt.savefig(f'{plot_fileroot}_Terr_x_Ttrue_binned.png')
        plt.savefig(f'{plot_fileroot}_Terr_x_Ttrue_binned.svg')
    plt.close()
    
    # This is an identity plot. Redundant with B-A style plot above
    if False:
        plt.plot(ref_median, pred_median, '-')
        plt.fill_between(ref_median, pred_lb, pred_ub, alpha=0.4)
        plt.plot([0,2], [0,2], '--r', linewidth=0.5)
        plt.ylim([-.1, 2.1])
        plt.xlim([-.1, 2.1])
        plt.ylabel('T pred')
        plt.xlabel('T reference')
        plt.legend(['median', 'IQR'])
        plt.show()  
        if plot_fileroot is not None:
            plt.savefig(f'{plot_fileroot}_Tpred_x_Ttrue_binned.png')
            plt.savefig(f'{plot_fileroot}_Tpred_x_Ttrue_binned.svg')
        plt.close()
            
#%%
# Plots the median T_err as a function of T and SNR, Binned
# TODO: extents, labels, bin sizes all hardwired
# Expects the array [S0_ref, S0_pred, T_ref, T_pred, SNR]
def plot_error_binned_2D(biga, plot_fileroot, show_text=True):    
    
    # Calculate bin start edges
    # SNR
    xbins = 50
    xmin = 0
    xmax = 40
    xbin_size = (xmax-xmin)/(xbins-1)
    xstart = np.arange(xmin, xmax, xbin_size)
    
    # T
    ybins = 60
    ymin = 0
    ymax = 2
    ybin_size = (ymax-ymin)/(ybins-1)
    ystart = np.arange(ymin, ymax, ybin_size)    

    # Now make a new data structure, [biga_index, x_index, y_index]
    print(f'Histogramming all pixels into {xbins} SNR bins, {ybins} T bins')
    z = np.zeros([biga.shape[0], 3], dtype=np.int32)
    for idx, row in enumerate(biga):
        snr = row[4]
        T_ref = row[2]
        T_pred = row[3]
        T_err = T_pred - T_ref
        
        # Calculate bin number. This is interpolation, nearest
        z[idx,0] = idx # shouldnt need this!
        z[idx,1] = get_histogram_bin_num(snr, xmin, xmax, xbins)
        z[idx,2] = get_histogram_bin_num(T_ref, ymin, ymax, ybins)


    
    # Now extract a count, median value, and IQR from each 2D bin
    T_err = biga[:,3] - biga[:,2]
    T_err_med = np.zeros([xbins, ybins])
    T_err_iqr = np.zeros([xbins, ybins])
    pix_count = np.zeros([xbins, ybins], dtype=np.int32)
    for xdx in range(xbins):
        for ydx in range(ybins):
            selected_indices = np.logical_and(z[:,1]==xdx, z[:,2]==ydx)
            vals = T_err[selected_indices]
            pix_count[xdx,ydx] = len(vals)
            if len(vals)>0:
                T_err_med[xdx,ydx] = np.median(vals)
                T_err_iqr[xdx,ydx] = np.percentile(vals,75) - np.percentile(vals,25)
    

    if False:
        # HACK! Tryu relative error
        T_relerr = (biga[:,3] - biga[:,2]) / biga[:,2]
        T_relerr_med = np.zeros([xbins, ybins])
        T_relerr_iqr = np.zeros([xbins, ybins])
        pix_count = np.zeros([xbins, ybins], dtype=np.int32)
        for xdx in range(xbins):
            for ydx in range(ybins):
                selected_indices = np.logical_and(z[:,1]==xdx, z[:,2]==ydx)
                vals = T_relerr[selected_indices]
                pix_count[xdx,ydx] = len(vals)
                if len(vals)>0:
                    T_relerr_med[xdx,ydx] = np.median(vals)
                    T_relerr_iqr[xdx,ydx] = np.percentile(vals,75) - np.percentile(vals,25)
   

    # Look at the counts like this:
    if False:
        _ = plt.hist(z[:,1], xbins)
        plt.show()
        _ = plt.hist(z[:,2], ybins)
        plt.show()
        #plt.imshow(pix_count, vmin=0, vmax=1000)
        plt.imshow(pix_count)
        plt.colorbar()
        plt.show()


    fig, ax = plt.subplots(1,1)
    #im = ax.matshow(T_err_med, vmin=-.5, vmax=.5, cmap='RdBu', alpha=1, interpolation='nearest')
    # Palettable colormaps with black centers are Berlin, Lisbon, Tofino
    #im = ax.matshow(T_err_med, vmin=-.5, vmax=.5, cmap=psgcm.Berlin_20.mpl_colormap, alpha=1, interpolation='nearest')
    # CMasher has lots of diverging cmaps. iceburn, redhisft, watermelon, wildfire
    im = ax.matshow(T_err_med, vmin=-.5, vmax=.5, cmap=cmr.guppy_r, alpha=1, interpolation='nearest')
    
    # Trim up the whitespace around the edges
    ax.set_xlim([2.5, ybins-1])
    ax.set_ylim([xbins-1,0])
    ax.set_xticks((np.arange(0,1.25,.25)*ybins).astype(np.int32), labels=[0, 0.5, 1, 1.5, 2])
    ax.set_yticks((np.arange(0,1.25,.25)*xbins).astype(np.int32), labels=[0, 10, 20, 30, 40])
      
    if show_text:
        ax.set_title('T')
        ax.set_ylabel('SNR (dB)')
        ax.set_xlabel('Median error') # I flip the title and label
        fig.colorbar(im)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    
    plt.show()   


    if plot_fileroot is not None:
        plt.savefig(f'{plot_fileroot}_Terr_median_2D.png')
        plt.savefig(f'{plot_fileroot}_Terr_median_2D.svg')
    plt.close()


    # Plot IQR
    fig, ax = plt.subplots(1,1)
    im = ax.matshow(T_err_iqr, vmin=0, vmax=2, alpha=1, interpolation='nearest')

    # Trim up the whitespace around the edges
    ax.set_xlim([2.5, ybins-1])
    ax.set_ylim([xbins-1,0])
    ax.set_xticks((np.arange(0,1.25,.25)*ybins).astype(np.int32), labels=[0, 0.5, 1, 1.5, 2])
    ax.set_yticks((np.arange(0,1.25,.25)*xbins).astype(np.int32), labels=[0, 10, 20, 30, 40])

    if show_text:
        ax.set_title('T')
        ax.set_ylabel('SNR (dB)')
        ax.set_xlabel('IQR error') # I flip the title and label
        fig.colorbar(im)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.show()   
    
    if plot_fileroot is not None:
        plt.savefig(f'{plot_fileroot}_Terr_iqr_2D.png')
        plt.savefig(f'{plot_fileroot}_Terr_iqr_2D.svg')
    plt.close()


#%%
def plot_by_pixel(ds_name, method_name):
    print(f'Plotting ds={ds_name}, method={method_name}')    
    plot_fileroot = path.join(get_plot_dir(), f'partA_{ds_name}_{method_name}')

    # Get the data from all pixels    
    biga = extract_all_pixels(ds_name, method_name)

    # For paper figures, turn off text
    show_text = False

    # Make 1D plots 
    plot_error_by_snr_binned(biga, plot_fileroot, show_text)
    plot_error_by_T_binned(biga, plot_fileroot, show_text)
    
    # Make 2D plots
    plot_error_binned_2D(biga, plot_fileroot, show_text)


#%% 
# For histogram calculations, figures out the bin # for a given value
# Returns a zero-basd bin, in range [0, Nbins-1]
# returns -1 or Nbins for out of range
def get_histogram_bin_num(val, bmin, bmax, Nbins):
    
    if val<bmin:
        return -1
    elif val>=bmax:
        return Nbins
    else:
        
        # First, interpolate to [0,1]
        t_score = (val-bmin) / (bmax-bmin) * Nbins
    
        # Get bin #
        bin_num = np.floor(t_score).astype(np.int32)
        
        return bin_num

#%%
# These really dont work. Struggles to get decent distribution 
# Boxplots work fine as long as you don't show outliers
def make_violin_plots(df_T):
    
    #tmp = df_T.head(10000)
    #sns.violinplot(data=df_T, x='method', y='ae', scale='width', cut=0, gridsize=1000)
    #sns.violinplot(data=df_T.sample(frac=0.01), x='method', y='diff', scale='width', cut=0, bw='silverman')
    sns.boxplot(data=df_T.sample(frac=0.01), x='method', y='diff', fliersize=0)
    plt.ylim([-1,1])
    plt.show()

    return


#%%
def evaluate_partA():

    datasets = get_partA_datasets()
    methods = get_partA_methods()

    
    for dataset in datasets:
        for method in methods:
            plot_by_pixel(dataset, method)

    # Hardwire    
    # #plot_by_pixel('IMAGENET_VALIDATION', 'FIT_LOGLIN')

# For testing only
def test_evaluation():
    ds_name = 'IMAGENET_TEST_1k'
    method = 'CNN_IMAGENET'

    # Short version for ISMRM/EAB    
    datasets = ['IMAGENET_TEST_1k']
    methods = ['FIT_NLLS',  'CNN_SS_INVIVO', 'CNN_SS_IMAGENET', 'CNN_IMAGENET', ]
    methods = ['FIT_NLLS',  'CNN_SS_INVIVO', 'CNN_IMAGENET', ]


    for ds_name in datasets:
        for method in methods:
            print(f'Plotting {ds_name}, method={method}')    
            plot_fileroot = path.join(get_plot_dir(), f'partA_{ds_name}_{method}')
        
            # [S0_ref, S0_pred, T_ref, T_pred, SNR]
            biga = extract_all_pixels(ds_name, method)
            
            show_text = True
            plot_error_by_snr_binned(biga, plot_fileroot, show_text)
            #plot_error_by_T_binned(biga, plot_fileroot, show_text)
            #plot_error_binned_2D(biga, None, show_text)
       
    

if __name__=='__main__':
    evaluate_partA()
    #test_evaluation()
    

