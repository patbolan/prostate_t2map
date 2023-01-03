#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 18:37:21 2022


@author: pbolan
"""
import numpy as np
from os import path, makedirs
import sys
import glob
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from skimage import filters
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi

import seaborn as sns
from utility_functions import *


    
#%%
# For part 2 (and 3) we compare each dataset to the results from a reference dataset
def make_big_df(ds_reference, datasets, methods):

    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']
    cols = ['method', 'dataset', 'case_num', 'slice'] + metrics
    df_T = pd.DataFrame(columns=cols)
    df_S0 = pd.DataFrame(columns=cols)
   
    # Loop over all datasets and methods    
    for method_name in methods:
        
        # Load up the reference data for this method
        output_ref_dir, _, _ = get_evaluation_paths(method_name, ds_reference)
            
        
        # Then loop overa all the other datasets    
        for ds_name in datasets:
            
            # Get all the output files
            output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths(method_name, ds_name)
            pred_files = sorted(glob.glob(path.join(output_dir, '*.nii.gz')))
            for jdx, pred_file in enumerate(pred_files):
                                
                # extract the case number, find matching file in ds_labels_dir
                file_num_str = get_file_numstr_from_file_name(pred_file)
                
                print(f'{ds_name}, {method_name}, {jdx}, {file_num_str}')
              
                # Load files
                img_pred = nib.load(pred_file).get_fdata()
                
                # Now load up the reference for this subject. 
                ref_fname = path.join(output_ref_dir, path.basename(pred_file))
                img_ref = nib.load(ref_fname).get_fdata()
                fgmask = (img_ref[:,:,:,0]*0) == 0

                # For each slice. In vivo files are multislice, synth are single
                for sl in range(img_pred.shape[2]):
                                         
                    # Calculate all metrics
                    metrics_S0 = compare_images(img_pred[:,:,sl,0], img_ref[:,:,sl,0], fgmask[:,:,sl])
                    metrics_T = compare_images(img_pred[:,:,sl,1], img_ref[:,:,sl,1], fgmask[:,:,sl])
                                        
                    # Save them in dataframe
                    df_S0.loc[len(df_S0.index)] = [method_name, ds_name, file_num_str, f'{sl}'] + [metrics_S0[colname] for colname in metrics]
                    df_T.loc[len(df_T.index)] = [method_name, ds_name, file_num_str, f'{sl}'] + [metrics_T[colname] for colname in metrics]
                    
    return df_S0, df_T
    
#%%
def plot_metrics(df, metrics, limits):

    for ds_name in df.dataset.unique():
        
        df_ds = df[df.dataset==ds_name]
        for idx, metric_name in enumerate(metrics):
                    
            values = list()
            methods = df_ds.method.unique()                    
            for method_name in methods:
                print(ds_name, metric_name, method_name)
                values.append(df_ds[df_ds.method==method_name][metric_name])
                
            # Make one plot for this metric
            fig,ax = plt.subplots(figsize=[8,4])
            if False:
                ax.violinplot(values, showmedians=True)
                #ax.boxplot(values) # Boxplots don't look good with this data 5, fontsize=8)
            else:
                #sns.violinplot(data=df, x='method', y=metric_name, scale='width', cut=0, gridsize=100, bw=0.1)        
                sns.boxplot(data=df_ds, x='method', y=metric_name, fliersize=0)        
                sns.stripplot(data=df_ds, x='method', y=metric_name, size=2, color='0.3')        
                ax.set_xticks(np.arange(0, len(methods)), labels=methods, rotation=15, fontsize=8)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
            ax.set_title(f'{ds_name} : {metric_name}')
            # Todo: consider adjusting y-lim to 2x that of method #1, loglin
            #refrange = df_ds[df_ds.method=='FIT_LOGLIN'][metric_name]
            #ax.set_ylim([np.min(refrange)/2, np.max(refrange)*2])
            if len(limits[idx])>0:
                ax.set_ylim(limits[idx])
            plt.show()

    return

#%%
def plot_trends(df_T, metrics, limits, start_val):
    datasets = df_T.dataset.unique()
    methods = df_T.method.unique()

    xvals = [i for i in range(len(datasets)+1)]

    # Prep figure
    fig, axs = plt.subplots(len(metrics), len(methods), figsize=[18,12])
    show_text = False
    
    for idx, metric in enumerate(metrics):
        for jdx, method in enumerate(methods):
            median_line = np.zeros(len(datasets)+1)
            err_lb = np.zeros(len(datasets)+1)
            err_ub = np.zeros(len(datasets)+1)
            
            median_line[0] = start_val[idx]
            err_lb[0] = start_val[idx]
            err_ub[0] = start_val[idx]
            
            for kdx, ds_name in enumerate(datasets):
                # Select a subset matching both ds and method
                df_sub = df_T.loc[df_T['method']==method].loc[df_T['dataset']==ds_name]
                median_line[kdx+1] = np.mean(df_sub[metric])
                err_lb[kdx+1] = np.percentile(df_sub[metric], 25)
                err_ub[kdx+1] = np.percentile(df_sub[metric], 75)
                
            axs[idx,jdx].plot(xvals, median_line, '.-b')
            axs[idx,jdx].fill_between(xvals, err_lb, err_ub, alpha=0.4)
            axs[idx,jdx].plot(xvals, median_line, '.-b')
            axs[idx,jdx].set_ylim([0,1])
            
            if show_text:
                axs[idx,jdx].legend(['median', 'IQR'])
                axs[idx,jdx].set_title(f'{method} : {metric}')
            else:
                axs[idx,jdx].set_xticklabels([])
                axs[idx,jdx].set_yticklabels([])
            
            if len(limits[idx])>0:
                axs[idx,jdx].set_ylim(limits[idx])
            plt.show()
                

#%%
def evaluate_partC():
    ds_reference = 'INVIVO2D_SET3'        
    datasets = [f'INVIVO2D_SET3_NOISE_{idx+2}' for idx in range(8)]
    methods = ['FIT_NLLS', 'CNN_IMAGENET', 'CNN_SS_INVIVO', ]
    methods = ['FIT_NLLS', 'NN1D_URAND', 'CNN_IMAGENET', 'CNN_SS_INVIVO', ]

    # methods = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE',
    #            'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND',
    #            'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND']
    
    df_S0, df_T = make_big_df(ds_reference, datasets, methods)
    
    
    metrics = ['medae', 'medbias', 'ssim']
    limits = [ [0, 1], [-1, 1], [0,1]]
    start_val = [0, 0, 1]

    plot_metrics(df_T, metrics, limits)
    plot_trends(df_T, metrics, limits, start_val)



if __name__=='__main__':
    #evaluate_partC()
    evaluate_partC()



