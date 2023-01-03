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

import seaborn as sns
from utility_functions import *
   
    
#%% Calculate metrics comparing a predicted 2D image with its correct label
# Returns dictionary with numerous metrics

def make_big_df(datasets, methods):

    # Optionally mask out the background.
    # Important if you use means; with medians not so much
    use_foreground_mask = False
    
    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']
    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim', 'rcrbm']
    cols = ['model', 'dataset', 'case_num', 'slice'] + metrics
    df_T = pd.DataFrame(columns=cols)
    df_S0 = pd.DataFrame(columns=cols)
   
    # Loop over all datasets and methods    
    for ds_name in datasets:
        for method_name in methods:
            
            # Get all the output files
            output_dir, ds_labels_dir, ds_images_dir = get_evaluation_paths(method_name, ds_name)
            pred_files = sorted(glob.glob(path.join(output_dir, '*.nii.gz')))
            for idx, pred_file in enumerate(pred_files):
                                
                # extract the case number, find matching file in ds_labels_dir
                file_num_str = get_file_numstr_from_file_name(pred_file)
                
                print(f'{ds_name}, {method_name}, {idx}, {file_num_str}')
                files_found = glob.glob(path.join(ds_labels_dir, f'*{file_num_str}.nii.gz'))
                if len(files_found) < 1:
                    print('ERROR: cannot find label file.')
                    
                label_file = files_found[0]
                
                # Load files
                img_true = nib.load(label_file).get_fdata()
                img_pred = nib.load(pred_file).get_fdata()

                # For each slice. In vivo files are multislice, synth are single
                for sl in range(img_true.shape[2]):
                    
                    if use_foreground_mask:
                        # Do a foreground mask, selecting only top 90% of S0 values
                        thr = image_threshold_percentage(img_true[:,:,sl,0], 10)
                        fgmask = img_true[:,:,sl,0] > thr
                    else:
                        fgmask = img_true[:,:,sl,0] > 0
                    
                    # Calculate all metrics
                    metrics_S0 = compare_images(img_pred[:,:,sl,0], img_true[:,:,sl,0], fgmask)
                    metrics_T = compare_images(img_pred[:,:,sl,1], img_true[:,:,sl,1], fgmask)
                    #review_files(label_file, pred_file)    
                                        
                    # Save them in dataframe
                    df_S0.loc[len(df_S0.index)] = [method_name, ds_name, file_num_str, f'{sl}'] + [metrics_S0[colname] for colname in metrics]
                    df_T.loc[len(df_T.index)] = [method_name, ds_name, file_num_str, f'{sl}'] + [metrics_T[colname] for colname in metrics]
                    1
                    
    return df_S0, df_T
    

#%%
# Make a print of summary statistics for each ds/method combo
def print_metrics(df):
    # Comprehensive
    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']

    # # With relative crete roffet blur metric
    #metrics = ['medae', 'medbias', 'ssim', 'rcrbm']
    #limits = [ [0, 1], [-.6, .6], [0,1], [-.5, .5]]
    
    print(f'dataset,method,bias,precision,accuracy,ssim')
    for ds_name in df.dataset.unique():
        for method_name in df.model.unique():
        
            df_specific = df[df.dataset==ds_name]
            df_specific = df_specific[df_specific.model==method_name]
            signed_error = df_specific.medbias
            absolute_error = df_specific.medae

            
            # Definitions, from paper:
            '''
            The median value of the per-slice signed error was interpreted as 
            the bias of a metric; the interquartile range (IQR, 75th-25th 
            percentile) of the per-slice signed error was interpreted as the 
            precision. The median value of the per-slice absolute error was 
            interpreted as the overall accuracy, which is dependent on both 
            bias and precision, with smaller errors indicating higher accuracy. 
            '''
            bias = np.median(signed_error)
            precision = np.percentile(signed_error, 75) - np.percentile(signed_error, 25)
            accuracy = np.median(absolute_error)
            ssim = np.median(df_specific.ssim)
    
            print(f'{ds_name},{method_name},{bias},{precision},{accuracy},{ssim}')



#%%
def plot_metrics(df, show_text=True):
    # Comprehensive
    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']
    limits = [[0, 2], [0, 1], [-1, 1], [0, 10], [0, 10], [], [0,1]]
    
    # Focused
    metrics = ['medae', 'medbias', 'ssim']
    limits = [ [0, 1], [-.6, .6], [0,1], [-.5, .5]]

    # # With relative crete roffet blur metric
    #metrics = ['medae', 'medbias', 'ssim', 'rcrbm']
    #limits = [ [0, 1], [-.6, .6], [0,1], [-.5, .5]]
    
    plot_dir = get_plot_dir()

    for ds_name in df.dataset.unique():
        
        df_ds = df[df.dataset==ds_name]
        for idx, metric_name in enumerate(metrics):
                    
            values = list()
            methods = df_ds.model.unique()                    
            for method_name in methods:
                #print(ds_name, metric_name, method_name)
                values.append(df_ds[df_ds.model==method_name][metric_name])
                
            # Make one plot for this metric
            # [10,4] was the size for all methods; [6,4] was for ISMRM abstract
            fig,ax = plt.subplots(figsize=[10,4])
            #fig,ax = plt.subplots(figsize=[6,4])
            
            
            # Add a zero line for the bias plot. Do it in backgrond
            if metric_name == 'medbias' or metric_name == 'rcrbm':
                plt.axhline(0, ls='--', lw=1, color ='red')

            # palette
            # Default palette is circular, color_palette('husl', N)
            # I will bulid my own from the first 3 of the default (matlab)
            cp_array = np.array(sns.color_palette())
            my_indices = [0,0,0,0,1,1,1,1,1,2,2,2,2,2]
            my_array = cp_array[my_indices]
            my_palette = sns.color_palette(my_array)

            # Overlaying a boxplot (stem) and stripplot, which are the points
            graph = sns.boxplot(data=df_ds, x='model', y=metric_name, fliersize=0, palette=my_palette)        
            graph = sns.stripplot(data=df_ds, x='model', y=metric_name, size=2, color='0.3')   
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Specify limits
            if len(limits[idx])>0:
                ax.set_ylim(limits[idx])
            
            # Make the PNG file with all the text parts
            if show_text:
                ax.set_xticks(np.arange(0, len(methods)), labels=methods, rotation=15, fontsize=8)
                ax.set_title(f'{ds_name} : {metric_name}')
            plt.show()
            plt.savefig(path.join(plot_dir, f'{ds_name}_{metric_name}.png'))

            # Now for the SVG file, turn off the text. Helps with paper figs
            ax.set_xticks(np.arange(0, len(methods)), labels=None)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title('')
            plt.show()                
            plt.savefig(path.join(plot_dir, f'{ds_name}_{metric_name}.svg'))

            plt.close()
    return


#%% This was suggested by KK, but needs to compare IMAGENET_VALIDATION_1k vs SCRAMBLEDIMAGENET_VALIDIATION_1k
def compare_performance_by_dataset(df_T):
    # Compare datasets, per KK
    # For metric, plot performance on URAND(y) vs IMAGNET(x)
    metric = 'medae'; lims = [0, .4]
    #metric = 'medbias'; lims = [-.2, .4]
    methods = get_partA_methods()
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    for method in methods:
        # plot one point
        xval = np.mean(df_T[df_T.dataset=='IMAGENET_VALIDATION_1k'][df_T.model==method][metric])
        yval = np.mean(df_T[df_T.dataset=='URAND_VALIDATION_1k'][df_T.model==method][metric])
        ax.plot(xval, yval, '.k')
        ax.annotate(method, textcoords='data', xy=[xval,yval])
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("IMAGENET")
    plt.ylabel("URAND")
    plt.show()

#%%
def evaluate_partA():
        
    datasets = get_partA_datasets()
    methods = get_partA_methods()
    
    df_S0, df_T = make_big_df(datasets=datasets, methods=methods)
    plot_metrics(df_T)
    print_metrics(df_T)

def test_plot():
    datasets = ['IMAGENET_VALIDATION_1k']
    methods = ['FIT_LOGLIN', 'FIT_NLLS', 'FIT_NLLS_BOUND', 'FIT_NLLS_RICE',
               'NN1D_IMAGENET', 'NN1D_URAND', 'NN1D_SS_IMAGENET', 'NN1D_SS_URAND', 'NN1D_SS_INVIVO',
               'CNN_IMAGENET', 'CNN_URAND', 'CNN_SS_IMAGENET', 'CNN_SS_URAND', 'CNN_SS_INVIVO']


    # Brief version, for ISMRM abstract/ EAB
    datasets = ['IMAGENET_TEST_1k']
    methods = ['FIT_NLLS', 
               'CNN_SS_INVIVO', 'CNN_IMAGENET', ]
 

    df_S0, df_T = make_big_df(datasets=datasets, methods=methods)
    plot_metrics(df_T)
    


if __name__=='__main__':
    evaluate_partA()
    #test_plot()



