#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:19:44 2022

@author: pbolan
"""
import glob
import os
import matplotlib
import numpy as np
import sys
from utility_functions import *

import evaluate_partA_byslice
import evaluate_partA_bypixel




#%%
def make_html(plot_dir):
    # Make the allplots html
    html_file = os.path.join(plot_dir, 'all_plots.html')

    ds_names = get_partA_datasets()
    methods = get_partA_methods()    

    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']
    metrics = ['medae', 'medbias', 'ssim']
    pix_plots = ['Terr_x_snr', 'Terr_x_snr_binned', 'Terr_x_Ttrue', 'Terr_x_Ttrue_binned', 'Tpred_x_Ttrue', 'Tpred_x_Ttrue_binned']
    pix_plots = ['Terr_x_snr_binned', 'Terr_x_Ttrue_binned', 'Tpred_x_Ttrue_binned', 'Terr_median_2D', 'Terr_iqr_2D']
    
    with open(html_file, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n\n')
        
        f.write('<h1>By Slice</h1>\n')
        for ds_name in ds_names:        
            f.write(f'<h2>{ds_name}</h2>\n')        
            for metric in metrics:        
                f.write(f'<img src="{ds_name}_{metric}.png">\n')
        f.write('\n')
    
        f.write('<h1>By Pixel</h1>\n')
        for ds_name in ds_names:        
            f.write(f'<h2>{ds_name}</h2>\n')
            for method in methods: 
                f.write(f'<h3>{method}</h3>\n')
                for pix_plot in pix_plots:
                    f.write(f'<img src="partA_{ds_name}_{method}_{pix_plot}.png">\n')
            f.write('\n')                    
        

        f.write('</body>\n</html>\n')

    return html_file


#%% This is for the supplemental figure S1
def make_html_transpose(plot_dir):
    # Make the allplots html
    html_file = os.path.join(plot_dir, 'all_plots_transpose.html')

    ds_names = get_partA_datasets()
    methods = get_partA_methods()    

    metrics = ['mae', 'medae', 'medbias', 'mse', 'nrmse', 'nmi', 'ssim']
    metrics = ['medae', 'medbias', 'ssim']
    pix_plots = ['Terr_x_snr', 'Terr_x_snr_binned', 'Terr_x_Ttrue', 'Terr_x_Ttrue_binned', 'Tpred_x_Ttrue', 'Tpred_x_Ttrue_binned']
    pix_plots = ['Terr_x_snr_binned', 'Terr_x_Ttrue_binned', 'Tpred_x_Ttrue_binned', 'Terr_median_2D', 'Terr_iqr_2D']
    
    with open(html_file, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n\n')
        
        f.write('<h1>By Slice</h1>\n')
        for ds_name in ds_names:        
            f.write(f'<h2>{ds_name}</h2>\n')        
            for metric in metrics:        
                f.write(f'<img src="{ds_name}_{metric}.png">\n')
        f.write('\n')
    
        f.write('<h1>By Pixel</h1>\n')
        for ds_name in ds_names:        
            f.write(f'<h2>{ds_name}</h2>\n')
            for pix_plot in pix_plots:
                f.write(f'<h3>{pix_plot}</h3>\n')
                for method in methods: 
                    f.write(f'<img src="partA_{ds_name}_{method}_{pix_plot}.png">\n')

            f.write('\n')                    
        

        f.write('</body>\n</html>\n')

    return html_file


#%%
def make_histgram_plots(plot_dir):
    html_file = os.path.join(plot_dir, 'snr_histograms.html')

    ds_names = get_partA_datasets()
    methods = get_partA_methods()    
    
    with open(html_file, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<body>\n\n')
        f.write('<h1>Histograms by SNR</h1>\n')
        for ds_name in ds_names:        
            f.write(f'<h2>{ds_name}</h2>\n')        
            for method in methods:        
                f.write(f'<h3>{method}</h3>\n')        
                f.write(f'<img src="partA_{ds_name}_{method}_snr_histogram.png">\n')
        f.write('\n')
        f.write('</body>\n</html>\n')

    return html_file 
        
    
#%%
def make_all_plots():
    plots_dir = get_plot_dir()
    
    # Remove existing plots and html
    print('*** Deleting old files.')
    fileslist = glob.glob(os.path.join(plots_dir, '*'))
    for file in fileslist: 
        os.remove(file)
    
    # Regnerate all plots
    #matplotlib.use('QtAgg')
    print('*** Regenerating all plots.')
    evaluate_partA_byslice.evaluate_partA()
    evaluate_partA_bypixel.evaluate_partA()
    
    # With my current environment and spyder version this crashes. 
    # Restart Spyder to change backends
    #matplotlib.use('module://matplotlib_inline.backend_inline')
    
    # Regenerate the HTML
    print('*** Making html.')
    html_fname = make_html(plots_dir)
    
    # Also, histogram plots
    
    print(f'*** Done. Results viewable here: ')
    print(f'{html_fname}')


if __name__=='__main__':
    make_all_plots()





