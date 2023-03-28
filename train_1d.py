#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 17:10:40 2020

For this version I am going to NOT use MONAI
I will load image datasets, flatten them, and then use my older 1D code

@author: bolan
"""
from matplotlib import pyplot as plt
import numpy as np

from monai.utils import first, set_determinism

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import sys
import glob
from os import path

import pandas as pd
from scipy.stats import pearsonr
from scipy import optimize
from scipy import special
import nibabel as nib

import pickle

# I pip installed this:
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from exp_fitting import *
from utility_functions import *

#%%
def fit_exp_nlls_2p_special(eta, ydata):
    y_mean = ydata.mean()
    
    # Two versions - bound and unbound
    params, params_covariance = optimize.curve_fit(exp_decay_2p, eta, ydata/y_mean, p0=[1, 0.5], maxfev=1e5, method='trf')
    #params, params_covariance = optimize.curve_fit(exp_decay_2p, eta, ydata/y_mean, p0=[1, 0.5], maxfev=10000, 
    #                                               method='trf', bounds=([-np.inf, 0], [np.inf, 3]))
    
    y_sim = exp_decay_2p(eta, params[0], params[1]) * y_mean    
    rsq = r_squared(ydata, y_sim)
    N = len(eta)
    rsq_adj = 1 - (((1-rsq)*(N-1)) / (N-2-1))
    return {'S0':params[0]*y_mean, 
            'R':params[1],
            'rsq': rsq,
            'rsq_adj': rsq_adj,
            'y_sim': y_sim}



#%% Load data from images 
class OneDExpDatasetFromImages(Dataset):
    
    def __init__(self, dataset_dir, num_images=2, image_offset=0):
        
        # This will load up the specified number of images, starting at offset
        # Users of this class will decide how many are train, test
        
        # get lists of files in both directories
        image_dir = path.join(dataset_dir, 'images')
        label_dir = path.join(dataset_dir, 'labels')
    
    
        # HACK Hardwired. Instead, load up file #1 and read size. Or pass in
        Nsize = 128
        Np = 10
        image_cache = np.zeros([num_images, Nsize, Nsize, Np])
        label_cache = np.zeros([num_images, Nsize, Nsize, 2])
        
        for idx in range(num_images):
            img_idx = idx+image_offset
            image_file = path.join(image_dir, f'synth_{img_idx:06d}.nii.gz')            
            label_file = path.join(label_dir, f'synth_{img_idx:06d}.nii.gz')    

            imgseries = nib.load(image_file).get_fdata()
            
            # Perform normalization here , per image (happens to be one slice)
            normalization_scaling = 1.0 / imgseries.max()
            image_cache[idx,:,:,:] = imgseries[:,:,0,:] * normalization_scaling

            label = nib.load(label_file).get_fdata()
            
            # Also need to scale S0
            # BUG - had been scaling T by S0max. Corrected here.
            #label[:,:,:,1] = label[:,:,:,1] * normalization_scaling
            label[:,:,:,0] = label[:,:,:,0] * normalization_scaling
            label_cache[idx,:,:,:] = label[:,:,0,:]        
            
            
        self.time_series = image_cache.reshape([num_images*Nsize*Nsize, Np]).copy()
        self.S0 = label_cache[:,:,:,0].reshape([num_images*Nsize*Nsize]).copy()        
        self.T = label_cache[:,:,:,1].reshape([num_images*Nsize*Nsize]).copy()

        print(f"Loaded {num_images} images of size {Nsize}x{Nsize}, giving {self.T.shape[0]} samples")   

    
    def __len__(self):        
        return self.time_series.shape[0]
    
        
    # Returns a dictionary holding all tensors, normalized, channel first
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract
        time_series = self.time_series[idx,:]
       
        if time_series.max() <= 0:
            print('Whoah! the whole thing is zeros!!!!')
       
        # Disable pixel-by-pixel normalization. THis is done per-slice, above
        normalization_scaling = 1.0
        
        time_series = time_series * normalization_scaling
        time_series = torch.from_numpy(time_series.astype(np.float32))
  
        # Labels. This "none" notation keeps them as arrays not scalars
        S0 = torch.from_numpy(self.S0[idx, None].astype(np.float32)) * normalization_scaling
        T = torch.from_numpy(self.T[idx, None].astype(np.float32))

        # Note .from_numpy needs array, not scalars
        normalization_scaling = torch.from_numpy(np.array(normalization_scaling))
                
        
        # Add channel dims
        # Don't need to unsqueeze teh time series - it's an array, R and S0 are scalars
        # time_series.unsqueeze_(0)
        # S0.unsqueeze_(0)
        # R.unsqueeze_(0)
        # normalization_scaling.unsqueeze_(0)
        
        # You can package up the data any way you want, but match it in the 
        # training loop. I like this explicit one because I can add elements
        # without many changes
        #return {'time_series': time_series, 'T': T, 'S0': S0, 'normalization_scaling':normalization_scaling}
        return {'time_series': time_series, 'T': T, 'S0': S0}


#%% This version is for invivo data, which has different naming and no labels
class OneDExpDatasetFromImages_Invivo(Dataset):
    
    def __init__(self, dataset_dir, num_images=2, image_offset=0):
        
        # This will load up the specified number of images, starting at offset
        # Users of this class will decide how many are train, test
        
        # get lists of files in both directories
        image_dir = path.join(dataset_dir, 'images')
    
        # HACK Hardwired. Instead, load up file #1 and read size. Or pass in
        Nsize = 128
        Np = 10
        image_cache = np.zeros([num_images, Nsize, Nsize, Np])
        label_cache = np.zeros([num_images, Nsize, Nsize, 2])
        
        for idx in range(num_images):
            img_idx = idx+image_offset
            image_file = path.join(image_dir, f'invivo_{img_idx:06d}.nii.gz')            

            imgseries = nib.load(image_file).get_fdata()
            
            # Perform normalization here , per image (happens to be one slice)
            normalization_scaling = 1.0 / imgseries.max()
            image_cache[idx,:,:,:] = imgseries[:,:,0,:] * normalization_scaling

                        
        self.time_series = image_cache.reshape([num_images*Nsize*Nsize, Np]).copy()

        print(f"Loaded {num_images} images of size {Nsize}x{Nsize}")   

    
    def __len__(self):        
        return self.time_series.shape[0]
    
        
    # Returns a dictionary holding all tensors, normalized, channel first
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract
        time_series = self.time_series[idx,:]
       
        if time_series.max() <= 0:
            print('Whoah! the whole thing is zeros!!!!')
       
        # Disable pixel-by-pixel normalization. THis is done per-slice, above
        normalization_scaling = 1.0
        
        time_series = time_series * normalization_scaling
        time_series = torch.from_numpy(time_series.astype(np.float32))
  
        # Note .from_numpy needs array, not scalars
        normalization_scaling = torch.from_numpy(np.array(normalization_scaling))
                
        return {'time_series': time_series}

    
#%% Select device for training    
def get_device():
    if torch.cuda.is_available():        
        device = torch.device('cuda') 
    else: 
        device = torch.device('cpu')
    print(f"Training on device {device}.")  
    return device

    
#%% Custom loss function    
# This loss function can handle supervised (R, S0), and self-supervised 
def special_loss(outputs, labels, time_series_norm, eta, loss_type):
    
    mse_loss_fn = nn.MSELoss()
    #L1_loss_fn = nn.L1Loss()
    
    # All normalized
    S0_pred = outputs[:,0]
    T_pred = outputs[:,1]

    if not labels == None:
        S0_true = labels[:,0]
        T_true = labels[:,1]

        # First, your standard loss on estimated paramters
        loss_T = mse_loss_fn(T_pred, T_true)
        loss_S0 = mse_loss_fn(S0_pred, S0_true)
        
    
    if loss_type=='labels':
        loss_total = 0.5 * loss_T + 0.5 * loss_S0
        
    elif loss_type=='label_T':
        loss_total = loss_T 
        
    else:
        # calculate self-supervised loss
        
        # Now a data consistency loss. Note ADC is denormalized, S0 is normalized
        eta = torch.from_numpy(eta).float()
    
        # Using absolute values to only consider physically reasonable values
        S0_pred = torch.abs(S0_pred.unsqueeze(1).repeat([1, eta.shape[0]]))
        T_pred = torch.abs(T_pred.unsqueeze(1).repeat([1, eta.shape[0]]))
        eta = eta.unsqueeze(0).repeat([S0_pred.shape[0], 1]).to(S0_pred.device)
        S_sim = S0_pred * torch.exp(-eta / (T_pred + 1e-16))
    
        # Gaussian model. MSE between data and curve
        loss_data = mse_loss_fn(S_sim, time_series_norm)
    
        if loss_type=='data_consistency':
            loss_total = loss_data
        elif loss_type=='labels_and_data_consistency':
            loss_total = 0.5 * loss_data + 0.25 * loss_T + 0.25 * loss_S0
        
    if torch.isnan(loss_total).any():
        print('*** HAVE A NAN')
    if torch.isinf(loss_total).any():
        print('*** HAVE A INf value')    
    return loss_total


#%% Advanced training loop. 
# This I borrowed from MONAI's 3d classification example
def training_loop(n_epochs, optimizer, loss_type, model, train_loader, val_loader, device):
    
    # Calculate once; needed for some loss functions
    eta = get_eta()
    
    # Early stopping values
    min_epochs = 5
    patience = 0 # number of epochs to wait for improvement
    trigger_times = 0
    best_loss = 1e6
    best_state_dict = None
    best_epoch = 0
    stopped_early = False
    
    train_loss_values = list()
    validation_loss_values = list()       
    print(f'Training for {n_epochs} epochs. Losses (average per batch):')
    
    n_train_batches = len(train_loader) # batches per epoch
    n_validation_batches = len(val_loader)
        
    for epoch in range(1, n_epochs + 1):
        
        model.train()
        start_time = time.time()
        epoch_loss = 0     
        
        for idx_batch, sample_batched in enumerate(train_loader):
            time_series = sample_batched['time_series'].to(device)
            
            if 'T' in sample_batched:
                T = sample_batched['T'].to(device)       
                S0 = sample_batched['S0'].to(device)       
                labels = torch.cat((S0, T), dim=1)
            else:
                labels = None
                
            optimizer.zero_grad()
            #outputs = model(time_series)
            outputs = model(time_series)
            
            loss = special_loss(outputs, labels, time_series, eta, loss_type)
            
            loss.backward()
            optimizer.step()        
    
            batch_loss = loss.item()
            epoch_loss += batch_loss
                                   
        # Gotta return this. DO I also want to plot per-batch losses?   
        #print(f'Epoch {epoch} training mean loss {epoch_loss / n_train_batches}')
        train_loss_values.append(epoch_loss / n_train_batches)
        
        # Now validate
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            
            for idx_batch, sample_batched in enumerate(val_loader):

                time_series = sample_batched['time_series'].to(device)
 
                if 'T' in sample_batched:
                    T = sample_batched['T'].to(device)       
                    S0 = sample_batched['S0'].to(device)       
                    labels = torch.cat((S0, T), dim=1)
                else:
                    labels = None                
 
                outputs = model(time_series)
                
                loss = special_loss(outputs, labels, time_series, eta, loss_type)
            
                batch_loss = loss.item()
                epoch_loss += batch_loss
                #print(f'Epoch {epoch} validation batch {idx_batch} loss {batch_loss}')    
                
            #print(f'Epoch {epoch} validation mean loss {epoch_loss / n_validation_batches}')
            val_loss = epoch_loss / n_validation_batches
            validation_loss_values.append(val_loss)            
            
        duration_s = time.time() - start_time
        # No newline here
        print(f'  {epoch}: time {duration_s:.1f} s, train {train_loss_values[-1]:.4f}, validation  {validation_loss_values[-1]:.4f} ', end = '')
        
        # Check for early stopping
        # Print a character to identify status
        early_stopping_code = ''
        if epoch >= (min_epochs-1):
            if val_loss < best_loss:
                # have a new winner!
                best_loss = val_loss
                best_epoch = epoch
                trigger_times = 0
                best_state_dict = model.state_dict()
                early_stopping_code = '<-- best'
            else:
                # Getting worse
                trigger_times += 1
                
                if trigger_times > patience:
                    print(f'   --> Early stopping. Best result {best_loss} at epoch {best_epoch}')
                    stopped_early = True
                    break
                else:
                    early_stopping_code = f'worse {trigger_times}x'
        print(early_stopping_code)
            
    if not stopped_early:
        print('Got to the end without meeting early stopping. Returning last model')
        best_state_dict = model.state_dict()
        best_epoch = epoch
        
    # Like Keras, I'll make a history dictionary
    history = dict()
    history['train_loss'] = train_loss_values
    history['validation_loss'] = validation_loss_values
    return {'history': history, 'state_dict': best_state_dict, 'epoch': best_epoch, 'loss': best_loss}


#%% Model
# FullyConnectedNN with configurable layers and size
class FullyConnectedNN(nn.Module):
    def __init__(self, num_filter=64, num_layers=5, input_length=10, output_length=1):
        super(FullyConnectedNN, self).__init__()
        
        self.num_filter = num_filter # number of filters for first layer

        # First layer        
        self.linearFirst = nn.Linear(input_length, self.num_filter)

        # Variable hidden layers
        self.layers = nn.ModuleList()
        for n_layer in range(1,num_layers-1):
            self.layers.append(nn.Linear(self.num_filter, self.num_filter))

        # Last layer,         
        self.linearLast = nn.Linear(self.num_filter, output_length)

    def forward(self, x):
        x = F.relu(self.linearFirst(x))
        
        # Connect all the hidden layers in a loop
        for hlayer in self.layers:
            x = F.relu(hlayer(x))        
        
        x = self.linearLast(x)
        return x


#%%
def train_1dnn(ds_name, loss_type, model_filename=None, n_train_images=80, n_val_images=20, length=10):

    # Filenames    
    dataset_dir = path.join(get_datasets_dir(), ds_name)
    model_dir = get_models_dir()
    model_fullfile = path.join(model_dir, model_filename)
    
    if model_filename is None:
        model_filename = 'model_temp_scripted.pt'
    elif path.exists(model_fullfile):
        print(f'Model [{model_filename}] exists - skipping.')
        return None, None    
    
    if ds_name.startswith('invivo'):
        ds_train = OneDExpDatasetFromImages_Invivo(dataset_dir=dataset_dir, num_images=n_train_images, image_offset=0)
        ds_valid = OneDExpDatasetFromImages_Invivo(dataset_dir=dataset_dir, num_images=n_val_images, image_offset=n_train_images)
    else:            
        ds_train = OneDExpDatasetFromImages(dataset_dir=dataset_dir, num_images=n_train_images, image_offset=0)
        ds_valid = OneDExpDatasetFromImages(dataset_dir=dataset_dir, num_images=n_val_images, image_offset=n_train_images)
    
    batch_size = 10000
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False)  
    
    device = get_device()    
    
    #model = FullyConnectedNN(num_filter=64, num_layers=6, input_length=len(get_eta()), output_length=2)
    model = FullyConnectedNN(num_filter=64, num_layers=6, input_length=length, output_length=2)
    model.to(device)
    show_model_details(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-3)    
    loss_fn = nn.MSELoss()
    
    model.train()
    
    # 10 seems like just a few epochs, but since each pixel is ~independent, it
    # is really a lot of training. Each epoch takes ~5 min
    n_epochs = 10
    n_epochs = 5
    train_results = training_loop(model=model, optimizer=optimizer, loss_type=loss_type,
                                  train_loader=train_loader, val_loader=val_loader, 
                                  device=device, n_epochs=n_epochs)
    
    # Plot here
    plot_training_history(train_results['history'])
    
    # To save as script, first apply the state dict, then save
    model.load_state_dict(train_results['state_dict']) # Load best, may not be current
    model_scripted = torch.jit.script(model)

    model_fullfile = path.join(model_dir, model_filename)
    print(f'Writing scripted model out as {model_fullfile}')
    model_scripted.save(model_fullfile)
    
    return model, train_results

#%%
def train_all():
    set_determinism(seed=0)
    model, results = train_1dnn('synth_urand_1k_train', 'labels', 'nn1d_urand.pt', 800, 200)
    model, results = train_1dnn('synth_urand_1k_train', 'data_consistency', 'nn1d_ss_urand.pt', 800, 200)
    model, results = train_1dnn('synth_imagenet_1k_train', 'labels', 'nn1d_imagenet.pt', 800, 200)
    model, results = train_1dnn('synth_imagenet_1k_train', 'data_consistency', 'nn1d_ss_imagenet.pt', 800, 200)

    model, results = train_1dnn('invivo2D_set2', 'data_consistency', 'nn1d_ss_invivo.pt', 800, 200)

    return;

if __name__=='__main__':
    set_determinism(seed=0)
    train_all()
    #model, results = train_1dnn('synth_urad_1k_train', 'labels', 'nn1d_urad_xxx.pt', 800, 200)



