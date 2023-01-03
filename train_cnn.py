#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 14:38:29 2022
Buidling on the spleen segmentation, this is my training loop

Modified train_synth_1 to scramble first and after

Actually I use this file to train both urand and imagenet cnns

@author: pbolan
"""

from monai.utils import first, set_determinism
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
    RandRicianNoised,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, BasicUNet, SegResNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import time
import os
import glob
import sys
import numpy as np

from utility_functions import *
from inference import *
#%% 
def prepare_loaders(data_dir, show_example=False):

    train_images = sorted(
        glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))

    # In vivo we have no labels
    if len(train_labels) == 0:
        data_dicts = [{"image": image_name} for image_name in train_images]    
        in_vivo = True
    else:
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels) ]   
        in_vivo = False    

    # # HACK
    # # Trying to add noise to the invivo data so the comparison with CNN_IMAGENET
    # # is more fair
    # in_vivo = False
    # in_vivo_withnoise = True

    # Go with 80/20 train/val split
    split_index = int(0.8 * len(train_images))
    print(f'Total {len(train_images)} datasets, using {split_index} for training')    
    train_files, val_files = data_dicts[-split_index:], data_dicts[:-split_index]
    
    # Loading and transformation
    if in_vivo:
        # Note all interpolations are bilinear, not nearest. 
        # Bilinear is appropriate for regression, nearest is for labels    
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1., 1., 1.), mode=("bilinear")),
                EnsureTyped(keys=["image"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1., 1., 1.), mode=("bilinear")),
                EnsureTyped(keys=["image"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
            ]
        )  
    elif in_vivo_withnoise:
        # Same as in_vivo, but after scaling, I'm adding Rician noise 
        print("WARNING!!!! This is my invivo noise hack")
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1., 1., 1.), mode=("bilinear")),
                EnsureTyped(keys=["image"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
                RandRicianNoised(keys=["image"], 
                                 prob=1.0, mean=0.0, std=0.1, channel_wise=True, relative=False, sample_std=True),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=(
                    1., 1., 1.), mode=("bilinear")),
                EnsureTyped(keys=["image"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
            ]
        )  
    else:
        # Note all interpolations are bilinear, not nearest. 
        # Bilinear is appropriate for regression, nearest is for labels    
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(
                    1., 1.), mode=("bilinear", "nearest")),
                EnsureTyped(keys=["image", "label"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(
                    1., 1.), mode=("bilinear", "nearest")),
                EnsureTyped(keys=["image", "label"]),
                IntensityStatsd(keys=["image"], ops=['max'], key_prefix='orig'),
                ScaleTransformSpecial(),
            ]
        ) 
    
    
    # # HACK Test this thing out
    # tmp_ds = Dataset(data=train_files, transform=train_transforms)
    # #tmp_loader = DataLoader(tmp_ds, batch_size=5, shuffle=True, num_workers=4)
    # tmp_loader = DataLoader(tmp_ds, batch_size=1)
    # tmp_data = first(tmp_loader)
    
    
    if show_example:
        # The monai loader brings the data in as [batch, channels, x, y, z]
        check_ds = Dataset(data=train_files, transform=train_transforms)
        #check_ds = CacheDataset(data=train_files, transform=train_transforms,
        #    cache_rate=1.0, num_workers=4)
        
        check_loader = DataLoader(check_ds, batch_size=1, shuffle=True, num_workers=4)
        check_data = first(check_loader)
        images, labels = (check_data["image"][0,:,:,:,0], check_data["label"][0,:,:,:,0])
        print(f"images shape: {images.shape}, labels shape: {labels.shape}")
            
        print('Image series:')
        imshow_montage(images.permute(1,2,0))
        print('Labels (S0, T):')
        imshow_montage(labels.permute(1,2,0))        
    
    num_workers = 4
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=num_workers)
        
    batch_size = 100    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


#%% Custom loss function    
# This loss function can handle supervised (R, S0), and self-supervised 
def special_loss(outputs, labels, inputs, eta, loss_type):

    # Use MSE for all possible losses
    loss_fn = torch.nn.MSELoss()  
    
    S0_pred = outputs[:,0,:,:]
    T_pred = outputs[:,1,:,:]

    if not labels == None:
        S0_true = labels[:,0,:,:,0] # Zero because there is only 1 slice
        T_true = labels[:,1,:,:,0]

        # First, your standard loss on estimated paramters
        loss_T = loss_fn(T_pred, T_true)
        loss_S0 = loss_fn(S0_pred, S0_true)
    
 
    if loss_type == 'labels':
        loss_total = 0.5 * loss_T + 0.5 * loss_S0      
        
    elif loss_type=='label_T':        
        loss_total =  loss_T 
        
    else:    
         
        # calculate self-supervised loss for ALL cases
        
        # Now a data consistency loss. Note ADC is denormalized, S0 is normalized
        eta = torch.from_numpy(eta).float()
         
        # Using absolute values to only consider physically reasonable values
        S0_pred = torch.abs(S0_pred.clone().unsqueeze(1).repeat([1, eta.shape[0], 1, 1]))
        T_pred = torch.abs(T_pred.clone().unsqueeze(1).repeat([1, eta.shape[0], 1 , 1]))
        eta = eta.unsqueeze(0).repeat([S0_pred.shape[0], 1]) # Expands the batch dim
        eta = eta.unsqueeze(2).repeat([1, 1, S0_pred.shape[2]])
        eta = eta.unsqueeze(3).repeat([1, 1, 1, S0_pred.shape[3]]).to(S0_pred.device)
        
        
        #eta = eta.repeat([S0_pred.shape[0], S0_pred.shape[1], S0_pred.shape[2], 1]).to(S0_pred.device)
        S_sim = (S0_pred * torch.exp(-eta / (T_pred + 1e-16)))
         
        # Gaussian model. MSE between data and curve
        loss_data = loss_fn(S_sim, inputs[:,:,:,:,0])
        
        if loss_type=='data_consistency':
            loss_total = loss_data
            
        elif loss_type=='labels_and_data_consistency':
            loss_total = 0.5 * loss_data + 0.25 * loss_T + 0.25 * loss_S0
         
            
         
    # HACK debug
    # # Plot some images
    # xS0_true = S0_true.clone().detach().cpu() 
    # xS0_pred = S0_pred.clone().detach().cpu() 
    # xT_true = T_true.clone().detach().cpu() 
    # xT_pred = T_pred.clone().detach().cpu() 
    # xS_sim = S_sim.clone().detach().cpu()
    # xinputs = inputs[:,:,:,:,0].clone().detach().cpu()
     
    # imshow_montage(xS_sim[2,:,:,:].permute([1, 2, 0]))
    # imshow_montage(xinputs[2,:,:,:].permute([1, 2, 0]))
    # imshow_montage((xinputs[2,:,:,:]-xS_sim[2,:,:,:]).permute([1, 2, 0]))
    
    #print(f'                        T2 loss: {loss_T:.4f}, data loss: {loss_data:.4f}')        

        
    # These are troubleshooting only
    if torch.isnan(loss_total).any():
        print('*** HAVE A NAN')
    if torch.isinf(loss_total).any():
        print('*** HAVE A INf value')   
        
    return loss_total



#%% Training loop
# Supports a shortened length of time series during training (part3)
def training_loop(n_epochs, optimizer, loss_type, model, train_loader, val_loader, device, length=10):
    
    # Calculate once; needed for some loss functions
    eta = get_eta()
         
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    
    # Early stopping values
    min_epochs = 6
    patience = 5000 # number of epochs to wait for improvement
    trigger_times = 0
    best_loss = 1e6
    best_state_dict = None
    best_epoch = 0
    stopped_early = False
         
    train_loss_values = []
    val_loss_values = []
           
    for epoch in range(n_epochs):
            
        model.train()
        train_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            
            inputs =  batch_data["image"].to(device)
            if 'label' in batch_data:
                labels = batch_data["label"].to(device)
            else:
                labels  = None
                
            optimizer.zero_grad()
            
            N_batch = inputs.shape[0]            
            inputs = inputs[:,:length,:,:,:] # Shorten the time domain
            outputs = model(inputs[:,:,:,:,0])   
            
            loss = special_loss(outputs, labels, inputs, eta, loss_type)
            #loss = loss_function(outputs, labels[:,:,:,:,0])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(
            #    f"{step}/{len(train_loader) // train_loader.batch_size}, "
            #   f"train_loss: {loss.item():.4f}")
        train_loss /= step
        train_loss_values.append(train_loss)
        print(f"epoch {epoch + 1}/{n_epochs} train loss: {train_loss:.4f}")
    
        model.eval()
        with torch.no_grad():
            
            val_loss = 0
            num_val_batches = len(val_loader)

            for val_data in val_loader:
                val_inputs =  val_data["image"].to(device)
                if 'label' in val_data:
                    val_labels = val_data["label"].to(device)
                else:
                    val_labels  = None
                
                N_batch = val_inputs.shape[0]
                val_inputs = val_inputs[:,:length,:,:,:] # Shorten the time domain
                val_outputs = model(val_inputs[:,:,:,:,0])
                                               
                loss = special_loss(val_outputs, val_labels, val_inputs, eta, loss_type)
                #loss = loss_function(val_outputs, val_labels[:,:,:,:,0])

                val_loss += loss.item()                

            val_loss /= num_val_batches
            val_loss_values.append(val_loss)
            print(f"     average val_loss: {val_loss:.4f} ", end = '')
            
        # Check for early stopping
        # Print a character to identify status
        early_stopping_code = ''
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
            
            if (trigger_times > patience) and epoch > min_epochs:
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
    
    print('Done.')
    
    # Like Keras, I'll make a history dictionary
    history = dict()
    history['train_loss'] = train_loss_values
    history['validation_loss'] = val_loss_values
    return {'history': history, 'state_dict': best_state_dict, 'epoch': best_epoch, 'loss': best_loss}

#%%
def train_cnn(ds_name, loss_type, model_filename=None, length=10, ):

    # Filenames    
    dataset_dir = path.join('/home/pbolan/prj/prostate_t2map/datasets/', ds_name)
    model_dir = '/home/pbolan/prj/prostate_t2map/models'
    model_fullfile = path.join(model_dir, model_filename)

    if model_filename is None:
        model_filename = 'model_temp_scripted.pt'
    elif path.exists(model_fullfile):
        print(f'Model [{model_filename}] exists - skipping.')
        return None, None  
    
    train_loader, val_loader = prepare_loaders(dataset_dir)

    # standard PyTorch program style: create UNet, Loss and Adam optimizer
    device = torch.device("cuda:0")
    if True:
        # My default model
        model = UNet(
            spatial_dims=2,
            in_channels=length,
            out_channels=2,
            #channels=(16, 32, 64, 128, 256),
            #channels=(32, 32, 64, 128, 256),
            #channels=(64, 64, 128, 256, 512), # super-flex model
            channels=(128, 128, 256, 512), # wide model
            strides=(1, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH ).to(device)
    else:
        model = BasicUNet(
            spatial_dims=2,
            in_channels=length,
            out_channels=2, 
            #features=(32, 32, 64, 128, 256, 32),
            ).to(device)    
    
    # print out a summary
    show_model_details(model)
    
    #loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    #optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), 2e-3)
    
    n_epochs = 1000
    #n_epochs = 5
    start = time.time()
    train_results = training_loop(n_epochs, optimizer, loss_type, model, train_loader, val_loader, device, length)
    duration = time.time() - start
    print(f'Training completed in {duration:1f} seconds.')

    # Plot here
    plot_training_history(train_results['history'])
    
    # To save as script, first apply the state dict, then save
    model.load_state_dict(train_results['state_dict']) # Load best, may not be current
    model_scripted = torch.jit.script(model)

    print(f'Writing scripted model out as {model_fullfile}')
    model_scripted.save(model_fullfile)

    return model, train_results




#%%
def train_all():
    
    # There are 5 CNNs in the study. Supervised and self-supervised trained on 
    # the two synthetic datsets
    model, train_results = train_cnn('synth_imagenet_10k_train', 'labels', 'cnn_imagenet.pt')
    model, train_results = train_cnn('synth_imagenet_10k_train', 'data_consistency', 'cnn_ss_imagenet.pt')
    model, train_results = train_cnn('synth_urand_10k_train', 'labels', 'cnn_urand.pt')
    model, train_results = train_cnn('synth_urand_10k_train', 'data_consistency', 'cnn_ss_urand.pt')

    # And one trained on the in vivo dataset, self-supervised
    model, train_results = train_cnn('invivo2D_set2', 'data_consistency', 'cnn_ss_invivo.pt')



    return;
     

if __name__=='__main__':
    set_determinism(seed=42.58)
    train_all()

    




