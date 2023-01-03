#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 11:31:38 2022

Note you can do NLLs fitting for either time or rate. 
The difference is in extrema values and on the covariance parameters, if used
S(eta) = S0 * exp(-eta/T)
S(eta) = S0 * exp(-eta*R)

@author: pbolan
"""
import numpy as np
from scipy import optimize
from scipy import special
import warnings

#%% Simple model functions
def exp_decay_2p(eta, S0, R):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore') # easy to overflow exp()
        retval = S0 * np.exp(-R * eta)
    return retval

# These fits normalize internally
def fit_exp_nlls_2p(eta, ydata):
    
    null_return = {'S0':0, 'R':0, 'T':0} # anything goes bad, this is our return
    
    # Need at least 2 non-zero values
    if sum(ydata>0) <= 1:
        return null_return
    
    # Normalization
    y_mean = ydata.mean()
    
    # Use linear for initial guess. Bounds are only for starting
    S0_min = 0; S0_max = 1000
    R_min = 0.05; R_max = 110.0
    fit_lin = fit_exp_linear(eta, ydata)
    S0_0 = np.min([np.max([fit_lin['S0'], S0_min]), S0_max])
    R_0 = np.min([np.max([fit_lin['R'], R_min]), R_max])    
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')      
        
        try:
            params, params_covariance = optimize.curve_fit(
                exp_decay_2p, eta, ydata/y_mean, p0=[S0_0, R_0], maxfev=10000, 
                method='lm') #LM is default, bounds not allowed            
            S0 = params[0]*y_mean
            R = params[1]
            T = 1/R
            
        except RuntimeError as err:
            print(f'Caught {err}. ydata values:')
            print(ydata)
            return null_return

    return {'S0': S0, 
            'R': R,
            'T': T }

# This version has reasonable bounds on S0 and R
def fit_exp_nlls_2p_bound(eta, ydata):

    null_return = {'S0':0, 'R':0, 'T':0} # anything goes bad, this is our return
    
    # Need at least 2 non-zero values
    if sum(ydata>0) <= 1:
        return null_return
    
    # Normalization
    y_mean = ydata.mean()

    # Bounds. Hardwire these        
    S0_min = 0; S0_max = 1000
    
    # SEe notes on 20220904 and 20221003 for discussion on bounds choices
    # These shyould match whatever is used for training, the k_factor
    R_min = 0.25; R_max = 22
    bounds=([S0_min, R_min], [S0_max, R_max])
    
    # Use linear for initial guess. Within bounds
    fit_lin = fit_exp_linear(eta, ydata)
    S0_0 = np.min([np.max([fit_lin['S0'], S0_min]), S0_max])
    R_0 = np.min([np.max([fit_lin['R'], R_min]), R_max]) 
    
    # Best to underestimate R at first; overestimation can find local minima
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')   
        try: 
            params, params_covariance = optimize.curve_fit(
                exp_decay_2p, eta, ydata/y_mean, p0=[S0_0, R_0], maxfev=10000, 
                method='trf', bounds=bounds)
            S0 = params[0]*y_mean
            R = params[1]
            T = 1/R
            
        except RuntimeError as err:
            print(f'Caught {err}. ydata values:')
            print(ydata)
            return null_return
    
    return {'S0': S0, 
            'R': R,
            'T': T }


# Log-linear fitting
def fit_exp_linear(eta, ydata):
    
    # Solution for y = m*x + c, right from numpy.linalg.lstsq docs:
    A = np.vstack([eta, np.ones(len(eta))]).T
    
    # Adding a tiny offset to avoid log(0) errors
    #m, c = np.linalg.lstsq(A, np.log(ydata), rcond=None)[0]         
    m, c = np.linalg.lstsq(A, np.log(ydata + 1e-16), rcond=None)[0]     
    
    # Translate back to exp decay
    S0 = np.exp(c)
    T = -1/(m + 1e-24)
        
    return {'S0': S0, 
            'T': T,
            'R':1/T }  

# Restrict number to range
def clamp(val, vmin, vmax):
    return min(max(val, vmin), vmax)

'''
Here we fit to a Rician model following Bouhrara MRM 2015. Using their notation
There are three parameters here: S0, R, sigma
THis version is bounded, since that is the best of the NLLS methods
'''    
def fit_exp_rician(eta, ydata):
    
    # Because this one can be a little unstable, seed S0 and R with the lin son
    fit_lin = fit_exp_linear(eta, ydata)
    S0_0 = fit_lin['S0']
    R_0 = fit_lin['R']
    
    # Prep for normalization
    ymean = ydata.mean()
    
    # Bounds
    S0_min = 0; S0_max = 1000
    R_min = 0.25; R_max = 22
    sigma_min = 0; sigma_max = S0_max;
    bounds=([S0_min, R_min, sigma_min], [S0_max, R_max,sigma_max] )
    
    # Clamp starting values to bound range
    R_0 = clamp(R_0, R_min, R_max)
    S0_0 = clamp(S0_0/ymean, S0_min, S0_max)
    
    # Here we minimized the difference not between model and residual, but 
    # expectation value
    def exp_decay_rician_residual(x):
        
        Ate = x[0] * np.exp(-x[1] * eta)
        sigma = x[2]
        alpha = ( Ate/(2*sigma) )**2        
         
        # If numerically unstable, treat as gaussian
        if alpha.max()>50:
            # Gaussian condition
            #warnings.warn('Rician model unstable, reverting to Gaussian')
            Erice = Ate
        else:
            # Bouhrara Eq 2
            Erice = sigma * np.sqrt(np.pi/2) * np.exp(-alpha) * ((1+2*alpha) * special.iv(0, alpha) + 2*alpha*special.iv(1,alpha))
        
        # Note Eq [4]: Egaussian = Ate
        
        residual = ydata/ymean - Erice

        return residual         



    # Here's the actual calc
    kwargs = {'method': 'trf', 'ftol':1e-8}       
    x0=[S0_0, R_0, S0_0/10]
    if ymean==0:
        S0=0; R=1; T=1; sigma=0
    else:
        res_lsq = optimize.least_squares(exp_decay_rician_residual, x0=x0, bounds=bounds, **kwargs)
        
        # Extract parameters
        S0 = res_lsq['x'][0]
        R = res_lsq['x'][1]
        sigma = res_lsq['x'][2]
    
    return {'S0': S0 * ymean, 
            'R': R,
            'T': 1/R,
            'sigma': sigma * ymean}

