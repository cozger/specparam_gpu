# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:25:45 2025

@author: canoz
"""
import os


import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmin import least_squares
import torch.nn as nn
import time
from tqdm import tqdm

from specparam.gpu.funcs_torch import batched_residual_function_generic, torch_gaussian_function, torch_expo_nk_function
# -----------------------------
# 1. Definitions and Setup
# -----------------------------

#def gpu_curve_fit(functionWIP,freqs,spectra,guessWIP,batch_size):
def gpu_curve_fit(freqs,spectra,batch_size,func,guess=None):
    
    def compute_guess_flat(x_batch, y_batch):
        """
        Compute the initial guess for the aperiodic parameters for each trial in the batch.
        
        Parameters
        ----------
        x_batch : torch.Tensor
            Tensor of shape (current_batch_size, n_freq) containing the frequency values.
        y_batch : torch.Tensor
            Tensor of shape (current_batch_size, n_freq) containing the power spectrum for each trial.
        
        Returns
        -------
        guess_flat : torch.Tensor
            A 1D tensor of shape (current_batch_size * 2,) containing the initial guess
            for each trial, where each guess is [offset, exp].
        """
        # Offset guess: take the first frequency value of y_batch for each trial.
        # Shape: (current_batch_size,)
        offset_guess = y_batch[:, 0]
        
        # Denominator: difference in log10 frequency values (assumed to be the same for every trial).
        # Use the first row of x_batch.
        denom = torch.log10(x_batch[0, -1]) - torch.log10(x_batch[0, 0])
        
        # Exponent guess: absolute difference between last and first y value divided by denom.
        # Shape: (current_batch_size,)
        exp_guess = torch.abs((y_batch[:, -1] - y_batch[:, 0]) / denom)
        
        # Stack the guesses for each trial along the second dimension.
        # The resulting shape is (current_batch_size, 2)
        guess_tensor = torch.stack([offset_guess, exp_guess], dim=1)
        
        # Flatten the guess tensor to a 1D tensor of shape (current_batch_size * 2,)
        guess_flat = guess_tensor.flatten()
        #print('COMPUTED EXP GUESS',exp_guess.shape)
        #print('COMPUTED OFFSET GUESS',offset_guess.shape)
        return guess_flat
    
    # -----------------------------
    # Helper: Convert input to a torch tensor of type double.
    # -----------------------------
    def _to_tensor(x, device=None):
        if isinstance(x, np.ndarray):
            tensor = torch.from_numpy(x).double()
        elif torch.is_tensor(x):
            tensor = x.double()
        else:
            raise ValueError("Input must be a numpy array or torch tensor.")
        if device is not None:
            tensor = tensor.to(device)
        return tensor
        
    # -----------------------------
    # Helper: Flatten a list of guess tensors using torch.
    # -----------------------------
    def flatten_guess_list_torch(guess_list, device):
         """
         Given a list of guess arrays (one per trial), flatten them into a single tensor
         and return a tensor of markers for where each guess came from.
         
         Parameters
         ----------
         guess_list : list
             Each element is an array-like (either a numpy.ndarray or a torch.Tensor)
             of shape (n_peaks, 3) for that trial.
         device : torch.device
             The device to use for the output tensor.
             
         Returns
         -------
         markers : torch.Tensor
             A tensor of shape (N, 2), where each row is [trial_index, peak_index].
         flat_guesses : torch.Tensor
             Tensor of shape (N, 3), where N is the total number of valid guesses.
         """
         markers = []
         flat_guess_list = []
         for trial_idx, g in enumerate(guess_list):
             # Convert g to a torch tensor if it is not already.
             if not torch.is_tensor(g):
                 g = torch.from_numpy(g).double().to(device)
             if g.numel() == 0:
                 continue
             # g should have shape (n_peaks, 3)
             for peak_idx in range(g.shape[0]):
                 markers.append([trial_idx, peak_idx])
                 flat_guess_list.append(g[peak_idx])
         if len(flat_guess_list) == 0:
             flat_guesses = torch.empty((0, 3), dtype=torch.float64, device=device)
             markers_tensor = torch.empty((0, 2), dtype=torch.long, device=device)
         else:
             flat_guesses = torch.stack(flat_guess_list, dim=0)
             markers_tensor = torch.tensor(markers, dtype=torch.long, device=device)
         return markers_tensor, flat_guesses
     

    
    #Set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cpu')
    print(f'Using device: {device}')
    
    #Prepare a torch tensor for freqs (used in the torch methods)    
    print(f'SPECTRA TYPE: {spectra.dtype}')

    # spectra_torch = torch.from_numpy(spectra.copy()).double().to(device)
    # freq_torch = _to_tensor(freqs, device=device)
    
    spectra_torch=spectra
    freq_torch=freqs

    # print(f'SPECTRA TYPE222: {spectra.dtype}')
    # torch.cuda.empty_cache()  # clean up memory
    
    #torch.cuda.empty_cache() #need to add other clean up functions later
    # -----------------------------
    # 2. Iterate over each time point of channel spectra
    # -----------------------------
    
    # Lists to store the fitted curves (one per spectrum) for each method.
    torchmin_fits_batch = []    # Torchmin least_squares fitted curves of each batch
    
    
    #Initialize depending on desired function
    if func == 'gaussian':
        func = 'gaussian'
        num_params = 3
        model_func = torch_gaussian_function  # defined earlier
        print('SELECTED GAUSSIAN')    
        batch_size=batch_size
        print('Type of batch_size:', type(batch_size))
        print('Type of guess:', type(guess))

        print('BATCH INPUT SIZE',batch_size)
        print('GUESS INPUT SIZE',len(guess))
        
        
        n_trials = spectra_torch.shape[1]
        n_freq = spectra_torch.shape[0]
    
        # Flatten the guess list using torch.
        markers, flat_guess_tensor = flatten_guess_list_torch(guess, device=device)
        n_valid = flat_guess_tensor.shape[0]
        print('N_VALID SIZE',n_valid)
        if n_valid == 0:
            fitted_params_full = torch.full((n_trials, 3), float('nan'), dtype=torch.float64, device=device)
            fitted_curves_full = torch.full((n_freq, n_trials), float('nan'), dtype=torch.float64, device=device)
            return fitted_params_full, fitted_curves_full, markers
        
        # For each valid guess, extract the corresponding trial's y-data.
        valid_trial_indices = [m[0] for m in markers]  # list of trial indices for each valid guess
        x_valid = freq_torch.unsqueeze(0).repeat(n_valid, 1)  # shape: (n_valid, n_freq)
        y_valid = spectra_torch[:, valid_trial_indices].T         # shape: (n_valid, n_freq)
        
        # Process the valid guesses in batches.
        fitted_params_list = []
        fitted_curves_list = []
        current_index = 0

        # Iterate over batches using a for loop
        for current_index in tqdm(range(0, n_valid, batch_size), desc="Batched fitting"):
            tqdm.write(f"Processing batch starting at index {current_index}")
            tqdm.write(f"Current index: {current_index}")

            
            current_batch_size = min(batch_size, n_valid - current_index)
            tqdm.write(f"CURRENT BATCH SIZE: {current_batch_size}")

            x_batch = x_valid[current_index: current_index + current_batch_size]  # (batch_size, n_freq)
            y_batch = y_valid[current_index: current_index + current_batch_size]  # (batch_size, n_freq)
            batch_guess_tensor = flat_guess_tensor[current_index: current_index + current_batch_size]  # (batch_size, 3)
            tqdm.write(f"CURRENT BATCH GUESS SIZE: {batch_guess_tensor.shape}")
            # Flatten the guess tensor for this batch.
            guess_flat = batch_guess_tensor.flatten()  # shape: (batch_size * 3,)
            print('GUESS FLAT',guess_flat.shape)
            # Define the residual function using the generic batched residual function.
            residual_func = lambda params_flat: batched_residual_function_generic(
                params_flat, current_batch_size, x_batch, y_batch, num_params, torch_gaussian_function
            )
            
            bounds = (-float('inf'), float('inf'))
            result = least_squares(residual_func,
                                    guess_flat,
                                    max_nfev=50,
                                    method='trf',
                                    bounds=bounds,
                                    gtol=1e-8, xtol=1e-8, ftol=1e-8,
                                    verbose=0,
                                    tr_solver='exact')
            fitted_params = result.x.view(current_batch_size, num_params)
            fitted_params_list.append(fitted_params.detach())
            
            # Compute fitted curves using the Gaussian model.
            fitted_curves = torch_gaussian_function(x_batch,
                                                    fitted_params[:, 0],
                                                    fitted_params[:, 1],
                                                    fitted_params[:, 2])
            fitted_curves_list.append(fitted_curves.detach())
            
            
        # Concatenate results for valid guesses.
        fitted_params_valid = torch.cat(fitted_params_list, dim=0)   # shape: (n_valid, 3)
        fitted_curves_valid = torch.cat(fitted_curves_list, dim=0)     # shape: (n_valid, n_freq)
        
        # Reassemble full-channel outputs.
        fitted_params_full = torch.full((n_trials, 3), float('nan'), dtype=torch.float64, device=device)
        fitted_curves_full = torch.full((n_freq, n_trials), float('nan'), dtype=torch.float64, device=device)
        
        fitted_params_full[valid_trial_indices, :] = fitted_params_valid
        fitted_curves_full[:, valid_trial_indices] = fitted_curves_valid.T
        # # For each valid guess, assign the fitted result into the proper trial index.
        # for i, trial_idx in enumerate(valid_trial_indices):
        #     fitted_params_full[trial_idx, :] = fitted_params_valid[i]
        #     fitted_curves_full[:, trial_idx] = fitted_curves_valid[i]
        
        return fitted_params_full, fitted_curves_full

            
        
        
    elif func == 'aperiodic':
        func = 'aperiodic'
        num_params = 2 
        aperiodic_params_batch = []
        model_func=torch_expo_nk_function
        # We use the imported batched_residual_function from our module.
        residual_func = lambda params_flat: batched_residual_function_generic(params_flat, current_batch_size, x_batch, y_batch,num_params, model_func)
        
        batch_size = batch_size
        n_trials = spectra_torch.shape[1]  # total number of trials

    
    for batch_idx in tqdm(range(0, n_trials, batch_size)):
        
        # Determine the current batch size (last batch might be smaller)
        current_batch_size = min(batch_size, n_trials - batch_idx)
                    
       
        x_batch = freq_torch.unsqueeze(0).repeat(current_batch_size, 1)  # shape: (current_batch_size, 55)
        y_batch = spectra_torch[:, batch_idx: batch_idx + current_batch_size].T  # shape: (current_batch_size, 55)
            #print('X BATCH',x_batch.shape)
            #print('Y BATCH',y_batch.shape)
            
        guess_flat = compute_guess_flat(x_batch, y_batch)
        #print('GUESS FLATT',guess_flat.shape)
        bounds = (-float('inf'), float('inf'))
       

        f0 = residual_func(guess_flat)
        # print("Initial residuals f0:", f0)
        # print("Are all finite?", f0.isfinite().all())
        # Check which entries are not finite
        # non_finite_indices = torch.nonzero(~f0.isfinite(), as_tuple=False)
        # if non_finite_indices.numel() > 0:
        #     print("Non-finite residual indices:", non_finite_indices)
        #     print("Non-finite residual values:", f0[non_finite_indices])

        # Run torchmin least_squares with TRF on the batched data.
        result = least_squares(residual_func,
                                guess_flat,
                                max_nfev=5000, method='trf',
                                bounds=bounds,
                                #x_scale='jac',
                                gtol=1e-15, xtol=1e-15, ftol=1e-15,
                                verbose=0,
                                tr_solver='exact') #this doesn't seem to change results despite 'exact' being much more expensive
        #print('GUESS in LSQ',guess_flat.shape)
    

        # Reshape the fitted parameters back to (current_batch_size, 2)
        with torch.no_grad():
    
            fitted_params = result.x.view(current_batch_size, 2)
            
            aperiodic_params_batch.append(fitted_params.detach())
            aperiodic_params_full = torch.cat(aperiodic_params_batch, dim=0)
            # aperiodic_params_full = aperiodic_params_full.cpu().numpy()
    
    
        # else:
        #     raise ValueError("func must be 'gaussian' or 'aperiodic'")

        



        #------------------------Curve Fitting-----------------------------------------------
        
        #fits the curves (predictions), vectorized across the batch for performance
        
        #Is the log here an issue?
        
        preds = fitted_params[:, 0].unsqueeze(1) - torch.log10(torch.pow(freq_torch.unsqueeze(0), fitted_params[:, 1].unsqueeze(1)))
        with torch.no_grad():
    
            # Append the whole batch
            torchmin_fits_batch.append(preds.detach())
            # After the loop, concatenate on the GPU:
            torchmin_fits_full = torch.cat(torchmin_fits_batch, dim=0)
            # torchmin_fits_full = torchmin_fits_full.cpu().numpy()
                         




    
    return aperiodic_params_full, torchmin_fits_full.T
        # Final cleanup: remove the long-lived tensors and empty cache.
    # del spectra_torch, freq_torch, torchmin_fits_batch, preds, aperiodic_params_batch, aperiodic_params_full, torchmin_fits_full, result
#    torch.cuda.empty_cache()
    # import gc
    # gc.collect()
    


            