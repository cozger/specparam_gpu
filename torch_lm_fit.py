# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:28:32 2025

@author: canoz
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import torch_levenberg_marquardt as tlm
import gc
from specparam.gpu.funcs_torch import torch_gaussian_function
import io
import contextlib
from torch import Tensor


def validate_flatten_guess_list(guess_list, device):
    """
    Given a list of guess arrays (one per trial), flatten them into a single tensor.
    
    Each element of guess_list may be:
      - A 1D tensor/array of shape (3,) representing a single guess, or 
      - A 2D tensor/array of shape (n_peaks, 3) representing multiple guesses.
      
    This function normalizes each guess so that a single guess always becomes a tensor of
    shape (1, 3) and then stacks them along the 0-th dimension.
    
    Parameters
    ----------
    guess_list : list
        Each element is array-like (either a numpy.ndarray or a torch.Tensor).
    device : torch.device
        The device to use for the output tensor.
        
    Returns
    -------
    flat_guesses : torch.Tensor
        A tensor of shape (N, 3), where N is the total number of valid guesses.
    """
    flat_guess_list = []
    for g in guess_list:
        # Convert g to tensor if not already.
        if not torch.is_tensor(g):
            g = torch.from_numpy(np.array(g, dtype=float)).double().to(device)
        if g.numel() == 0:
            continue
        # Remove extra singleton dimensions.
        g = g.squeeze()
        # If g is 1D (a single guess), convert to shape (1, 3)
        if g.ndim == 1:
            if g.shape[0] != 3:
                raise ValueError(f"Expected a guess of length 3, got length {g.shape[0]}")
            normalized = g.unsqueeze(0)  # shape: (1, 3)
            flat_guess_list.append(normalized)
        else:
            # Otherwise assume g has shape (n_peaks, 3)
            if g.shape[1] != 3:
                raise ValueError(f"Expected guesses of shape (_, 3), got shape {g.shape}")
            for peak_idx in range(g.shape[0]):
                guess_tensor = g[peak_idx]
                if guess_tensor.ndim == 1:
                    guess_tensor = guess_tensor.unsqueeze(0)
                flat_guess_list.append(guess_tensor)
    if len(flat_guess_list) == 0:
        flat_guesses = torch.empty((0, 3), dtype=torch.float64, device=device)
    else:
        # Stack all guesses along dimension 0; each guess has shape (1, 3)
        stacked = torch.cat(flat_guess_list, dim=0)
        flat_guesses = stacked.view(-1, 3)
    return flat_guesses

 

def get_guesses_from_list(guess_list, device):
    """
    Process a list of per-trial guesses (each of which can be an array of shape (n_peaks, 3)
    or a 1D array of shape (3,)). If a trial is empty, replace it with a tensor of shape (1,3)
    filled with NaNs. Otherwise, keep all rows.
    
    Returns a list of tensors, one per trial, each having shape (n, 3) with n >= 1.
    Also returns a list of valid trial indices (which, in this case, is all indices).
    """
    processed_list = []
    valid_indices = []
    for i, g in enumerate(guess_list):
        if not torch.is_tensor(g):
            try:
                t = torch.from_numpy(np.array(g, dtype=float)).double().to(device)
            except Exception as e:
                t = torch.empty((0, 3), dtype=torch.float64, device=device)
        else:
            t = g.double().to(device)
        # If empty, force it to shape (1,3) filled with NaNs.
        if t.numel() == 0:
            t = torch.full((1, 3), float('nan'), dtype=torch.float64, device=device)
        else:
            # Do not squeeze everything—if it is 1D, unsqueeze to make it 2D.
            if t.ndim == 1:
                if t.shape[0] != 3:
                    raise ValueError(f"Expected a guess of length 3, got length {t.shape[0]} for trial {i}")
                t = t.unsqueeze(0)  # now shape (1,3)
            elif t.ndim == 2:
                # Leave it as is (could be (1,3), (2,3), etc.)
                pass
            else:
                # For higher dims, collapse extra dimensions into the first dimension.
                t = t.reshape(-1, 3)
        processed_list.append(t)
        valid_indices.append(i)
    return processed_list, valid_indices

class VectorExpoNKModel(nn.Module):
    def __init__(self, init_offsets, init_exps):
        """
        Parameters:
          init_offsets, init_exps: 1D torch tensors of shape (batch_size,)
          (Here, batch_size corresponds to the number of trials.)
        """
        super().__init__()
        # Store parameters as (batch_size, 1)
        self.offset = nn.Parameter(init_offsets.unsqueeze(1))  # shape: (full_batch_size, 1)
        # print('SELF OFFSET', self.offset.shape)
        self.exp = nn.Parameter(init_exps.unsqueeze(1))          # shape: (full_batch_size, 1)
        # print('SELF EXP', self.exp.shape)

    def forward(self, xs):
        """
        xs: tensor of shape (current_batch_size, n_freq)
        We expect that the full model was created for the entire dataset (e.g., 1000 trials).
        When a mini-batch (with size < full batch size) is passed in, we slice the parameters.
        """
        current_bs = xs.shape[0]
        full_bs = self.offset.shape[0]
        if current_bs != full_bs:
            # Here we assume that the mini-batch corresponds to the first current_bs samples.
            # (For a more robust solution, you could have the dataset return indices.)
            offset = self.offset[:current_bs]
            exp = self.exp[:current_bs]
        else:
            offset = self.offset
            exp = self.exp

        # print('XS SHAPE', xs.shape)
        # Compute: offset - log10(xs ** exp)
        return offset - torch.log10(torch.pow(xs, exp))


class MaskedMSELoss(nn.Module):
    def __init__(self) -> None:
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Computes the masked mean squared error loss. Only valid (non-NaN) entries in y_true
        are considered in the loss computation.
        """
        mask = ~torch.isnan(y_true)
        # If no valid entries, return 0 to avoid division by zero.
        if torch.sum(mask) == 0:
            return torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)
        # Compute the squared error only where mask is True.
        # Then, sum over valid entries and divide by the count.
        loss = ((y_pred - y_true).square() * mask).sum() / mask.sum()
        return loss

    def residuals(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Computes the residuals (errors) for each valid element. Returns a flattened vector
        of errors for entries where y_true is not NaN.
        """
        mask = ~torch.isnan(y_true)
        res = (y_pred - y_true) * mask
        return res.view(-1)


# ---------------------------
# Vectorized LM Fitting for Aperiodic Model
# ---------------------------
def gpu_curve_fit_lm_vectorized_aperiodic(freqs, spectra, batch_size, func, guess=None):
    """
    Vectorized LM fitting for the aperiodic (no-knee) model.
    
    Inputs:
      - freqs: 1D numpy array of frequency values (length n_freq).
      - spectra: 2D numpy array of power spectra with shape [n_freq, n_trials].
      - batch_size: Maximum number of trials to process per mini-batch.
      - func: (unused; kept for interface compatibility)
      - guess: Either None or a list (length n_trials) where each element is a 2-element array-like
               [offset, exp]. If provided, these override the computed initial guesses.
    
    Returns:
      - fitted_params: tensor of shape (n_trials, 2) containing [offset, exp] for each trial.
      - fitted_curves: tensor of shape (n_trials, n_freq) containing the modeled spectra.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assume spectra and freqs are already torch tensors on the appropriate device.
    spectra_torch = spectra
    freq_torch = freqs
    
    n_trials = spectra_torch.shape[1]
    n_freq = spectra_torch.shape[0]
    
    # --- Compute initial guesses for each trial ---
    if guess is None:
        # Use first value in each trial for offset.
        init_offsets = spectra_torch[0, :]  # shape: (n_trials,)
        # Compute exponent using the difference between the last and first values, normalized by frequency range in log10 space.
        denom = torch.log10(freq_torch[-1]) - torch.log10(freq_torch[0])
        init_exps = torch.abs((spectra_torch[-1, :] - spectra_torch[0, :]) / denom)  # shape: (n_trials,)
    else:
        init_offsets = torch.tensor([g[0] for g in guess], dtype=torch.float64, device=device)
        init_exps    = torch.tensor([g[1] for g in guess], dtype=torch.float64, device=device)
    
    # --- Global parameters (one per trial) ---
    global_offsets = init_offsets.clone()  # shape: (n_trials,)
    global_exps = init_exps.clone()          # shape: (n_trials,)
    
    # --- Create a reusable model sized to the maximum mini-batch (i.e. batch_size) ---
    # Pass raw 1D tensors so that the model's constructor applies the unsqueeze once.
    reusable_model = VectorExpoNKModel(global_offsets[:batch_size],
                                       global_exps[:batch_size]).to(device)
    
    # Loss function and LM module.
    loss_fn = MaskedMSELoss()  # or tlm.loss.MSELoss() if preferred.
    lm_module = tlm.training.LevenbergMarquardtModule(
        model=reusable_model,
        loss_fn=loss_fn,
        learning_rate=0.1,
        attempts_per_step=1000,
        solve_method='cholesky'
    )
    
    # --- Build batched inputs for all trials ---
    # x_batch: common frequency vector repeated for each trial.
    x_batch = freq_torch.unsqueeze(0).repeat(n_trials, 1)  # (n_trials, n_freq)
    # y_batch: each trial's spectrum as a row.
    y_batch = spectra_torch.transpose(0, 1)  # (n_trials, n_freq)
    
    # --- Process trials in mini-batches ---
    fitted_params_list = []
    fitted_curves_list = []
    n_batches = (n_trials + batch_size - 1) // batch_size
    
    for start in range(0, n_trials, batch_size):
        end = min(start + batch_size, n_trials)
        current_bs = end - start

        # Update the reusable model's parameters in-place (only the first current_bs rows).
        reusable_model.offset.data[:current_bs].copy_(global_offsets[start:end].unsqueeze(1))
        reusable_model.exp.data[:current_bs].copy_(global_exps[start:end].unsqueeze(1))
        
        # Extract mini-batch inputs.
        x_chunk = x_batch[start:end]  # (current_bs, n_freq)
        y_chunk = y_batch[start:end]  # (current_bs, n_freq)
        
        # If the loss function supports an expected batch size, update it.
        if hasattr(loss_fn, 'expected_batch_size'):
            loss_fn.expected_batch_size = current_bs

        # Reset LM internal state.
        lm_module.reset()
        
        # Optionally, check for NaNs in the parameters.
        if (torch.isnan(reusable_model.offset[:current_bs]).any() or
            torch.isnan(reusable_model.exp[:current_bs]).any()):
            nan_params = torch.full((current_bs, 2), float('nan'), dtype=torch.float64, device=device)
            nan_curves = torch.full((current_bs, n_freq), float('nan'), dtype=torch.float64, device=device)
            fitted_params_list.append(nan_params)
            fitted_curves_list.append(nan_curves)
            continue
        
        # Create a temporary dataset and fast data loader for this mini-batch.
        train_dataset = torch.utils.data.TensorDataset(x_chunk, y_chunk)
        train_loader = tlm.utils.FastDataLoader(
            train_dataset,
            batch_size=current_bs,
            repeat=1,
            shuffle=False,
            device=device
        )
        
        # Use an asynchronous CUDA stream for the LM fit.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            tlm.utils.fit(
                training_module=lm_module,
                dataloader=train_loader,
                epochs=3,
                overwrite_progress_bar=False
            )
        torch.cuda.synchronize()
        
        with torch.no_grad():
            # Retrieve fitted parameters (each of shape (current_bs, 1)) and concatenate along dim=1 to get (current_bs, 2).
            chunk_params = torch.cat((reusable_model.offset[:current_bs],
                                      reusable_model.exp[:current_bs]), dim=1)
            # Evaluate the fitted model on the mini-batch.
            chunk_curves = reusable_model(x_chunk)  # (current_bs, n_freq)
        
        fitted_params_list.append(chunk_params.detach().cpu())
        fitted_curves_list.append(chunk_curves.detach().cpu())
        
        # Update the corresponding slice of the global parameters.
        global_offsets[start:end] = reusable_model.offset[:current_bs].detach().squeeze(1)
        global_exps[start:end] = reusable_model.exp[:current_bs].detach().squeeze(1)
    
    # Concatenate results from all mini-batches.
    fitted_params = torch.cat(fitted_params_list, dim=0)  # (n_trials, 2)
    fitted_curves = torch.cat(fitted_curves_list, dim=0)      # (n_trials, n_freq)
    
    return fitted_params, fitted_curves  # Optionally, call .cpu().numpy() if needed.

        