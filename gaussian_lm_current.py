# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:52:28 2025

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
from torch.utils.data import TensorDataset, DataLoader

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
        A tensor of shape (n_with_peaks, 3), where n_with_peaks is the total number of valid guesses.
    """
    flat_guess_list = []
    for g in guess_list:
        # Convert to tensor if needed.
        if not torch.is_tensor(g):
            g = torch.from_numpy(np.array(g, dtype=float)).double().to(device)
        if g.numel() == 0:
            continue
        g = g.squeeze()
        if g.ndim == 1:
            if g.shape[0] != 3:
                raise ValueError(f"Expected guess of length 3, got {g.shape[0]}")
            flat_guess_list.append(g.unsqueeze(0))
        else:
            if g.shape[1] != 3:
                raise ValueError(f"Expected guesses with 3 columns, got {g.shape}")
            for peak in g:
                flat_guess_list.append(peak.unsqueeze(0))
    if len(flat_guess_list) == 0:
        return torch.empty((0, 3), dtype=torch.float64, device=device)
    return torch.cat(flat_guess_list, dim=0).view(-1, 3)


def get_guesses_from_list(guess_list, device):
    """
    Process a list of per-trial guesses (each of which can be an array of shape (n_peaks, 3)
    or a 1D array of shape (3,)). If a trial is empty, replace it with a tensor of shape (1,3)
    filled with NaNs.
    
    Returns:
       - processed_list: list of tensors (each of shape (n, 3), n>=1)
       - valid_indices: list of trial indices (here, all indices are returned)
    """
    processed_list = []
    valid_indices = []
    for i, g in enumerate(guess_list):
        if not torch.is_tensor(g):
            try:
                t = torch.from_numpy(np.array(g, dtype=float)).double().to(device)
            except Exception:
                t = torch.empty((0, 3), dtype=torch.float64, device=device)
        else:
            t = g.double().to(device)
        if t.numel() == 0:
            t = torch.full((1, 3), float('nan'), dtype=torch.float64, device=device)
        else:
            if t.ndim == 1:
                if t.shape[0] != 3:
                    raise ValueError(f"Expected guess of length 3, got {t.shape[0]} for trial {i}")
                t = t.unsqueeze(0)
            elif t.ndim > 2:
                t = t.reshape(-1, 3)
        processed_list.append(t)
        valid_indices.append(i)
    return processed_list, valid_indices


class VectorGaussianNKModel(nn.Module): 
    def __init__(self, init_centers, init_heights, init_widths):
        """
        Parameters:
          init_centers, init_heights, init_widths: 1D torch tensors of shape (batch_size,)
        """
        super().__init__()
        self.center = nn.Parameter(init_centers.unsqueeze(1))  # shape: (batch_size, 1)
        self.height = nn.Parameter(init_heights.unsqueeze(1))  # shape: (batch_size, 1)
        self.width  = nn.Parameter(init_widths.unsqueeze(1))   # shape: (batch_size, 1)

    def forward(self, xs):
        # If xs is a mini-batch (batch size < full batch), slice parameters accordingly.
        current_bs = xs.shape[0]
        if current_bs != self.center.shape[0]:
            center = self.center[:current_bs]
            height = self.height[:current_bs]
            width  = self.width[:current_bs]
        else:
            center, height, width = self.center, self.height, self.width
            
        model_output = torch_gaussian_function(xs, center, height, width)
        return model_output.squeeze(-1)
    

class GaussianMSELoss(nn.Module):
    def __init__(self, expected_batch_size: int):
        super(GaussianMSELoss, self).__init__()
        self.expected_batch_size = expected_batch_size

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return (y_pred - y_true).square().mean()

    def residuals(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        res = (y_pred - y_true).view(-1, 1)
        current_bs = y_true.shape[0]
        if current_bs < self.expected_batch_size:
            factor = self.expected_batch_size // current_bs
            res = res.repeat(factor, 1)
        return res


def gpu_vectorized_gaussian_two(freqs, spectra, batch_size, func, guess):
    """
    Vectorized Levenberg-Marquardt fitting for a Gaussian model using mini-batch processing.
    
    Inputs:
      - freqs: 1D numpy array of frequency values (length n_freq).
      - spectra: 2D numpy array of power spectra with shape [n_freq, n_trials].
      - batch_size: Number of flattened guesses to process per mini-batch.
      - func: (unused here; typically 'gaussian').
      - guess: List (length n_trials) where each element is either a 1D array (shape (3,))
               or a 2D array (shape (n_peaks, 3)).
    
    Returns:
      - periodic_params: numpy array of shape (n_trials, max_guesses, 3)
      - periodic_curves: numpy array of shape (n_trials, max_guesses, n_freq)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert spectra and freqs to torch tensors.
    # spectra_cpu = torch.from_numpy(spectra.copy()).double().pin_memory()
    # spectra_torch = spectra_cpu.to(device, non_blocking=True)
    
    # freq_cpu = torch.from_numpy(freqs.copy()).double().pin_memory()
    # freq_torch = freq_cpu.to(device, non_blocking=True)

    spectra_torch=spectra
    freq_torch=freqs
    
    n_freq = spectra_torch.shape[0]
    n_trials = spectra_torch.shape[1]
    
    # Process guess list and record trial indices.
    processed_guess_list, valid_indices = get_guesses_from_list(guess, device)
    
    # --- Build flat lists of valid guesses (non-NaN) and their corresponding trial indices ---
    flat_guess_list = []
    trial_indices_list = []
    for t, guess_tensor in zip(valid_indices, processed_guess_list):
        # Skip trial if all entries are NaN.
        if torch.isnan(guess_tensor).all():
            continue
        # Otherwise, iterate over each guess in the trial.
        for j in range(guess_tensor.shape[0]):
            guess_j = guess_tensor[j].unsqueeze(0)  # shape: (1, 3)
            if torch.isnan(guess_j).all():
                tqdm.write("No Guesses Found; skipping curve fitting for this trial.")
                continue  # skip if this guess is NaN
            flat_guess_list.append(guess_j)
            trial_indices_list.append(t)
    
    if len(flat_guess_list) == 0:
        # If no valid guesses, return NaN arrays.
        periodic_params = torch.full((n_trials, 1, 3), float('nan'), dtype=torch.float64, device=device)
        periodic_curves = torch.full((n_trials, 1, n_freq), float('nan'), dtype=torch.float64, device=device)
        return periodic_params.cpu().numpy(), periodic_curves.cpu().numpy()
    
    flat_guess_tensor = torch.cat(flat_guess_list, dim=0)  # (N, 3) where n_with_peaks is total valid guesses.
    n_with_peaks = flat_guess_tensor.shape[0]
    
    # Build x_batch and y_batch: each row in x_batch is the common frequency vector.
    x_batch = freq_torch.unsqueeze(0).repeat(n_with_peaks, 1)  # (n_with_peaks, n_freq)
    y_list = []
    for t in trial_indices_list:
        y_list.append(spectra_torch[:, t].unsqueeze(0))  # (1, n_freq)
    y_batch = torch.cat(y_list, dim=0)  # (n_with_peaks, n_freq)
    
    # Extract initial parameters.
    # Extract initial parameters.
    init_centers = flat_guess_tensor[:, 0]
    init_heights = flat_guess_tensor[:, 1]
    init_widths  = flat_guess_tensor[:, 2]
    global_centers = init_centers.clone()  # shape: (n_with_peaks,)
    global_heights = init_heights.clone()
    global_widths  = init_widths.clone()
    
    # Create a reusable model with parameters sized to the maximum mini-batch (i.e. batch_size).
    # We take the first `batch_size` elements of the global parameters.
    reusable_model = VectorGaussianNKModel(
        global_centers[:batch_size],
        global_heights[:batch_size],
        global_widths[:batch_size]
    ).to(device)
    loss_fn = GaussianMSELoss(expected_batch_size=batch_size)
    
    lm_module = tlm.training.LevenbergMarquardtModule(
        model=reusable_model,
        loss_fn=loss_fn,
        learning_rate=0.1,
        attempts_per_step=1000,
        solve_method='cholesky',
        use_vmap=True
    )
    
    stream = torch.cuda.Stream()
    fitted_params_list = []
    fitted_curves_list = []
    n_batches = (n_with_peaks + batch_size - 1) // batch_size
    
    for start in range(0, n_with_peaks, batch_size):
        end = min(start + batch_size, n_with_peaks)
        current_bs = end - start
    
        # Update only the first current_bs rows of the reusable model's parameters.
        reusable_model.center.data[:current_bs].copy_(global_centers[start:end].unsqueeze(1))
        reusable_model.height.data[:current_bs].copy_(global_heights[start:end].unsqueeze(1))
        reusable_model.width.data[:current_bs].copy_(global_widths[start:end].unsqueeze(1))
    
        x_chunk = x_batch[start:end]
        y_chunk = y_batch[start:end]
    
        # Update the loss function's expected batch size.
        loss_fn.expected_batch_size = current_bs
        lm_module.reset()  # Reset LM internal state.
    
        # Skip the batch if any of the first current_bs parameters are NaN.
        if (torch.isnan(reusable_model.center[:current_bs]).any() or 
            torch.isnan(reusable_model.height[:current_bs]).any() or 
            torch.isnan(reusable_model.width[:current_bs]).any()):
            nan_params = torch.full((current_bs, 3), float('nan'), dtype=torch.float64, device=device)
            nan_curves = torch.full((current_bs, n_freq), float('nan'), dtype=torch.float64, device=device)
            fitted_params_list.append(nan_params)
            fitted_curves_list.append(nan_curves)
            tqdm.write("Batch contains NaN parameters; skipping curve fitting for this batch.")
            continue
    
        with torch.cuda.stream(stream):
            batch_iter = [(x_chunk, y_chunk)]
            tlm.utils.fit(
                training_module=lm_module,
                dataloader=batch_iter,
                epochs=3,
                overwrite_progress_bar=False,
                update_every_n_steps=1
            )
    
        with torch.no_grad():
            # Use only the first current_bs parameters when retrieving results.
            chunk_params = torch.cat((
                reusable_model.center[:current_bs],
                reusable_model.height[:current_bs],
                reusable_model.width[:current_bs]
            ), dim=1)  # Expected shape: (current_bs, 3)
            chunk_curves = reusable_model(x_chunk)  # Expected shape: (current_bs, n_freq)
    
        fitted_params_list.append(chunk_params.detach().cpu())
        fitted_curves_list.append(chunk_curves.mean(dim=1).detach().cpu())
    
        # Update the corresponding global parameters.
        global_centers[start:end] = reusable_model.center[:current_bs].detach().squeeze(1)
        global_heights[start:end] = reusable_model.height[:current_bs].detach().squeeze(1)
        global_widths[start:end]  = reusable_model.width[:current_bs].detach().squeeze(1)
    
    torch.cuda.synchronize()
    
    fitted_params_flat = torch.cat(fitted_params_list, dim=0)  # shape: (n_with_peaks, 3)
    fitted_curves_flat = torch.cat(fitted_curves_list, dim=0)      # shape: (n_with_peaks, n_freq)
    
    # --- Reassemble results by trial ---
    trial_params_dict = {i: [] for i in range(n_trials)}
    trial_curves_dict = {i: [] for i in range(n_trials)}
    for i in range(n_with_peaks):
        t_idx = trial_indices_list[i]
        trial_params_dict[t_idx].append(fitted_params_flat[i])
        trial_curves_dict[t_idx].append(fitted_curves_flat[i])
    
    max_guesses = max((len(v) for v in trial_curves_dict.values() if isinstance(v, list)), default=0)
    periodic_params = torch.full((n_trials, max_guesses, 3), float('nan'), dtype=torch.float64)
    periodic_curves = torch.full((n_trials, max_guesses, n_freq), float('nan'), dtype=torch.float64)
    
    for t in range(n_trials):
        if len(trial_params_dict[t]) > 0:
            params_tensor = torch.stack(trial_params_dict[t], dim=0)
            curves_tensor = torch.stack(trial_curves_dict[t], dim=0)
            periodic_params[t, :params_tensor.shape[0], :] = params_tensor
            periodic_curves[t, :curves_tensor.shape[0], :] = curves_tensor
    
    return periodic_params, periodic_curves  # Optionally, call .cpu().numpy()
# Optionally call .cpu().numpy()
  #.cpu().numpy()
    # del spectra_torch, freq_torch, model, module_lm, train_dataset, train_loader
