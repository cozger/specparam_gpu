# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:15:21 2025

@author: canoz
"""

import torch
import numpy as np
from specparam.gpu.funcs_torch import torch_compute_fwhm, torch_compute_gauss_std
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------------------------------------
# Helper: Drop peaks near the edges of the frequency range
# -------------------------------------------------------
def torch_drop_peak_cf(guess, freq_range, bw_std_edge):
    """
    Drop peaks whose center frequencies are too close to the edge of the frequency range.

    Parameters
    ----------
    guess : torch.Tensor
        A tensor of shape [n_peaks, 3] where each row is [center, height, std].
    freq_range : tuple
        (lower_bound, upper_bound) of the frequency range.
    bw_std_edge : float
        Multiplicative factor applied to the guessed standard deviation to define a “drop”
        distance from the edge.

    Returns
    -------
    torch.Tensor
        Filtered tensor of guess parameters.
    """
    # Extract the center frequency (cf) and compute the bandwidth parameter for each guess.
    cf_params = guess[:, 0]
    bw_params = guess[:, 2] * bw_std_edge

    # Unpack the frequency range bounds.
    lower_bound, upper_bound = freq_range

    # Compute conditions: peak is kept if its center is farther than bw_params
    # from both the lower and upper edges.
    keep_peak = (torch.abs(cf_params - lower_bound) > bw_params) & \
                (torch.abs(cf_params - upper_bound) > bw_params)
    

    # if len(guess[keep_peak]) < len(guess):      
    #     print(' FREQUENCY DROPPED DUE TO EDGE, COUNT: ', len(guess) - len(guess[keep_peak]) )
    #     print("GUESS PRIOR TO DROP",guess)
    #     print("FREQ REMOVED GUESS",guess[keep_peak])
    # Filter the guess tensor based on the mask.
    return guess[keep_peak]

def torch_drop_peak_overlap_vectorized(guess, gauss_overlap_thresh):
    """
    Drop overlapping Gaussian peaks based on their overlap, vectorized in tensor space.

    Parameters
    ----------
    guess : torch.Tensor
        A tensor of shape [n_peaks, 3] where each row is [center, height, std].
    gauss_overlap_thresh : float
        The multiplicative factor for the guessed standard deviation to determine overlap.
        For any two adjacent peaks that overlap (i.e. the upper bound of the left peak exceeds
        the lower bound of the right peak), the one with the lower height is dropped.

    Returns
    -------
    torch.Tensor
        The filtered guess tensor (with shape [n_remaining_peaks, 3]).
    """
    # Sort by center frequency.
    sorted_indices = torch.argsort(guess[:, 0])
    guess = guess[sorted_indices]

    # Compute lower and upper bounds for each peak.
    lower_bounds = guess[:, 0] - guess[:, 2] * gauss_overlap_thresh
    upper_bounds = guess[:, 0] + guess[:, 2] * gauss_overlap_thresh

    n_peaks = guess.shape[0]
    if n_peaks < 2:
        return guess
    
    # Compute overlap between adjacent peaks.
    # overlap[i] is True if peak i and peak i+1 overlap.
    overlap = (upper_bounds[:-1] > lower_bounds[1:])

    # For each overlapping pair, select the index corresponding to the lower height.
    indices = torch.arange(n_peaks, device=guess.device)
    candidate_drop = torch.where(
        guess[:-1, 1] < guess[1:, 1],
        indices[:-1],
        indices[1:]
    )
    # Select only the candidates corresponding to overlapping adjacent peaks.
    drop_inds = candidate_drop[overlap]

    # Remove duplicates if a peak is dropped in more than one comparison.
    drop_inds = torch.unique(drop_inds)

    # Build a boolean mask to keep peaks not in drop_inds.
    mask = torch.ones(n_peaks, dtype=torch.bool, device=guess.device)
    mask[drop_inds] = False
    
    # if len(guess[mask])< len(guess):      
        # print(' FREQUENCY DROPPED DUE TO OVERLAP, COUNT: ', len(guess)-len(guess[mask]) )
        # print("GUESS PRIOR TO MASK",guess)
        # print("MASK REMOVED GUESS",guess[mask])

    
    # if len(guess) < len(guess[keep_peak]):      
    #     print(' FREQUENCY DROPPED, COUNT: ', len(guess[keep_peak]) - len(guess))
    return guess[mask]


def guess_peaks(spectrum, freqs, freq_res, peak_threshold, min_peak_height,
                max_n_peaks, gauss_std_limits, bw_std_edge, gauss_overlap_thresh):
    """
    Process a single spectrum or multi-channel spectra using torch_fit_peaks
    and return the guesses as a NumPy array or a list of NumPy arrays.
    
    Parameters
    ----------
    spectrum : np.ndarray
        Either a 1D array (single spectrum of shape (n_freq,)) or a 2D array 
        (multi-channel with shape (n_channels, n_freq)).
    freqs : np.ndarray
        1D array of frequency values.
    freq_res : float
        Frequency resolution (can be computed as freqs[1]-freqs[0]).
    peak_threshold : float
        Multiplier for determining if a candidate peak is high enough.
    min_peak_height : float
        Minimum required peak height.
    max_n_peaks : int
        Maximum number of peaks to find.
    gauss_std_limits : tuple
        Allowed bounds for the guessed standard deviation (e.g., (0.5, 12.0)).
    bw_std_edge : float
        Multiplier used when dropping peaks near the frequency edges.
    gauss_overlap_thresh : float
        Multiplier used when dropping overlapping peaks.
    
    Returns
    -------
    If `spectrum` is 1D: np.ndarray of shape (n_peaks, 3)
    If `spectrum` is 2D: list of np.ndarray (one per channel, each of shape (n_peaks, 3))
    """
    # Check whether the input is single-channel (1D) or multi-channel (2D)
    if spectrum.ndim == 1:
        # Process single channel
        results = []
        spectrum = spectrum.squeeze(0)
        guess_tensor = torch_fit_peaks(
            flat_iter_batch=spectrum,
            freqs=freqs,
            freq_res=freq_res,
            peak_threshold=peak_threshold,
            min_peak_height=min_peak_height,
            max_n_peaks=max_n_peaks,
            gauss_std_limits=gauss_std_limits,
            bw_std_edge=bw_std_edge,
            gauss_overlap_thresh=gauss_overlap_thresh
        )
        guess_np = guess_tensor.clone().detach().cpu().numpy()
        results.append(guess_np)
        return results
    elif spectrum.ndim == 2:
        # Process multiple channels: assume each row is one channel's spectrum.
        results = []
        for chan in spectrum:
            guess_tensor = torch_fit_peaks(
                flat_iter=chan,
                freqs=freqs,
                freq_res=freq_res,
                peak_threshold=peak_threshold,
                min_peak_height=min_peak_height,
                max_n_peaks=max_n_peaks,
                gauss_std_limits=gauss_std_limits,
                bw_std_edge=bw_std_edge,
                gauss_overlap_thresh=gauss_overlap_thresh
            )
            guess_np = guess_tensor.clone().detach().cpu().numpy()
            results.append(guess_np)
        return results
    else:
        raise ValueError("Input spectrum must be 1D (single channel) or 2D (multi-channel).")


def torch_fit_peaks(flat_iter_batch, freqs, freq_res=None, peak_threshold=2.0,
                            min_peak_height=0.5, max_n_peaks=5,
                            bw_std_edge=1.0, gauss_overlap_thresh=0.75,
                            gauss_std_limits=(0.5, 12.0)):
    """
    Batch version of torch_fit_peaks.
    
    Parameters
    ----------
    flat_iter_batch : torch.Tensor
        A 2D tensor of shape (B, N), where B is the batch (trial) size and
        N is the number of frequency bins.
    freqs : torch.Tensor
        1D tensor of frequency values of shape (N,).
    freq_res : float
        Frequency resolution; if None it is computed as freqs[1]-freqs[0].
    peak_threshold, min_peak_height, max_n_peaks, bw_std_edge, gauss_overlap_thresh, 
    gauss_std_limits : various
        Same as in the single-spectrum version.
    
    Returns
    -------
    batch_guess_tensors : list of torch.Tensor
        A list (length B) where each element is a tensor of shape (n_peaks, 3) 
        for that trial (or an empty tensor if no peaks were found).
    """
    with torch.no_grad():
    
        device = flat_iter_batch.device
        B, N = flat_iter_batch.shape
        if freq_res is None:
            freq_res = freqs[1] - freqs[0]
        
        # This will hold the list of guesses for each trial.
        all_guesses = [[] for _ in range(B)]
        
        # Make a working copy so we do not modify the original.
        flat_batch = flat_iter_batch.clone()
        
        # Precompute an indices tensor for the frequency axis, shape (B, N)
        indices = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    
        # Iterate up to max_n_peaks times.
        for _ in range(max_n_peaks):
            # For each trial in the batch, compute the current standard deviation.
            current_std = torch.std(flat_batch, dim=1)  # shape (B,)
            # Find the index of the maximum value for each trial.
            max_indices = torch.argmax(flat_batch, dim=1)  # shape (B,)
            max_heights = flat_batch[torch.arange(B, device=device), max_indices]  # shape (B,)
            
            # Determine validity: the candidate must be above threshold and min height.
            valid = (max_heights > (peak_threshold * current_std)) & (max_heights > min_peak_height)
            
            # If no trial has a valid peak, break out.
            if not valid.any():
                break
            
            # Half height per trial.
            half_heights = 0.5 * max_heights  # shape (B,)
            
            # --- Find left boundary for each trial ---
            # Create a mask: positions before max_indices and where the value is <= half_height.
            left_mask = (indices < max_indices.unsqueeze(1)) & (flat_batch <= half_heights.unsqueeze(1))
            # Replace invalid positions with -1.
            left_candidates = torch.where(left_mask, indices, torch.full_like(indices, -1))
            # The best candidate is the maximum index (i.e. the one closest to the peak).
            left_ind = left_candidates.max(dim=1).values  # shape (B,)
            
            # --- Find right boundary for each trial ---
            right_mask = (indices > max_indices.unsqueeze(1)) & (flat_batch <= half_heights.unsqueeze(1))
            # Replace invalid positions with N.
            right_candidates = torch.where(right_mask, indices, torch.full_like(indices, N))
            right_ind = right_candidates.min(dim=1).values  # shape (B,)
            
            # Convert integer tensors to floats.
            max_indices_float = max_indices.to(torch.float)
            left_ind_float = left_ind.to(torch.float)
            right_ind_float = right_ind.to(torch.float)
        
            distance_left = torch.where(
            left_ind == -1,
            torch.full_like(left_ind_float, float('inf')),
            max_indices_float - left_ind_float
            )
            distance_right = torch.where(
                right_ind == N,
                torch.full_like(right_ind_float, float('inf')),
                right_ind_float - max_indices_float
            )
            # Compute distances; if no candidate is found, set distance to infinity.
            # distance_left = torch.where(left_ind == -1, torch.full_like(left_ind, float('inf')), max_indices - left_ind)
            # distance_right = torch.where(right_ind == N, torch.full_like(right_ind, float('inf')), right_ind - max_indices)
            
            
            short_side = torch.min(distance_left, distance_right).to(torch.float)  # shape (B,)
            
            # Estimate FWHM and then the guess standard deviation.
            fwhm = short_side * 2 * freq_res  # shape (B,)
            # Assume torch_compute_gauss_std is vectorized; if not, wrap it.
            guess_std = torch_compute_gauss_std(fwhm.to(torch.double))
            guess_std = torch.clamp(guess_std, min=gauss_std_limits[0], max=gauss_std_limits[1])
            
            # Guess frequency and height.
            guess_freq = freqs[max_indices]  # shape (B,)
            guess_height = max_heights
            
            # Record the guess for each trial where the candidate is valid.
            for j in range(B):
                if valid[j]:
                    all_guesses[j].append([guess_freq[j].item(), guess_height[j].item(), guess_std[j].item()])
            
            # Compute the Gaussian to subtract.
            # For each trial, compute: G(x) = guess_height * exp( -((x - guess_freq)**2)/(2*guess_std**2) )
            diff = freqs.unsqueeze(0) - guess_freq.unsqueeze(1)  # shape (B, N)
            gaussian = guess_height.unsqueeze(1) * torch.exp(- (diff ** 2) / (2 * guess_std.unsqueeze(1) ** 2))
            # Only subtract for valid trials.
            mask = valid.unsqueeze(1).to(torch.float)
            
            flat_batch = (flat_batch - gaussian * mask).detach() #avoid stack overflow
    
        # Now, for each trial, convert the list of guesses to a tensor and apply drop functions.
        batch_guess_tensors = []
        for guesses in all_guesses:
            if len(guesses) > 0:
                guess_tensor = torch.tensor(guesses, dtype=flat_iter_batch.dtype, device=device)
                sorted_indices = torch.argsort(guess_tensor[:, 0])
                guess_tensor = guess_tensor[sorted_indices]
            else:
                guess_tensor = torch.empty((0, 3), dtype=flat_iter_batch.dtype, device=device)
            # Apply the drop functions.
            freq_range = (freqs[0].item(), freqs[-1].item())
            guess_tensor = torch_drop_peak_cf(guess_tensor, freq_range, bw_std_edge)
            guess_tensor = torch_drop_peak_overlap_vectorized(guess_tensor, gauss_overlap_thresh)
            batch_guess_tensors.append(guess_tensor)
            
        if B == 1:
            return batch_guess_tensors[0]
    
        return batch_guess_tensors
