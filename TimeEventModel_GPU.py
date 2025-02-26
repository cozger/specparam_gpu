# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:35:47 2025

@author: canoz
"""

import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import matplotlib.pyplot as plt
import time  
from specparam.gpu.torch_curve_fit import gpu_curve_fit
from specparam.gpu.torch_lm_fit import gpu_curve_fit_lm_vectorized_aperiodic
from specparam.gpu.torch_periodic import torch_fit_peaks
from specparam.gpu.torch_periodic import guess_peaks
from specparam.gpu.gaussian_lm_current import gpu_vectorized_gaussian_two
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from tqdm_joblib import ParallelPbar

os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"


# -------------------------------
# Parameters & Data Preparation
# -------------------------------
freqs = np.arange(1, 41)  # 1D frequency vector
ap_percentile_thresh = 25
peak_threshold     = 2.0
min_peak_height    = 0.5
max_n_peaks        = 5
gauss_std_limits   = (0.5, 12.0)
bw_std_edge        = 1.0
gauss_overlap_thresh = 0.75

# Load and prepare full spectra (assumed shape: [n_channels, n_freq, n_trials])
fullspectra = np.load(r"C:\Users\canoz\OneDrive\Masaüstü\FOOFpractice\fullspecogram.npy")
# For example, if you want only a subset (here “onechan” is used)
fullspectra = fullspectra[0:2, freqs, :]

# Convert to log space on the CPU
Input_Data_cpu = np.log10(fullspectra + 1e-20)
n_channels, n_freq, n_trials = Input_Data_cpu.shape

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

freqs = torch.from_numpy(freqs).double().to(device)

# Preallocate final outputs (on CPU)
all_aparams       = np.empty((n_channels, 2, n_trials))
all_fitted        = np.empty((n_channels, n_freq, n_trials))
all_robust_aparams= np.empty((n_channels, 2, n_trials))
all_robust_fitted = np.empty((n_channels, n_freq, n_trials))
all_flattened = np.empty((n_channels,n_freq, n_trials))
full_modeled_spectra = np.empty((n_channels,n_freq, n_trials))
all_gaussian_params= []
all_gaussian_curves= []
gaussian_params   = []
gaussian_curves   = []

# Set batch sizes
batch_size_aperiodic  = 45
batch_size_robust     = 150
batch_size_guess_peaks= 800
gaussian_batch_size   = 20




performance_GPU = dict(overall_time=[], spectra_per_sec=[])


def process_minibatch(minibatch, freqs, freq_res, peak_threshold, min_peak_height,
                      max_n_peaks, gauss_std_limits, bw_std_edge, gauss_overlap_thresh):
    # Ensure minibatch is 2D
    if minibatch.ndim == 1:
        minibatch = minibatch.unsqueeze(0)
    # Call the batched peak-fitting function (torch_fit_peaks)
    guesses_batch = torch_fit_peaks(
        minibatch, 
        freqs,  # a 1D torch.Tensor of frequency values
        freq_res=1.0,
        peak_threshold=peak_threshold,
        min_peak_height=min_peak_height,
        max_n_peaks=max_n_peaks,
        gauss_std_limits=gauss_std_limits,
        bw_std_edge=bw_std_edge,
        gauss_overlap_thresh=gauss_overlap_thresh
    )
    return guesses_batch

for i in range(0,10):
    start_time_overall = time.perf_counter()
    
    if __name__ == '__main__':
        for chan_index in range(n_channels):
            print(f"\nProcessing channel {chan_index} ...")
            
            
            #Get the channel data to GPU
            chan = torch.from_numpy(Input_Data_cpu[chan_index]).to(device).double()
            
            # ---- Aperiodic Initial Fit (per channel) ----
            t0 = time.perf_counter()
    
            aperiodic_fits, fitted_curves = gpu_curve_fit(
                freqs, chan, guess=None, batch_size=batch_size_aperiodic, func='aperiodic'
            )
            t1 = time.perf_counter()
            print(f"Aperiodic Fit took: {t1 - t0:.4f} seconds")
            
            # Store the results
            all_aparams[chan_index] = aperiodic_fits.detach().cpu().T.numpy()
            all_fitted[chan_index]  = fitted_curves.detach().cpu().numpy()
            
            # ---- Robust Aperiodic Fit ----
            
            t0 = time.perf_counter()
            robust_flattened = chan - fitted_curves
            robust_flattened = torch.clamp(robust_flattened, min=0)
            perc_thresh = torch.quantile(robust_flattened.flatten(), ap_percentile_thresh/100.0)
            perc_mask   = robust_flattened <= perc_thresh
            spectrum_ignore = torch.where(perc_mask, chan, torch.tensor(float('nan'), device=device))
            del perc_thresh, perc_mask, chan 
            
            all_flattened[chan_index,:,:] = robust_flattened.detach().cpu().numpy()
            
            def process_aperiodics_in_batches(freqs, spectra, batch_size, func, guess):
                n_trials = spectra.shape[1]
                batch_fitted_params = []
                batch_fitted_curves = []
                n_batches = (n_trials + batch_size - 1) // batch_size
                for start in tqdm(range(0, n_trials, batch_size),
                                  total=n_batches,
                                  desc="Robust Fitting Aperiodics in Batches",
                                  leave=False):
                    end = min(start + batch_size, n_trials)
                    spectra_batch = spectra[:, start:end]
                    guess_batch = guess[start:end] if guess is not None else None
                    fitted_params, fitted_curves = gpu_curve_fit_lm_vectorized_aperiodic(
                        freqs, spectra_batch, batch_size, func, guess=guess_batch
                    )
                    batch_fitted_params.append(fitted_params)
                    batch_fitted_curves.append(fitted_curves)
                    
                chan_fitted_params = torch.concatenate(batch_fitted_params, axis=0)
                chan_fitted_curves = torch.concatenate(batch_fitted_curves, axis=0)
                return chan_fitted_params, chan_fitted_curves
            
            t_mid = time.perf_counter()
            robust_params, robust_curves = process_aperiodics_in_batches(
                freqs, spectrum_ignore, batch_size=batch_size_robust, func='aperiodic', guess=aperiodic_fits
            )
            t1 = time.perf_counter()
            print(f"Robust Aperiodic Fit took: {t1 - t0:.4f} seconds (batches processing: {t1 - t_mid:.4f} seconds)")
            
            all_robust_aparams[chan_index] = robust_params.T.detach().cpu().numpy()
            all_robust_fitted[chan_index]  = robust_curves.T.detach().cpu().numpy()
            del aperiodic_fits, spectrum_ignore, robust_params, robust_curves  #GPU Memory housekeeping
            
            
            # ---- Periodic Peak Guessing ----
            t0 = time.perf_counter()
            n_trials, n_freq = robust_flattened.T.shape
            mini_batches = []
            for i in range(0, n_trials, batch_size_guess_peaks):
                minibatch = robust_flattened.T[i: i + batch_size_guess_peaks]
                mini_batches.append(minibatch)
                
            with tqdm_joblib(tqdm(desc="Guessing Periodic Peak Parameters in Batches", total=len(mini_batches))):
                results = Parallel(n_jobs=5, backend='loky')(
                    delayed(process_minibatch)(
                        minibatch, freqs, 1.0, peak_threshold, min_peak_height,
                        max_n_peaks, gauss_std_limits, bw_std_edge, gauss_overlap_thresh
                    )
                    for minibatch in mini_batches
                )
            all_peak_guesses = []
            
            for guesses_batch in results:
                all_peak_guesses.extend(guesses_batch)
            t1 = time.perf_counter()
            print(f"Periodic Peak Guessing took: {t1 - t0:.4f} seconds")
            
            # ---- Gaussian LM Fitting for Periodic Peaks ----
            t0 = time.perf_counter()
            def process_in_batches_gaussian(freqs, spectra, batch_size, guess):
                def flatten(xss):
                    return [x for xs in xss for x in xs]
                n_trials = spectra.shape[1]
                batch_fitted_params = []
                batch_fitted_curves = []
                n_batches = (n_trials + batch_size - 1) // batch_size
                for start in tqdm(range(0, n_trials, batch_size),
                                  total=n_batches,
                                  desc="Fitting Gaussians in Batches", leave=False):
                    end = min(start + batch_size, n_trials)
                    spectra_batch = spectra[:, start:end]
                    guess_batch = guess[start:end]
                    fitted_params, fitted_curves = gpu_vectorized_gaussian_two(
                        freqs, spectra_batch, batch_size, func='gaussian', guess=guess_batch
                    )
                    batch_fitted_params.append(fitted_params)
                    batch_fitted_curves.append(fitted_curves)
                chan_fitted_params = flatten(batch_fitted_params)
                chan_fitted_curves = flatten(batch_fitted_curves)
                return chan_fitted_params, chan_fitted_curves
            
            t_mid = time.perf_counter()
            gaussian_params, gaussian_curves = process_in_batches_gaussian(
                freqs, robust_flattened, batch_size=gaussian_batch_size, guess=all_peak_guesses
            )
            t1 = time.perf_counter()
            print(f"Gaussian LM Fitting took: {t1 - t0:.4f} seconds (batches processing: {t1 - t_mid:.4f} seconds)")
            
            gaussian_params = [
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in gaussian_params
            ]
            
            gaussian_curves = [
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in gaussian_curves
            ]
    
            all_gaussian_params.append(gaussian_params)
            all_gaussian_curves.append(gaussian_curves)
            del robust_flattened, fitted_curves, minibatch, all_peak_guesses, gaussian_params
            
            # ---- Full Model Computation (Aperiodic + Gaussian Peaks) ----
            t0 = time.perf_counter()
            full_modeled_channel = torch.empty((n_freq, n_trials))
            all_robust_fitted_chan_torch= torch.from_numpy(all_robust_fitted[chan_index])
            
            for trial in range(n_trials):
                aperiodic_fit = all_robust_fitted_chan_torch[:, trial]
                peaks_item = gaussian_curves[trial] if gaussian_curves[trial] is not None else None
                peak_sum = torch.zeros(n_freq)
                if peaks_item is not None:
                    if isinstance(peaks_item, np.ndarray):
                        peaks_item = torch.from_numpy(peaks_item)
                    peaks_array = torch.atleast_2d(peaks_item)
                    valid_peaks = [row for row in peaks_array if not torch.all(torch.isnan(row))]
                    if valid_peaks:
                        peak_sum = torch.sum(torch.stack(valid_peaks), dim=0)
                full_modeled_channel[:, trial] = aperiodic_fit + peak_sum
            full_modeled_spectra[chan_index, :, :] = full_modeled_channel.detach().cpu().numpy()
            t1 = time.perf_counter()
            print(f"Full Model Computation took: {t1 - t0:.4f} seconds")
            
            
            # ---- Metrics Computation ----
            t0 = time.perf_counter()
            def calc_r_squared(power_spectrum, modeled_spectrum):
                if not torch.is_tensor(power_spectrum):
                    ps = torch.tensor(power_spectrum, dtype=torch.float32).flatten()
                else:
                    ps = power_spectrum.flatten()
                if not torch.is_tensor(modeled_spectrum):
                    ms = torch.tensor(modeled_spectrum, dtype=torch.float32).flatten()
                else:
                    ms = modeled_spectrum.flatten()
                ps = ps.cpu()
                ms = ms.cpu()
                X = torch.stack([ps, ms])
                r_val = torch.corrcoef(X)
                return (r_val[0, 1] ** 2).item()
            
            def calc_error(power_spectrum, modeled_spectrum, metric="MAE"):
                if metric == 'MAE':
                    return torch.abs(power_spectrum - modeled_spectrum).mean()
                elif metric == 'MSE':
                    return ((power_spectrum - modeled_spectrum) ** 2).mean()
                elif metric == 'RMSE':
                    return np.sqrt(((power_spectrum - modeled_spectrum) ** 2).mean())
                else:
                    raise ValueError(f"Error metric '{metric}' not understood or not implemented.")
            
            orig_channel = Input_Data_cpu[chan_index]
            full_modeled_channel_np = full_modeled_channel.detach().cpu().numpy()
            del full_modeled_channel, gaussian_curves
    
            r_squared_channel = np.empty(n_trials)
            error_channel = np.empty(n_trials)
            for trial in range(n_trials):
                orig_spec = orig_channel[:, trial]
                modeled_spec = full_modeled_channel_np[:, trial]
                r_squared_channel[trial] = calc_r_squared(orig_spec, modeled_spec)
                error_channel[trial] = calc_error(orig_spec, modeled_spec, metric="RMSE")
            t1 = time.perf_counter()
            print(f"Metrics Computation took: {t1 - t0:.4f} seconds")
            
            # ---- Cleanup ----
            del results, mini_batches, guesses_batch
            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            gc.collect()
            t1 = time.perf_counter()
            print(f"Cleanup took: {t1 - t0:.4f} seconds")
       
           
    end_time_overall = time.perf_counter()
    elapsed_time = end_time_overall - start_time_overall
    print(f"\nTotal elapsed time: {elapsed_time:.4f} seconds")
    points_per_second = (n_channels * n_trials) / elapsed_time
    print(f"Spectra fit per second: {points_per_second:.4f} Spectra")
    
    performance_GPU['overall_time'].append(elapsed_time)
    performance_GPU['spectra_per_sec'].append(points_per_second)

