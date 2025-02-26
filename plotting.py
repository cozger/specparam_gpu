# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:56:23 2025

@author: canoz
"""


import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

#Plot slicing

plot_start = 100  # adjust this slice as needed
plot_end = 900
plot_chan = 1

#%% Original Spectra vs. Robust A periodic
plt.figure(figsize=(8, 4))


plt.plot(Input_Data_cpu[plot_chan,:,plot_start:plot_end], 'b-', alpha=0.3, label='Original Spectra (log₁₀)')
plt.plot(all_robust_fitted[plot_chan,:,plot_start:plot_end], 'g-', alpha=0.1, label='Robust Fit Spectra (log₁₀)')

plt.xlabel('Frequency Index')
plt.ylabel('Log₁₀(Power)')
plt.show()

#%%

plt.figure(figsize=(8, 4))

plt.plot(Input_Data_cpu[plot_chan,:,plot_start:plot_end],'b-', alpha=0.1, label='Empirical Data')
plt.plot(full_modeled_spectra[plot_chan,:,plot_start:plot_end],'g-', alpha=0.1, label='Full Model Spectra')
plt
plt.show()

#%%

plt.figure(figsize=(8, 4))

plt.plot(full_modeled_spectra[1,:,700]+8,'b-', alpha=0.5, label='Empirical Data')


plt.plot(freqs.cpu(), gaussian_curves[700][1].squeeze(),'g-', alpha=0.5, label='Full Model Spectra')
plt
plt.show()
#%%


plt.figure(figsize=(8, 4))

plt.plot(all_flattened[plot_chan,:,plot_start:plot_end],'r-', alpha=0.3, label='Flattened Spectra')
# plt.plot(robust_flattened[0,:,0:rng2plot],'g-', alpha=0.1, label='Robust FLattened Spectra')
plt.show()
#%%
import json
import numpy as np
from scipy import stats
import statistics as sts
sts.mean(performance_native['overall_time'])/sts.mean(performance_GPU['overall_time'])
t_statistic, p_value = stats.ttest_ind(performance_native['overall_time'], performance_GPU['overall_time'])   


import json


# Save the variable to a file
filename = 'Performance_GPU.json'
with open(filename, 'w') as file:
    json.dump(performance_GPU, file, indent=4) # Indent for readability
   
    
# Save the variable to a file
filename = 'Performance_Native.json'
with open(filename, 'w') as file:
    json.dump(performance_native, file, indent=4) # Indent for readability
    
       
    

# # Load the variable from the file
# with open(filename, 'r') as file:
#     loaded_data = json.load(file)

# print(loaded_data)
#%%
plt.figure(figsize=(8, 4))

flattened_curves = []
for channel in all_gaussian_curves:
    for arr in channel:
        # Remove extra dimensions. For example, an array with shape (1,1,40)
        # will be squeezed to shape (40,).
        arr_squeezed = np.squeeze(arr)
        
        # Ensure we have the frequency axis of length 40.
        # If the result is 1D, we assume it represents a single spectrum.
        if arr_squeezed.ndim == 1:
            if arr_squeezed.shape[0] != 40:
                print(f"Warning: Expected 40 frequency bins but got {arr_squeezed.shape[0]} in array: {arr.shape}")
            # Convert a 1D vector to a 2D array with one row.
            spectrum = arr_squeezed.reshape(1, -1)
            flattened_curves.append(spectrum)
        elif arr_squeezed.ndim == 2:
            # In a 2D array we assume that each row is a spectrum.
            if arr_squeezed.shape[1] != 40:
                print(f"Warning: Expected 40 frequency bins but got {arr_squeezed.shape[1]} in array: {arr.shape}")
            # Append each row (spectrum) separately.
            for row in arr_squeezed:
                # Ensure that each row is 1D and then reshape to (1,40)
                row = np.atleast_1d(row)
                if row.shape[0] != 40:
                    print(f"Warning: Expected 40 frequency bins but got {row.shape[0]} in row from array: {arr.shape}")
                flattened_curves.append(row.reshape(1, -1))
        else:
            print(f"Unexpected array shape after squeezing: {arr_squeezed.shape} (original: {arr.shape})")

# Now, stack all the individual spectra vertically.
if flattened_curves:
    concatenated_array = np.vstack(flattened_curves)
else:
    concatenated_array = np.empty((0, 40))

# Remove any rows that are entirely NaN.
concatenated_array = concatenated_array[~np.all(np.isnan(concatenated_array), axis=1)]


plt.plot(all_flattened[:, plot_start:plot_end].T, 'g-', alpha=0.3, label='Robust Flattened Spectra')
plt.plot(concatenated_array.T[plot_start:plot_end], 'r-', alpha=0.1, label='Fitted Periodic Peaks')
plt.xlabel('Frequency Index')
plt.ylabel('Log₁₀(Power)')
plt.legend()
plt.show()

