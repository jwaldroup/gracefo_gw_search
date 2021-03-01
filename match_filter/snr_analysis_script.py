# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:34:24 2021

@author: John Waldroup

GRACE-FO SNR Analysis Script
"""

"""
Script Overview

1. read in gracefo model noise data from csv
2. initialize an array of masses to test and an array of distances
3. for loop over the array of distances that for each mass:
3.1 computes waveform and inject copy into noise 
3.2 take psd of waveform and noise data
3.3 calculates the snr estimate for that selected mass at that distance iteration
3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
4. for loop through mass vs distance array and returns a new array of mass vs distance
the snr is closest to the chosen value of ten
5. plot

"""
#imports
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt

from pycbc import types
from pycbc import psd

import pycbc_welch_function as welch_function
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder

#1 - read in gracefo model noise data from csv----------------------------------------------------------------------------------------

#replace with csv reading later

#Import data from csv
fileobj = open('GRACE-Copy.csv', 'r')
readerobj = csv.reader(fileobj)

data = []
for row in readerobj:
    data.append(row)
    
fileobj.close()

#Extract frequency and Strain data
data_array = np.array(data, dtype='d')
grace_freqs = data_array[0:,0]
grace_signal = data_array[0:,1]
grace_asd = grace_signal / 220.0e3 #converts between m/sqr(Hz) and 1/sqr(Hz)

N = 2000000 #long enough to encompass waveforms with a f_low of 0.1 and minimal mass of 100 sol masses each 
cutoff = 0.001 
order = 950 
beta = 17  
seg_num = 15
        
#noise signal
np.random.seed(138374923)
noise1 = np.random.uniform(-1, 1, size=N)
            
#convert to TimeSeries pycbc object
noise1_ts = types.timeseries.TimeSeries(noise1, delta_t=0.1) #delta_t = 0.1 to match gracefo sample frequency of 10 Hz
        
#adjust amplitude
noise1_ts = noise1_ts * 10e-8
        
#filter it
filtered1 = noise1_ts.lowpass_fir(cutoff, order, beta=beta)
        
#psd.welch to create psd
noise1_psd = welch_function.pycbc_welch(filtered1, seg_num)
        
#conversion to asd
noise1_asd = np.sqrt(noise1_psd)
        
#plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='curve 1 - "hump"' )
#plt.loglog(grace_freqs, grace_asd, label='gracefo asd')
#plt.grid()
#plt.legend()
#plt.show()

#some parameters
df = noise1_asd.delta_f
dt = filtered1.delta_t 
T = dt * N
f_s = 1.0 / dt
f_nyq = f_s / 2.0
#print('Noise Curve 1 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)


N = 2000000
cutoff = 0.00001
order = 20000
beta = 15.0
seg_num = 15

#noise signal
np.random.seed(138374923)
noise2 = np.random.uniform(-1, 1, size=N)
    
#convert to TimeSeries pycbc object
noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)

#adjust amplitude
noise2_ts = noise2_ts * 10e-6

#filter it
filtered2 = noise2_ts.lowpass_fir(cutoff, order, beta=beta)

#psd.welch to create psd
noise2_psd = welch_function.pycbc_welch(filtered2, seg_num)

#conversion to asd
noise2_asd = np.sqrt(noise2_psd)

#some parameters
df = noise2_asd.delta_f
dt = filtered2.delta_t
T = dt * N
f_s = 1.0 / dt
f_nyq = f_s / 2.0

##print('Noise Curve 2 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)

#uncomment to check dimensions of timeseries
#print(np.size(filtered1), np.size(filtered2))

#pad the smaller one to equivalent lengths
filtered2c = filtered2.copy()
filtered2c.append_zeros((np.size(filtered1)-np.size(filtered2)))

#uncomment to check that the sizes match
#print(np.size(filtered1), np.size(filtered2c))

#Add the two 
merged_noise = np.array(filtered1) + np.array(filtered2c)
merged_noise_ts = types.timeseries.TimeSeries(merged_noise, delta_t=0.1) #ensures same delta_t


#Compare merged noise curves with gracefo data
seg_num = 15
noise_psd = welch_function.pycbc_welch(merged_noise_ts, seg_num)
plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')
#plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
#plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')
plt.loglog(grace_freqs, grace_asd, label='gracefo asd')

#compare merged noise curve with psd computed via fft with gracefo data
noise_freqs = np.fft.fftfreq(np.size(merged_noise_ts.copy()))
noise_mask = noise_freqs > 0
raw_fft_noise = np.fft.fft(merged_noise_ts.copy())
psd_of_noise = 2.0 * ( np.abs( raw_fft_noise) / float(np.size(merged_noise_ts.copy()) ) )** 2.0

plt.loglog(noise_freqs[noise_mask], np.sqrt(psd_of_noise[noise_mask]), label='noise model fft asd')

plt.legend()
plt.xlabel('frequency (Hz)')
plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
plt.grid()
plt.show()


# # 2 - initialize an array of masses to test and an array of distances----------------------------------------------------------------

# mass_array = np.logspace(500, 800) #in sol masses
# distances_array = np.logspace(1000, 1500) #in pc

# #print('mass array size:', np.size(mass_array), 'distance array size:', np.size(distances_array))
# #print(mass_array[0:10])
# #print(distances_array[0:10])

# # 3 - for loop over the array of distances that for each mass:-----------------------------------------------------------------------
#     #3.1 computes waveform 
#     #3.2 take psd of waveform 
#     #3.3 calculates the snr estimate for that selected mass at that distance iteration
#     #3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
    
# snr_values_row_mass_column_distance = np.zeros((np.size(mass_array), np.size(distances_array)), dtype=merged_noise_ts.dtype)

# for i in range(np.size(mass_array)):
    
#     #print('mass:'+str(mass_array[i]))
    
#     #set mass and other waveform parameters 
#     m1 = mass_array[i]
#     m2 = mass_array[i]
#     f_low = 0.1
#     dt = 0.1
#     theta = 0.0
    
#     for j in range(np.size(distances_array)):
        
#         #print('distance:'+str(distances_array[j]))
        
#         #set distance parameter
#         r = distances_array[j]
        
#         #3.1 computes waveform
#         t_array, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)
        
# #        #edit waveform by truncation at first and last zero
# ##        waveform_max = np.max(hp)
# ##        last_tolerance = waveform_max * 1e-2
# ##        first_tolerance = waveform_max * 1e-4
# ##        
# #        #hp = zero_finder.last_zero_finder(hp) #, abs_tol=last_tolerance)
# #       # hp = zero_finder.first_zero_finder(hp) #, abs_tol=first_tolerance)
# #        
# #        #convert to pycbc compatible timeseries
# #        hp_ts = types.timeseries.TimeSeries(hp, merged_noise_ts.delta_t) #ensures same delta_t
# #        
# #        #3.2 take psd of waveform
# #        hp_psd_fs = welch_function.pycbc_welch(hp_ts, 15)
# #
# #        #3.3 calculates the snr estimate for that selected mass at that distance iteration
# ##        print("vector sizes:" , np.size(hp_psd_fs), np.size(noise_psd))
# ##        print('df h_fs:', hp_psd_fs.delta_f, 'df noise:', noise_psd.delta_f) #to check the df of each
# #
# #        hp_psd_fs = psd.interpolate(hp_psd_fs, noise_psd.delta_f) #interpolate the larger df of the two to match
# ##        print("vector sizes:" , np.size(hp_psd_fs), np.size(noise_psd))
# #
# #        print(noise_psd.sample_frequencies[-10:])
# #        print(hp_psd_fs.sample_frequencies[-10:])
# #        print(np.size(noise_psd.sample_frequencies), np.size(hp_psd_fs.sample_frequencies))
        
        
# #        psd_ratio_pycbc_welch = (4.0 * hp_psd_fs  ) / (noise_psd[0:np.size(hp_psd_fs)]) 
# #        snr_squared_pycbc_welch = psd_ratio_pycbc_welch.sum()
# #        snr_estimate_pycbc_welch = np.sqrt(snr_squared_pycbc_welch)
        
#         #repeating all of the above with fft since pycbc is a butthole
#         #hp = zero_finder.last_zero_finder(hp.copy(), 1e-14)
#         #h_ts = zero_finder.first_zero_finder(hp, 1e-15)
#         h_ts = hp.copy()
        
#         signal_freqs = np.fft.fftfreq(np.size(h_ts))
#         signal_mask = signal_freqs > 0
#         raw_fft_h_ts = np.fft.fft(h_ts)
#         psd_of_h_ts = 2.0 * ( np.abs( raw_fft_h_ts / float( np.size(h_ts) ) ) )** 2.0

#         #print("vector sizes:" , np.size(psd_of_h_ts), np.size(psd_of_noise))
#         fft_psd = np.interp(noise_freqs, signal_freqs, psd_of_h_ts)
#         #print("vector sizes:" , np.size(fft_psd), np.size(psd_of_noise))

#         #calculate snr
#         psd_ratio_fft = (4.0 * fft_psd[noise_mask]) / (psd_of_noise[noise_mask])
#         snr_squared_fft = psd_ratio_fft.sum()
#         snr_estimate_fft = np.sqrt(snr_squared_fft)
        
        
#         #3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
#         snr_values_row_mass_column_distance[i,j] = snr_estimate_fft #snr_estimate_pycbc_welch

# # 4 - for loop through mass vs distance array and return a new array of -------------------------------------------------------------
#         #mass vs distance the snr is closest to the chosen value of ten
#     #4.1 - initialize snr_threshold value
#     #4.2 - loop through the rows of the snr_values_row_mass_column_distance array
#     #4.2.1 - for each row get the column index of snr closest to threshold value of 10
#     #4.3 - with the column index, get the distance value for that snr
#     #4.4 - package mass, and distance into two respective vectors for plotting
    
# print(snr_values_row_mass_column_distance)
# print(snr_values_row_mass_column_distance[0,:])

# #4.1 - initialize SNR threshold value
# snr_threshold = 8.0

# distance_with_snr_threshold = np.zeros(np.size(mass_array))
# #distance_with_snr_threshold = []
# #4.2 - loop through the rows of the snr_values_row_mass_column_distance array
# for i in range(np.size(mass_array)):
    
#     #4.2.1 - for each row get the column index of snr closest to threshold value of 8
#     index_of_min = np.argmin(np.abs(snr_values_row_mass_column_distance[i,:] - snr_threshold))
    
#     #4.3/.4 - with the column index, get the distance value for that snr
#     distance_with_snr_threshold[i] = distances_array[index_of_min]
#     #distance_with_snr_threshold.append(distances_array[index_of_min])
    
    
# print(distance_with_snr_threshold)
# print(np.size(distance_with_snr_threshold))      
  
# # 5 - plottinng the snr threshold distance for each mass

# plt.loglog(mass_array, distance_with_snr_threshold, label=('snr threshold: '+str(snr_threshold)))
# plt.xlabel('Mass (sol)')
# plt.ylabel('Distance (pc)')
# plt.grid()
# plt.legend()
# plt.title('Mass vs Distance Where SNR of 8.0 is Reached')
# plt.show()


    
    
    
    