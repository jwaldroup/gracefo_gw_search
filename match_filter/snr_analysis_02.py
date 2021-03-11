# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:42:14 2021

@author: john
"""

"""
Script Overview

1. read in or create gracefo model noise data
2. initialize an array of masses to test and an array of distances
3. for loop over the array of distances that for each mass:
3.1 computes waveform 
3.2 take psd of waveform and noise data
3.3 calculates the snr estimate for that selected mass at that distance iteration
3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
4. for loop through mass vs distance array and returns a new array of mass vs distance
the snr is closest to the chosen value of ten
5. plot

"""

import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt

from pycbc.filter import highpass, matched_filter
from pycbc import types
from pycbc import psd
from scipy import signal

import pycbc_welch_function as welch_function
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder


#1 - read in gracefo model noise data from csv----------------------------------------------------------------------------------------

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



#1.1 Build Noise Curve Model

#Curve 1 parameters
N = 2000000 #Possibly increase in length to incorporate lower component mass
cutoff = 0.001 
order = 950 
beta = 17  
seg_size = 150000
        
#noise signal
np.random.seed(138374923)
noise1 = np.random.uniform(-1, 1, size=N)
            
#convert to TimeSeries pycbc object
noise1_ts = types.timeseries.TimeSeries(noise1, delta_t=0.1) #delta_t = 0.1 to match gracefo sample frequency of 10 Hz
        
#adjust amplitude
noise1_ts = noise1_ts * 10e-8
        
#filter it
filtered1 = noise1_ts.lowpass_fir(cutoff, order, beta=beta)


#Curve 2 parameters    
cutoff = 0.00001
order = 14000
beta = 11.0

#noise signal
np.random.seed(138374923)
noise2 = np.random.uniform(-1, 1, size=N)
    
#convert to TimeSeries pycbc object
noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)

#adjust amplitude
noise2_ts = noise2_ts * 10e-8

#filter it
filtered2 = noise2_ts.lowpass_fir(cutoff, order, beta=beta)


# 1.2 - Merge noise curves

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

# #psd.welch to create psd
# noise1_psd = welch_function.pyc_welch(filtered1, seg_size)
# noise1_asd = np.sqrt(noise1_psd)

# noise2_psd = welch_function.pyc_welch(filtered2c, seg_size)
# noise2_asd = np.sqrt(noise2_psd)

# noise_psd = welch_function.pyc_welch(merged_noise_ts, seg_size)

# #Compare merged noise curves with gracefo data
# plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
# plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')

# plt.loglog(grace_freqs, grace_asd, label='gracefo asd')
# plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')

# plt.legend()
# plt.xlabel('frequency (Hz)')
# plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
# plt.grid()
# plt.show()



# 2 - initialize an array of masses to test and an array of distances----------------------------------------------------------------

#m_array = np.logspace(500, 1000) #in sol masses
#d_array = np.logspace(1000, 2000) #in pc
m_array = np.arange(500, 10000, 200)
d_array = np.arange(100, 5000, 100)
m_array = np.array([500])
d_array = np.array([2000, 3000, 4000, 5000, 6000])




# 3 - for loop over the array of distances that for each mass:-----------------------------------------------------------------------
    #3.1 computes waveform 
    #3.2 take psd of waveform and of noise psd
    #3.3 calculates the snr estimate for that selected mass at that distance iteration
    #3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
    
snr_row_m_column_d = np.zeros((np.size(m_array), np.size(d_array)), 
                              dtype=merged_noise_ts.dtype)

for i in range(np.size(m_array)):
    
    print('mass: '+str(m_array[i])+' sol')
    
    #set mass and other waveform parameters 
    m1 = m_array[i]
    m2 = m1 
    f_low = 0.05 #cutoff frequency at which waveform begins
    dt = 0.01 #sampling step size
    theta = 0.0
    
    for j in range(np.size(d_array)):
        
        #3.1 computes waveform
        print('distance: '+str(d_array[j])+' pc')
        
        #set distance parameter
        r = d_array[j]
        
        t_array, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)
                
        #downsample the waveform to gracef's sampling frequency (ie dt=0.1)
        T = dt * np.size(hp)
        dt = 0.1
        resample_x_values = np.arange(0, T, 0.1)
        hp = np.interp(resample_x_values, t_array, hp.copy())
        
        #prep waveform with truncation
        hp_cut = zero_finder.first_zero_finder_02(hp, m1, f_low, dt)
        obs_t_cut = resample_x_values[-np.size(hp_cut):]
        
        hp_cut = zero_finder.last_zero_finder_03(hp_cut, m1, dt)
        obs_t_cut = obs_t_cut[0:np.size(hp_cut)]
        
        
        #3.2 compute psd of noise data and psd of waveform
        hp_ts = types.timeseries.TimeSeries(hp_cut, delta_t=dt)
        hp_psd = welch_function.pyc_welch(hp_ts, np.size(hp_cut))
        noise_psd = welch_function.pyc_welch(merged_noise_ts.copy(), np.size(hp_cut))
        
        print(np.size(hp_psd), np.size(noise_psd))
        #3.3 compute snr estimate
        psd_ratio = (4.0 * hp_psd) / (noise_psd)
        snr_squared = psd_ratio.sum()
        snr_estimate = np.sqrt(snr_squared)
        
        #3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
        snr_row_m_column_d[i,j] = snr_estimate
        
        
        
# 4 - for loop through mass vs distance array and return a new array of -------------------------------------------------------------
        #mass vs distance the snr is closest to the chosen value of ten
    #4.1 - initialize snr_threshold value
    #4.2 - loop through the rows of the snr_row_m_column_d array
    #4.2.1 - for each row get the column index of snr closest to threshold value of 10
    #4.3 - with the column index, get the distance value for that snr
    #4.4 - package mass, and distance into two respective vectors for plotting

#4.1 - initialize SNR threshold value
snr_threshold = 8.0
distance_with_snr_threshold = np.zeros(np.size(m_array))



#4.2 - loop through the rows of the snr_values_row_mass_column_distance array
for i in range(np.size(m_array)):
    
    #4.2.1 - for each row get the column index of snr closest to threshold value of 8
    index_of_min = np.argmin(np.abs(snr_row_m_column_d[i,:] - snr_threshold))
    
    #4.3/.4 - with the column index, get the distance value for that snr
    distance_with_snr_threshold[i] = d_array[index_of_min]

print(distance_with_snr_threshold)     



# 5 - plottinng the snr threshold distance for each mass

# plt.loglog(m_array, distance_with_snr_threshold)
# plt.xlabel('Component Mass (sol)')
# plt.ylabel('Distance (pc)')
# plt.grid()
# #plt.legend()
# plt.title('Mass vs Distance Where SNR of 8.0 is Reached')
# plt.show()

