 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 02:42:19 2021

@author: john
"""

#Imports
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt

from pycbc.filter import highpass, matched_filter
from pycbc import types
from pycbc import psd

import pycbc_welch_function as welch_function
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder

#Script Overview
# 1. Import and display GRACE-FO data
# 2. Model data with two white noise curves and lowpass filters
# 3. Merge the noise curves
# 4. Generate waveform template from approximant
# 5. Inject one polarizations strain from template into noise curve
# 6. Perform Match Filter/test snr output

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


## 2. Model data with two white noise curves and lowpass filters------------------------------------------------------------

#Curve 1 parameters------------------------------------------------------------

#0.5k sol mass
# N = 2000000
# cutoff = 0.001 
# order = 1000 
# beta = 17  
# seg_size = 760000

#1k sol mass
N = 2000000 #Possibly increase in length to incorporate lower component mass
cutoff = 0.001 
order = 950 
beta = 17  
seg_size = 150000

# #5k sol mass
# N = 2000000
# cutoff = 0.001 
# order = 1000 
# beta = 17  
# seg_size = 17000

# #10k sol mass
# N = 2000000 
# cutoff = 0.001 
# order = 1000 
# beta = 17  
# seg_size = 5000

#noise signal
np.random.seed(138374923)
noise1 = np.random.uniform(-1, 1, size=N)
            
#convert to TimeSeries pycbc object
noise1_ts = types.timeseries.TimeSeries(noise1, delta_t=0.1) #delta_t = 0.1 to match gracefo sample frequency of 10 Hz
        
#adjust amplitude

#0.5k sol mass, 1k sol mass
noise1_ts = noise1_ts * 10e-8

#5k sol mass, 10k sol mass
#noise1_ts = noise1_ts * 10e-9
        
#filter it
filtered1 = noise1_ts.lowpass_fir(cutoff, order, beta=beta)


#Curve 2 parameters------------------------------------------------------------

# #0.5k sol mass
# cutoff = 0.00001
# order = 40000
# beta = 12.0

#1k sol mass
cutoff = 0.00001
order = 14000
beta = 11.0

# #5k sol mass
# cutoff = 0.001
# order = 14000
# beta = 11.0

# # #10k sol mass
# cutoff = 0.001
# order = 14000
# beta = 11.0

#noise signal
np.random.seed(138374923)
noise2 = np.random.uniform(-1, 1, size=N)
    
#convert to TimeSeries pycbc object
noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)

#adjust amplitude

#0.5k sol mass, 1k sol mass
noise2_ts = noise2_ts * 10e-8

#5k sol mass, 10k sol mass
#noise2_ts = noise2_ts * 10e-11

#filter it
filtered2 = noise2_ts.lowpass_fir(cutoff, order, beta=beta)


# 3 - Merge noise curves---------------------------------------------------------------------------------------

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

#psd.welch to create psd
noise_psd = welch_function.pyc_welch(merged_noise_ts, seg_size)
noise1_psd = welch_function.pyc_welch(filtered1, seg_size)
noise1_asd = np.sqrt(noise1_psd)

noise2_psd = welch_function.pyc_welch(filtered2c, seg_size)
noise2_asd = np.sqrt(noise2_psd)

#Compare merged noise curves with gracefo data
# plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
# plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')

# plt.loglog(grace_freqs, grace_asd, label='gracefo asd')
# plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')

# plt.legend()
# plt.xlabel('frequency (Hz)')
# plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
# plt.grid()
# plt.show()

# 4 - Generate waveform template----------------------------------------------------------------------------

#Original seg size 150000
#10000 sol mass gives under estimate by factor of 200
#500 sol mass gives over estimate by about a factor of 2 [4000, 7000, 15000] wf 760000, diff 61000
#1000 sol mass  close, wf 239505, diff  89505
#1500 [8000, 15000, 25000] half of what it should be, wf 121802, diff -28198

#Segment sizing checklist
# window size (seg_size)?
# filter cutoff?
# filter order?
# filter beta parameter?
# noise data amplitude scaling?
# distances at apprx expected snr of 8, 4, 2?
# mass in wf generator?

#Mass and window segment size matching equivalent to wf length at snr of 8, 4, 2 (respectively) list
#0.5k sol mass match within ~0.4 at [3800, 7000, 16000]
#1k sol mass match within ~0.5 at []
#5k sol mass match within ~1.2 at [13000, 25000, 50000]
#10k sol mass match within ~2.1 at [12000, 22000, 45000]



# for i in [3800, 7000, 16000]:
#     #Generate inspiral model waveform from q_c_orbit_waveform_py2
#     #binary system parameters
#     m1 = 500.0 #solar mass multiples
#     m2 = m1
#     f_low = 0.05
#     r = i
#     # r = 6000.0 #in parsecs
#     dt = 0.01
#     theta = 0.0 
    
#     #generate inspiral waveform
#     t_array, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)
                    
#     #downsample the waveform to gracef's sampling frequency (ie dt=0.1)
#     T = dt * np.size(hp)
#     dt = 0.1
#     resample_x_values = np.arange(0, T, dt)
#     hp = np.interp(resample_x_values, t_array, hp.copy())
            
#     #prep waveform with truncation
#     hp_cut = zero_finder.first_zero_finder_02(hp, m1, f_low, dt)
#     obs_t_cut = resample_x_values[-np.size(hp_cut):]
            
#     hp_cut = zero_finder.last_zero_finder_03(hp_cut, m1, dt)
#     obs_t_cut = obs_t_cut[0:np.size(hp_cut)]
    
#     hp_ts = types.timeseries.TimeSeries(hp_cut, delta_t=dt)
    
#     # ## 5 - Injection ----------------------------------------------------------------------------------------------------
    
#     # # #Noise curve unique copy
#     merged_noise_tsc = merged_noise_ts.copy()
    
#     #copy of waveform with additional padding
#     wf_for_injection = hp_ts.copy()
#     wf_for_injection.resize(np.size(merged_noise_ts))
    
#     #stand in for random injection
#     random_index = 700000
#     random_waveform = np.roll(wf_for_injection, random_index)
    
#     random_waveform = types.timeseries.TimeSeries(random_waveform, delta_t=dt)
    
#     # plt.figure()
#     # plt.plot(random_waveform.sample_times, random_waveform, label='wf injection location')
#     # plt.legend()
#     # plt.grid()
#     # plt.show()
    
#     injected_array = np.array(merged_noise_tsc) + np.array(random_waveform)
#     signal_and_noise = types.timeseries.TimeSeries(injected_array, delta_t=merged_noise_tsc.delta_t)
    
#     ## 6 - Matched filter - condition the noise curve and prepare psd -----------------------------------
    
#     # 6.1 Prepatory steps and creating the match template-------------------------------------------------
    
#     #Highpass the noise curve with injected waveform above 1e-2 Hz
#     injected_ts_highpass = highpass(signal_and_noise, 0.01)
    
#     #crop to avoid filter wraparound
#     conditioned = injected_ts_highpass.crop(2,2)
    
#     #template
#     template = hp_ts.copy()
#     template.resize(np.size(conditioned))
#     template = template.cyclic_time_shift(conditioned.duration)
    
#     # plt.figure()
#     # plt.plot(template, label='template rotated for filter')
#     # plt.grid()
#     # plt.legend()
#     # plt.plot()
#     # plt.show()
    
#     # seg_size = np.size(hp_ts)
#     # noise_psd = welch_function.pyc_welch(merged_noise_ts.copy(), seg_size)
#     # grace_psd = psd.interpolate(noise_psd, conditioned.delta_f)
    
#     #matched_filter
#     #snr1 = matched_filter(template, conditioned, psd=grace_psd) 
    
#     #snr1 = snr1.crop(10, 10)
    
#     #Viewing matched filter snr timeseries
#     # plt.figure()
#     # plt.plot(snr1.sample_times, abs(snr1), label='abs snr')
#     # plt.ylabel('Signal-to-noise')
#     # plt.xlabel('Time (s)')
#     # plt.grid()
#     # plt.legend()
#     # plt.show()
    
#     hp_psd = welch_function.pyc_welch(hp_ts, np.size(hp_cut))
#     noise_psd = welch_function.pyc_welch(merged_noise_ts.copy(), np.size(hp_cut))
#     #print('wf length: ', np.size(hp_cut))

#     #seg_size = 150000

#     print('wf length difference with seg size: ', np.size(hp_cut)-seg_size)
            
    
#     #3.3 compute snr estimate
#     psd_ratio = (4.0 * hp_psd) / (noise_psd)
#     snr_squared = psd_ratio.sum()
#     snr_estimate = np.sqrt(snr_squared)
    
#     print(snr_estimate)



#4 generate waveform template---------------------------------------------------------------------
    

#Generate inspiral model waveform from q_c_orbit_waveform_py2
#binary system parameters
m1 = 1000.0 #solar mass multiples
m2 = m1
f_low = 0.05
r = 6000
dt = 0.01
theta = 0.0 
    
#generate inspiral waveform
t_array, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)
                
#downsample the waveform to gracef's sampling frequency (ie dt=0.1)
T = dt * np.size(hp)
dt = 0.1
resample_x_values = np.arange(0, T, dt)
hp = np.interp(resample_x_values, t_array, hp.copy())
            
#prep waveform with truncation
hp_cut = zero_finder.first_zero_finder_02(hp, m1, f_low, dt)
obs_t_cut = resample_x_values[-np.size(hp_cut):]
            
hp_cut = zero_finder.last_zero_finder_03(hp_cut, m1, dt)
obs_t_cut = obs_t_cut[0:np.size(hp_cut)]
    
hp_ts = types.timeseries.TimeSeries(hp_cut, delta_t=dt)
    
## 5 - Injection ----------------------------------------------------------------------------------------------------
    
#Noise curve unique copy
merged_noise_tsc = merged_noise_ts.copy()
    
#copy of waveform with additional padding
wf_for_injection = hp_ts.copy()
wf_for_injection.resize(np.size(merged_noise_ts))
    
#stand in for random injection
random_index = 700000
random_waveform = np.roll(wf_for_injection, random_index)
    
random_waveform = types.timeseries.TimeSeries(random_waveform, delta_t = merged_noise_ts.delta_t)
    
plt.figure()
plt.plot(random_waveform.sample_times, random_waveform, label='wf injection location')
plt.legend()
plt.xlabel('Time (s)')
plt.grid()
plt.show()
    
injected_array = np.array(merged_noise_tsc) + np.array(random_waveform)
signal_and_noise = types.timeseries.TimeSeries(injected_array, delta_t=merged_noise_tsc.delta_t)
    
## 6 - Matched filter - condition the noise curve and prepare psd -----------------------------------
    
# # 6.1 Prepatory steps and creating the match template-------------------------------------------------
    
#Highpass the noise curve with injected waveform above 1e-2 Hz
injected_ts_highpass = highpass(signal_and_noise, 0.01)
    
#crop to avoid filter wraparound
conditioned = injected_ts_highpass.crop(2,2)
    
#template
template = hp_ts.copy()
template.resize(np.size(conditioned))
#template = template.cyclic_time_shift(conditioned.duration)
# template = np.roll(template, (np.size(template) - np.size(hp_ts) ))
# template = types.timeseries.TimeSeries(template, delta_t=dt)
    
plt.figure()
plt.plot(template.sample_times, template, label='template rotated for filter')
plt.grid()
plt.legend()
plt.show()


noise_psd = welch_function.pyc_welch(merged_noise_ts.copy(), seg_size)



#grace_psd = psd.inverse_spectrum_truncation(noise_psd, int((dt*seg_size) * conditioned.sample_rate), low_frequency_cutoff=0.01)
grace_psd = psd.interpolate(noise_psd, conditioned.delta_f)

plt.figure()
plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')
plt.loglog(grace_psd.sample_frequencies, np.sqrt(grace_psd), label='interpolated noise curve')

plt.grid()
plt.legend()
plt.show()


#matched_filter
snr1 = matched_filter(template, conditioned,  psd=grace_psd, low_frequency_cutoff=0.01) 
    
snr1 = snr1.crop(10, 10)
    
#Viewing matched filter snr timeseries
plt.figure()
plt.plot(snr1.sample_times, abs(snr1), label='abs snr')
plt.ylabel('Signal-to-noise')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.show()
    
# print('hp_ts.size' , np.size(hp_ts))

# if seg_size > np.size(hp_ts):
    
#     quotient = seg_size / np.size(hp_ts)
#     print(quotient)
#     wf_periodic = hp_ts.copy()
    
#     count = quotient
#     while quotient > 1:
        
#         wf_periodic = np.concatenate((wf_periodic, hp_ts.copy()), axis=0)
#         quotient = quotient - 1
    
#     print(wf_periodic[np.argmax(wf_periodic)])
#     wf_periodic = (1/float(count) ) * wf_periodic
#     print(wf_periodic[np.argmax(wf_periodic)])
#     print('wf_periodic.size', np.size(wf_periodic))
    
#     wf_periodic = types.timeseries.TimeSeries(wf_periodic, delta_t=hp_ts.delta_t)
#     wf_periodic.resize(seg_size)
    
# #hp_ts = types.timeseries.TimeSeries(hp_ts, delta_t=dt)


# hp_psd = welch_function.pyc_welch(wf_periodic, seg_size) #segment size must be equiv to ts input
# noise_psd = welch_function.pyc_welch(merged_noise_ts.copy(), seg_size)
# #print('wf length: ', np.size(hp_cut))
# #print('wf length difference with seg size: ', np.size(hp_cut)-seg_size)
            
    
# #3.3 compute snr estimate
# psd_ratio = (4.0 * hp_psd) / (noise_psd)
# snr_squared = psd_ratio.sum()
# snr_estimate = np.sqrt(snr_squared)
    
# print(snr_estimate)


