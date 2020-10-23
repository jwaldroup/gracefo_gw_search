# -*- coding: utf-8 -*-
"""

td generated waveform match filter script

John Waldroup
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
#import q_c_orbit_waveform_gen_functions as q_c_apx
import q_c_orbit_waveform_py2 as q_c_py2


#Script Overview
# 1. Import and display GRACE-FO data
# 2. Model data with two white noise curves and lowpass filters
# 3. Merge the noise curves
# 4. Generate waveform template from approximant
# 5. Inject one polarizations strain from template into noise curve
# 6. Perform Match Filter

## 1 - Read in Grace-FO data to model------------------------------------------

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
grace_strain = grace_signal / 220.0e3 #converts between m/sqr(Hz) and 1/sqr(Hz)

### 2 - Build Noise Curve Model

##Curve 1: the "hump"---------------------------------------------------------------------------------------------------------------------
N = 400000 
cutoff = 0.001 #0.001 #0.001  
order = 1000 #610 #570 
beta = 18 #11.75 #11.0 
seg_num = 3

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

#some parameters
df = noise1_asd.delta_f
dt = filtered1.delta_t 
T = dt * N
f_s = 1.0 / dt
f_nyq = f_s / 2.0
#print('Noise Curve 1 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)

##Curve 2: the "linear" portion------------------------------------------------------------------------------
N = 400000
cutoff = 0.0001 #0.001 #0.001   
order = 3400 #3000 #1000  
beta = 9.0 #8.0 #1  
seg_num = 3

#noise signal
np.random.seed(138374923)
noise2 = np.random.uniform(-1, 1, size=N)
    
#convert to TimeSeries pycbc object
noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)

#adjust amplitude
noise2_ts = noise2_ts * 10e-9

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
#print('Noise Curve 2 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)


#----------------------------------------------------------------------------------------------------------

## 3 - Merge noise curves

#uncomment to check dimensions of timeseries
#print(np.size(filtered1), np.size(filtered2))

#pad the smaller one to equivalent lengths
filtered2c = filtered2.copy()
filtered2c.append_zeros((np.size(filtered1)-np.size(filtered2)))
#print(np.size(filtered1), np.size(filtered2c))

#Add the two
combined = np.array(filtered1) + np.array(filtered2c)
combined_ts = types.timeseries.TimeSeries(combined, filtered1.delta_t) #ensures same delta_t

#welch's psd
combined_fs = welch_function.pycbc_welch(combined_ts, 2)

#display some important parameters
#print('Combined psd FrequencySeries:','size:', np.size(combined_fs), 'df:', combined_fs.delta_f)
print('Combined Noise Timeseries:','size:', np.size(combined_ts), 'duration:', 
      combined_ts.duration, 'dt:', combined_ts.delta_t,'df:', 
      combined_ts.delta_f,'f_s:', (1.0/combined_ts.delta_t),'f_nyq:', (1.0/combined_ts.delta_t)/2.0)

##Compare merged noise curves with gracefo data
# plt.loglog(combined_fs.sample_frequencies, np.sqrt(combined_fs), label='test')
# #plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
# #plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')
# plt.loglog(grace_freqs, grace_strain, label='gracefo')

# plt.legend()
# plt.xlabel('frequency (Hz)')
# plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
# plt.grid()
# plt.legend()

# # plt.savefig('grace_fo_model_curve_comparison.png')
# plt.show()

# 4 - Generate time domain noise curve----------------------------------------------------------------------------


#binary system parameters
m1 = 500.0 #solar mass multiples
m2 = 500.0
f_low = 0.1
r = 1.0 #in parsecs
dt = 0.1 #combined_ts.delta_t
theta = 0.0 

#generate waveform as seen by observer
wf_t_array, hp, hc = q_c_py2.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)


#convert strain arrays to timeseries objects
hp_ts = types.timeseries.TimeSeries(hp, combined_ts.delta_t) #ensures same delta_t
hc_ts = types.timeseries.TimeSeries(hc, combined_ts.delta_t)

print('Generated waveform properties', 'size:', np.size(hp_ts), 
      'duration:', hp_ts.duration, 'dt:', hp_ts.delta_t, 
      'df:', hp_ts.delta_f)

# 4.1 preparing waveform for injection

#Adjusting the waveform for injection 
#need time duration to be equivalent to the noise curve time (combined_ts.duration) = 39800 s

# #Copy Waveform Template
waveform = hp_ts.copy()

# #increase length of waveform to match noise curve
waveform.resize(np.size(combined_ts))

print('Resized Waveform properties:', 'size:', np.size(waveform), 
        'duration:', waveform.duration, 'dt:', waveform.delta_t, 
        'df:', waveform.delta_f)

#plot waveform after resizing
# plt.figure()
# plt.plot(waveform.sample_times, waveform, label='resized')
# #plt.xlim(10000,12000)
# plt.legend()
# plt.show()

#Resample ( I may not need this if my waveform generator already creates a waveform at 0.1 dt)
#resample_num = int(waveform.duration / 0.1)
#waveform_resampled = sp.signal.resample(waveform, resample_num)
#waveform = types.timeseries.TimeSeries(waveform_resampled, delta_t = 0.1)
#print('Resampled waveform:', 'size:', np.size(waveform), 'duration:', 
#      waveform.duration, 'dt:', waveform.delta_t, 'df:', waveform.delta_f )

# waveform.resize(np.size(combined_ts))
# print('Resampled and resized:', 'size:', np.size(waveform), 
#       'duration:', waveform.duration, 'dt:', waveform.delta_t, 
#       'df:', waveform.delta_f )

# plt.figure()
# plt.plot(waveform.sample_times, waveform, label='waveform resampled')
# #plt.xlim(10000,12000)
# plt.legend()
# plt.show()

# ## 5 - Injection ----------------------------------------------------------------------------------------------------

# # #Noise curve unique copy
combined_tsc = combined_ts.copy()

# #roll the template vector to a random index
random_index = 0
random_waveform = np.roll(waveform, random_index)

# #inject into timeseries
injected_array = np.array(combined_tsc) + np.array(random_waveform)
injected_ts = types.timeseries.TimeSeries(injected_array, delta_t=combined_tsc.delta_t)

print('injected ts properties:', 'size:', np.size(injected_ts), 
        'duration:', injected_ts.duration, 'dt:', injected_ts.delta_t, 
        'df:', injected_ts.delta_f)

# #display for examination
test_waveform = types.timeseries.TimeSeries(random_waveform, delta_t=combined_tsc.delta_t)

# plt.figure()
# plt.plot(waveform.sample_times, waveform, label='original')
# plt.plot(test_waveform.sample_times, test_waveform, label='random shift')
# #plt.xlim(10000, 10050)
# plt.grid()
# plt.legend()
# plt.show()

# ## 6 - Matched filter - condition the noise curve and prepare psd -----------------------------------


#Highpass the noise curve with injected waveform above 10e-2 Hz
injected_ts_highpass = highpass(injected_ts, 0.01)

#crop to avoid filter wraparound
conditioned = injected_ts_highpass.crop(2,2)

#display to check for excessive wraparound -> increase crop length
plt.figure()
plt.plot(conditioned.sample_times, conditioned, label='conditioned data for matched filter')
plt.legend()
plt.show()

# #make sure psd is of same delta_f as the noise data timeseries
grace_psd = psd.interpolate(combined_fs, conditioned.delta_f)

# #adjust template for match
match_template = hp.copy()

#match_template.resize(np.size(combined_tsc))

# resample_num = int(match_template.duration / 0.1)
# match_template_resampled = sp.signal.resample(match_template, resample_num)

#match_template = types.timeseries.TimeSeries(match_template_resampled, delta_t = 0.1)
match_template = types.timeseries.TimeSeries(match_template, delta_t = 0.1)

#get the match template to the same size as the noise data and rotate the match template so the merger is apprx
#at the first bin
match_template.resize(np.size(conditioned))
match_template = match_template.cyclic_time_shift(-hp_ts.end_time+100)

# #Check properties and Perform the Matched filtering
# print('mt, cond., grace_psd:', 'sizes:', np.size(match_template), np.size(conditioned), np.size(grace_psd),
#       'dt:', match_template.delta_t, conditioned.delta_t, grace_psd.delta_t, 
#       'df:', match_template.delta_f, conditioned.delta_f, grace_psd.delta_f)

#check to see if shift approximately has merger at start of the data
plt.figure()
plt.plot(match_template.sample_times, match_template, label='match template')
#plt.xlim(-0,50)
plt.legend()
plt.show()

snr1 = matched_filter(match_template, conditioned, psd=grace_psd)

snr1 = snr1.crop(10,10)

#Viewing matched filter snr timeseries
plt.plot(snr1.sample_times, abs(snr1), label='snr')
plt.ylabel('Signal-to-noise')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.show()