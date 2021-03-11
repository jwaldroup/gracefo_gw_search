# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:26:58 2021

@author: John Waldroup
"""

#Filter editing code section from td_match_filter.py

import numpy as np
#import scipy as sp
import csv
import matplotlib.pyplot as plt

#from pycbc.filter import highpass, matched_filter
from pycbc import types
#from pycbc import psd

import pycbc_welch_function as welch_function
#import q_c_orbit_waveform_py2 as q_c_py2
#import zero_finder

#Script Overview
# 1. Import and display GRACE-FO data
# 2. Model data with two white noise curves and lowpass filters
# 3. Merge the noise curves
# 4. export the noise curve model as a csv file


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
order = 20000
beta = 20.0

#noise signal
np.random.seed(138374923)
noise2 = np.random.uniform(-1, 1, size=N)
    
#convert to TimeSeries pycbc object
noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)

#adjust amplitude
noise2_ts = noise2_ts * 10e-8

for i in [14000]:
    for j in [11.0]:
        filtered2 = noise2_ts.lowpass_fir(cutoff, i, beta=j)


        filtered2c = filtered2.copy()
        print(np.size(filtered1), np.size(filtered2c))
        filtered2c.append_zeros((np.size(filtered1)-np.size(filtered2)))
        
        noise2_psd = welch_function.pyc_welch(filtered2c, seg_size)
        noise2_asd = np.sqrt(noise2_psd)
        
        plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label=(str(i)+', '+str(j)))



#uncomment to check that the sizes match
print(np.size(filtered1), np.size(filtered2c))

#Add the two 
merged_noise = np.array(filtered1) + np.array(filtered2c)
merged_noise_ts = types.timeseries.TimeSeries(merged_noise, delta_t=0.1) #ensures same delta_t

#psd.welch to create psd
noise1_psd = welch_function.pyc_welch(filtered1, seg_size)
noise1_asd = np.sqrt(noise1_psd)



noise_psd = welch_function.pyc_welch(merged_noise_ts, seg_size)

#Compare merged noise curves with gracefo data
plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
plt.loglog(grace_freqs, grace_asd, label='gracefo asd')
plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')

plt.legend()
plt.xlabel('frequency (Hz)')
plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
plt.grid()
plt.show()


## 4 - export as csv file----------------------------------------------------------------------------------

# fileobj = open('noise_model.csv', 'w')
# writerobj = csv.writer(fileobj)

# timeseries_data = merged_noise_ts.copy()

# writerobj.writerows(timeseries_data)
# fileobj.close()