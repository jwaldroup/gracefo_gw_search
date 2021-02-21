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
grace_asd = grace_signal / 220.0e3 #converts between m/sqr(Hz) and 1/sqr(Hz)


### 2 - Build Noise Curve Model

##Curve 1: the "hump"---------------------------------------------------------------------------------------------------------------------

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


##Curve 2: the "linear" portion------------------------------------------------------------------------------
#N = 400000
#cutoff = 0.0001 #0.001 #0.001   
#order = 3400 #3000 #1000  
#beta = 9.0 #8.0 #1  
#seg_num = 3

#N = 2000000
#seg_num = 15
#
##[1000, 3400, 5800] for order
#for i in [20000]:
#    for j in [15.0]:
#        
#        cutoff = 0.00001 #0.0001 # #0.0008 # #0.001 #0.001   
#        order = i #20000
#        beta = j #12.0   at current vertical amplitude scaling: 10e-7
#        
#
##noise signal
#        np.random.seed(138374923)
#        noise2 = np.random.uniform(-1, 1, size=N)
#    
##convert to TimeSeries pycbc object
#        noise2_ts = types.timeseries.TimeSeries(noise2, delta_t=0.1)
#
##adjust amplitude
#        noise2_ts = noise2_ts * 10e-6
#
##filter it
#        filtered2 = noise2_ts.lowpass_fir(cutoff, order, beta=beta)
#
##psd.welch to create psd
#        noise2_psd = welch_function.pycbc_welch(filtered2, seg_num)
#
##conversion to asd
#        noise2_asd = np.sqrt(noise2_psd)
#
#        #plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label=(str(i)+ " " + str(j)) )

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


#----------------------------------------------------------------------------------------------------------

## 3 - Merge noise curves

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
noise_psd = welch_function.pycbc_welch(merged_noise_ts, seg_num)
plt.loglog(noise_psd.sample_frequencies, np.sqrt(noise_psd), label='noise model asd')
plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')
plt.loglog(grace_freqs, grace_asd, label='gracefo asd')

plt.legend()
plt.xlabel('frequency (Hz)')
plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
plt.grid()

plt.show()

## 4 - export as csv file----------------------------------------------------------------------------------
#Import data from csv

fileobj = open('noise_model.csv', 'w')
writerobj = csv.writer(fileobj)

timeseries_data = merged_noise_ts.copy()

writerobj.writerows(timeseries_data)
fileobj.close()