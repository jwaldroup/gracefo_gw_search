# -*- coding: utf-8 -*-
"""
fd match filter

@author: john
"""

#Imports
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt

from pycbc.waveform import get_fd_waveform
from pycbc.filter import highpass, matched_filter
from pycbc import types
from pycbc import psd

import pycbc_welch_function as welch_function
import q_c_orbit_waveform_gen_functions as q_c_apx

#Import and display GRACE-FO data
#2. Model data with two white noise curves and lowpass filters
#3. Merge the noise curves
#4. Generate fd waveform template from approximant
#5. Get timeseries of waveform template via ifft
#6. Inject one polarizations strain from template into noise curve
#7. Perform Match Filter

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
print('Noise Curve 1 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)

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
print('Noise Curve 2 - ','N:', N, 'dt:', 0.1,'df:', df,'f_s:', f_s,'f_nyq:', f_nyq)

# 3 - Merge noise curves---------------------------------------------------------------------------------------------------------

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
print('Combined psd frequencyseries:','size:', np.size(combined_fs), 
      'df:', combined_fs.delta_f)
print('Combined Noise Timeseries:','size:', np.size(combined_ts), 'duration:', 
      combined_ts.duration, 'dt:', combined_ts.delta_t,'df:', 
      combined_ts.delta_f,'f_s:', (1.0/combined_ts.delta_t),'f_nyq:', (1.0/combined_ts.delta_t)/2)

# 4 - Generate Frequency Domain Waveform-------------------------------------------------------------------

#apx = 'TaylorF2'
apx = 'IMRPhenomD'

m1 = 100.0
m2 = 100.0
f_low = 0.01
df = combined_fs.delta_f

hp_fd, hc_fd = get_fd_waveform(approximant = apx,
                               mass1=m1, mass2=m2,
                               f_lower=f_low,
                               delta_f=df)

plt.loglog(hp_fd.sample_frequencies, np.real(hp_fd), label='real val waveform')
plt.loglog(hp_fd.sample_frequencies, hp_fd, label='raw waveform')
plt.xlabel('freq (Hz)')
plt.ylabel('Asd (?)')
plt.legend(loc='upper right')
plt.show()

print(hp_fd.sample_frequencies[int(np.size(hp_fd.sample_frequencies)/2 - 10): int(np.size(hp_fd.sample_frequencies)/2 + 10)])
