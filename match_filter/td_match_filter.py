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
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder

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
grace_asd = grace_signal / 220.0e3 #converts between m/sqr(Hz) and 1/sqr(Hz)

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
combined_psd = welch_function.pycbc_welch(combined_ts, 2)

#display some important parameters
#print('Combined psd FrequencySeries:','size:', np.size(combined_psd), 'df:', combined_psd.delta_f)
print('Combined Noise Timeseries:','size:', np.size(combined_ts), 'duration:', 
      combined_ts.duration, 'dt:', combined_ts.delta_t,'df:', 
      combined_ts.delta_f,'f_s:', (1.0/combined_ts.delta_t),'f_nyq:', (1.0/combined_ts.delta_t)/2.0)

##Compare merged noise curves with gracefo data
# plt.loglog(combined_psd.sample_frequencies, np.sqrt(combined_psd), label='test asd')
# #plt.loglog(noise1_asd.sample_frequencies, noise1_asd, label='noise 1')
# #plt.loglog(noise2_asd.sample_frequencies, noise2_asd, label='noise 2')
# plt.loglog(grace_freqs, grace_asd, label='gracefo asd')

# plt.legend()
# plt.xlabel('frequency (Hz)')
# plt.ylabel('strain amplitude spectral density (1/sqrt(Hz))')
# plt.grid()
# plt.legend()

# # plt.savefig('grace_fo_model_curve_comparison.png')
# plt.show()

# 4 - Generate waveform template----------------------------------------------------------------------------

#Generate inspiral model waveform from q_c_orbit_waveform_py2
#binary system parameters
m1 = 500.0 #solar mass multiples
m2 = 500.0
f_low = 0.1
r = 5000.0 #in parsecs
dt = combined_ts.delta_t #0.1
theta = 0.0 

#generate waveform as seen by observer
t_array, hp, hc = q_c_py2.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)


#note for later - get abs_tol from amplitude of wave at each r value and input automatically into zero_finder functions
# :)

#truncate such that graph ends by going smoothly to zero
hp = zero_finder.last_zero_finder(hp, abs_tol=1e-14)
hp = zero_finder.first_zero_finder(hp, abs_tol=1e-15)

#convert strain arrays to timeseries objects
hp_ts = types.timeseries.TimeSeries(hp, combined_ts.delta_t) #ensures same delta_t
hc_ts = types.timeseries.TimeSeries(hc, combined_ts.delta_t)


#Testing matched filter with zero mean sinusoid waveform------------------------------------------------
#Generate test sinusoidal waveform at constant frequency to try injecting instead

# t = np.arange(0, 10*np.pi , 0.1)
# sine_signal = 1e-9 * np.sin(t)
# sine_signal = sine_signal - np.mean(sine_signal)

# plt.plot(t, sine_signal, label='basic zero mean sine')
# plt.legend()
# plt.show()

# print('sine mean:', np.mean(sine_signal))

# waveform = sine_signal.copy()
# waveform = types.timeseries.TimeSeries(waveform, delta_t=combined_ts.delta_t)

#Create template for sine wave test
# match_template = sine_signal.copy()
# match_template = types.timeseries.TimeSeries(match_template, delta_t=combined_ts.delta_t)


# 4.1 preparing waveform for injection-----------------------------------------------------------------------

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
# plt.plot(waveform.sample_times, waveform, label='resized and plotted against time')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(waveform, label='plotted against array elements')
# plt.legend()
# plt.show()


#Resampling of waveform incase waveform generated at dt other than 0.1------------------------------------------------
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

# ## 5 - Injection ----------------------------------------------------------------------------------------------------

# # #Noise curve unique copy
combined_tsc = combined_ts.copy()

# #roll the template vector to a random index using pycbc cyclic time shift
#shift_seconds = 10000 #test example for now - should be random second between 0 and combined_tsc.duration-length of waveform
#random_waveform = waveform.cyclic_time_shift(shift_seconds)#np.roll(waveform, random_index)

#roll template instead with np roll
random_index = 200000
random_waveform = np.roll(waveform, random_index)

## this section checks the roll position of the waveform when uncommented
# rand_wf_ts = types.timeseries.TimeSeries(random_waveform, delta_t=combined_tsc.delta_t)

# plt.figure()
# plt.plot(rand_wf_ts.sample_times, rand_wf_ts, label='resized and plotted against time')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(rand_wf_ts, label='plotted against array elements')
# plt.legend()
# plt.show()

# #inject into timeseries
injected_array = np.array(combined_tsc) + np.array(random_waveform)
injected_ts = types.timeseries.TimeSeries(injected_array, delta_t=combined_tsc.delta_t)

print('injected ts properties:', 'size:', np.size(injected_ts), 
        'duration:', injected_ts.duration, 'dt:', injected_ts.delta_t, 
        'df:', injected_ts.delta_f)

# #display for visual evaluation
# plt.figure()
# plt.plot(injected_ts.sample_times, injected_ts, label=('random shift '+str(random_index)+' pts'))
# plt.xlabel('times (s)')
# plt.ylabel('Noise + hidden signal strain amplitude')
# plt.grid()
# plt.legend()
# plt.show()

## 6 - Matched filter - condition the noise curve and prepare psd -----------------------------------

# 6.1 Prepatory steps and creating the match template-------------------------------------------------

#Highpass the noise curve with injected waveform above 10e-2 Hz
injected_ts_highpass = highpass(injected_ts, 0.01)

#crop to avoid filter wraparound
conditioned = injected_ts_highpass.crop(2,2)

#display to check for excessive wraparound -> increase crop length
# plt.figure()
# plt.plot(conditioned.sample_times, conditioned, label='conditioned data for matched filter')
# plt.xlabel('times (s)')
# plt.ylabel('Noise + hidden signal strain amplitude')
# plt.legend()
# plt.show()

#make sure psd is of same delta_f as the noise data timeseries
grace_psd = psd.interpolate(combined_psd, conditioned.delta_f)

#create the template for the matched filter 
match_template = hp_ts.copy()
#match_template = types.timeseries.TimeSeries(match_template, delta_t = 0.1)

#get the match template to the same size as the noise data 
match_template.resize(np.size(conditioned))

# plt.figure()
# plt.plot(match_template.sample_times, match_template, label='match template')
# plt.xlabel('time (s)')
# plt.legend()
# plt.show()

#Tutorial says rotate the template to have merger at approx the first bin
#match_template = match_template.cyclic_time_shift((match_template.start_time - hp_ts.duration + 0.1))

# match_roll = np.roll(match_template, -np.size(match_template)/4 )
# match_template = types.timeseries.TimeSeries(match_roll, delta_t=conditioned.delta_t)

# plt.figure()
# plt.plot(match_template.sample_times, match_template, label='match template rolled')
# plt.xlabel('time (s)')
# plt.legend()
# plt.show()

#Check properties and Perform the Matched filtering
# print('match template, conditioned data, grace_psd:', 'sizes:', np.size(match_template), np.size(conditioned), np.size(grace_psd),
#       'dt:', match_template.delta_t, conditioned.delta_t, grace_psd.delta_t, 
#       'df:', match_template.delta_f, conditioned.delta_f, grace_psd.delta_f)

# 6.2 - Perform the matched filter via pycbc's filter module-------------------------------------------------------------------------
snr1 = matched_filter(match_template, conditioned, psd=grace_psd)

snr1 = snr1.crop(10,10)

#Viewing matched filter snr timeseries
plt.plot(snr1.sample_times, abs(snr1), label='abs snr')
#plt.plot(np.real(snr1), label='real snr')
plt.ylabel('Signal-to-noise')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.show()

# plt.figure()
# plt.plot(np.real(snr1), label='real snr')
# plt.ylabel('Signal-to-noise')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.legend()
# plt.show()

# check = np.real(snr1)
# print(check.get())

# 6.3 - check snr magnitude to see if approximately what is expected--------------------------------------------------------------------

#make unique copies of grace psd frequencyseries and strain timeseries
h_ts = hp_ts.copy() #h(t) 
noise_psd = combined_psd.copy() #S_n(f) in 1/Hz

#extend length of template to match noise timeseries length
h_ts.resize(np.size(combined_ts))

#take psd of strain timeseries
h_fs = welch_function.pycbc_welch(h_ts, 1)

plt.figure()
plt.loglog(h_fs.sample_frequencies, np.abs(h_fs))
plt.grid()
plt.show()

#trial of psd via np.fft instead
# h_fs = np.fft.fft(h_ts)
# h_fs = types.frequencyseries.FrequencySeries(h_fs, delta_f=noise_psd.delta_f)

#equate df of both frequencyseries
print('df h_fs:', h_fs.delta_f, 'df noise:', noise_psd.delta_f)
h_fs = psd.interpolate(h_fs, noise_psd.delta_f) #interpolate the larger df of the two to match
print("vector sizes:" , np.size(h_fs), np.size(noise_psd))

#calculate snr estimate
#need to also multiply by the df before summing
psd_ratio = (noise_psd.delta_f * 4.0 * (np.abs(h_fs)) ) / np.abs(noise_psd) #form "integrand" ratio
snr_squared = psd_ratio.sum() #take the discrete sum 


snr_estimate = np.sqrt(snr_squared)
print("snr estimate", snr_estimate, np.iscomplex(snr_estimate))

#calculate difference between peak value of actual matched filter snr and estimate
theoretical_difference = (max(np.abs(snr1))) - snr_estimate
print("Theoretical and actual snr difference:", theoretical_difference)

