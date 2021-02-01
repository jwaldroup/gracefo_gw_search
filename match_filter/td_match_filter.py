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
# 6. Perform Match Filter/test snr output

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

#merged_noise = np.array(filtered1) + np.array(filtered2c)
#merged_noise_ts = types.timeseries.TimeSeries(merged_noise, delta_t=0.1) #ensures same delta_t

# 4 - Generate waveform template----------------------------------------------------------------------------

#Generate inspiral model waveform from q_c_orbit_waveform_py2
#binary system parameters
m1 = 500.0 #solar mass multiples
m2 = 500.0
f_low = 0.1
r = 1000.0 #in parsecs
dt = combined_ts.delta_t #0.1
theta = 0.0 

#generate waveform as seen by observer
t_array, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)


#note for later - get abs_tol from amplitude of wave at each r value and input automatically into zero_finder functions
# :)

#truncate such that graph ends by going smoothly to zero
hp = zero_finder.last_zero_finder(hp, abs_tol=1e-14)
hp = zero_finder.first_zero_finder(hp, abs_tol=1e-15)

#convert strain arrays to timeseries objects
hp_ts = types.timeseries.TimeSeries(hp, combined_ts.delta_t) #ensures same delta_t
hc_ts = types.timeseries.TimeSeries(hc, combined_ts.delta_t)

# print('Generated Waveform properties:', 'size:', np.size(hp_ts), 
#          'duration:', hp_ts.duration, 'dt:', hp_ts.delta_t, 
#          'df:', hp_ts.delta_f)


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

# print('Resized Waveform properties:', 'size:', np.size(waveform), 
#         'duration:', waveform.duration, 'dt:', waveform.delta_t, 
#         'df:', waveform.delta_f)

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

# print('injected ts properties:', 'size:', np.size(injected_ts), 
#         'duration:', injected_ts.duration, 'dt:', injected_ts.delta_t, 
#         'df:', injected_ts.delta_f)

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
# plt.legend()ju
# plt.show()

#make sure noise psd is of same delta_f as the noise data timeseries
#combined_psd = welch_function.pycbc_welch(combined_ts, 1) #with no injection
combined_psd = welch_function.pycbc_welch(injected_ts, 3) #with injection
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
# plt.figure()
# plt.plot(snr1.sample_times, abs(snr1), label='abs snr')
# #plt.plot(np.real(snr1), label='real snr')
# plt.ylabel('Signal-to-noise')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(np.real(snr1), label='real snr')
# plt.ylabel('Signal-to-noise')
# plt.xlabel('Time (s)')
# plt.grid()
# plt.legend()
# plt.show()

# check = np.real(snr1)
# print(check.get())


#Below here is my shit doesn't work and I'm trying to see why section :)

# 6.3 - check snr magnitude to see if approximately what is expected--------------------------------------------------------------------

#make unique copies of grace psd frequencyseries and strain timeseries
h_ts = hp_ts.copy() #h(t) 
noise_psd = combined_psd.copy() #S_n(f) in 1/Hz

#take psd of strain timeseries
h_fs = welch_function.pycbc_welch(h_ts, 3)

# 6.3.1 - Adjust h_fs for frequency domain plotting check--------------------------------------------------------------------------
#already checked that noise psd is consistent with what's expected

# plt.figure()
# plt.plot(h_ts.sample_times, h_ts, label ='td waveform ')
# plt.legend()
# plt.grid()

# plt.figure()
# #plt.plot(h_fs.sample_frequencies, np.abs(h_fs) )
# plt.loglog(h_fs.sample_frequencies, np.abs(h_fs), label='fd waveform')
# plt.grid()
# plt.legend()
#plt.show()

# 6.3.2 snr estimate calculation----------------------------------------------------------------------------------------------------------------------------------

#equate df of both frequencyseries < - Note for future generalization - rewrite as if else statement
#print("vector sizes:" , np.size(h_fs), np.size(noise_psd))
#print('df h_fs:', h_fs.delta_f, 'df noise:', noise_psd.delta_f) #to check the df of each
h_fs = psd.interpolate(h_fs, noise_psd.delta_f) #interpolate the larger df of the two to match
#print("vector sizes:" , np.size(h_fs), np.size(noise_psd))

#calculate snr estimate using pycbc welch (original)
#need to also multiply by the df before summing
# signal_psd = np.abs(h_fs)
# psd_ratio = (4.0 * signal_psd * noise_psd.delta_f * 2.0) / noise_psd
# snr_squared_welch = psd_ratio.sum()
# snr_estimate_welch = np.sqrt(snr_squared_welch)

#same as code block directly above but altered by what I think the correct equation with correct units should be
# signal_psd = h_fs
# psd_ratio_welch = (4.0 * signal_psd ) / (noise_psd) 
# snr_squared_welch = psd_ratio_welch.sum()
# snr_estimate_welch = np.sqrt(snr_squared_welch)

signal_psd = h_fs
psd_ratio_pycbc_welch = (4.0 * signal_psd  ) / (noise_psd[:-2]) 
snr_squared_pycbc_welch = psd_ratio_pycbc_welch.sum()
snr_estimate_pycbc_welch = np.sqrt(snr_squared_pycbc_welch)

print("snr estimate via pycbc welch", snr_estimate_pycbc_welch) #, "is complex?", np.iscomplex(snr_estimate_welch))

#calculate snr estimate using numpy fft
freqs = np.fft.fftfreq(np.size(h_ts))
mask = freqs > 0
raw_fft_h_ts = np.fft.fft(h_ts)
psd_of_h_ts = ( 2.0 * np.abs( raw_fft_h_ts / float( np.size(h_ts) ) ) )** 2.0

# #plt.figure()
# # plt.loglog(freqs[mask], psd_of_h_ts[mask], label='fft waveform')
# # plt.legend()
# # plt.xlabel('freqs')
# # plt.show()

#turn psd to frequencyseries with df of fftfreqs
fft_psd = types.frequencyseries.FrequencySeries(psd_of_h_ts[mask], delta_f=(1.0/float(h_ts.duration)))
#fft_psd = types.frequencyseries.FrequencySeries(psd_of_h_ts[mask], delta_f=(freqs[1]-freqs[0]) )
#print('fft df?', freqs[1]-freqs[0], 1.0/float(h_ts.duration))

# #interpolate psd to match noise_psd.delta_f
# print("vector sizes:" , np.size(fft_psd), np.size(noise_psd))
# print('df fft_psd:', fft_psd.delta_f, 'df noise:', noise_psd.delta_f) 
# fft_psd = psd.interpolate(fft_psd, noise_psd.delta_f)
# print("vector sizes:" , np.size(fft_psd), np.size(noise_psd))

print("vector sizes:" , np.size(fft_psd), np.size(noise_psd))
print('df fft_psd:', fft_psd.delta_f, 'df noise:', noise_psd.delta_f) 
noise_psd_1 = psd.interpolate(noise_psd, fft_psd.delta_f)
print("vector sizes:" , np.size(fft_psd), np.size(noise_psd_1))

#calculate snr
psd_ratio_fft = (4.0 * fft_psd) / (noise_psd_1[:-2]) #/ (noise_psd[:-1])

#print("is psd_ratio complex?", np.iscomplex(psd_ratio_fft))
snr_squared_fft = psd_ratio_fft.sum()
snr_estimate_fft = np.sqrt(snr_squared_fft)
 
print("snr estimate via fft", snr_estimate_fft) #, "is complex?", np.iscomplex(snr_estimate_fft))

#revise to include windowing (done via scipy welch function that also uses the Hanning Window like Pycbc Welch)
frequency_array, sp_welch_psd = welch_function.scipy_welch(h_ts, f_s, seg_num)

sp_welch_psd = types.frequencyseries.FrequencySeries(sp_welch_psd, delta_f=(frequency_array[1]-frequency_array[0]))

print("vector sizes:" , np.size(sp_welch_psd), np.size(noise_psd))
print('df sp psd:', sp_welch_psd.delta_f, 'df noise:', noise_psd.delta_f) 
noise_psd_2 = psd.interpolate(noise_psd, sp_welch_psd.delta_f)
print("vector sizes:" , np.size(sp_welch_psd), np.size(noise_psd_2))

psd_ratio_sp_welch = (4.0 * sp_welch_psd) / (noise_psd_2)
snr_squared_sp_welch = psd_ratio_sp_welch.sum()
snr_estimate_sp_welch = np.sqrt(snr_squared_sp_welch)

print("snr estimate via scipy welch", snr_estimate_sp_welch)




#6.3.2.1 check snr estimate's change with distance------------------------------------------------------------------------------------------
import snr_distance_comparison as snr_dc

# distances = np.arange(50, 5000, 50)
# # snr_outputs = snr_dc.snr_distance_plotter(m1, m2, f_low, dt, theta, distances, noise_psd, lgd_label='no injection of waveform in noise data')

# inject_c = injected_ts.copy()
# injected_fs = welch_function.pycbc_welch(inject_c, 3)

# #snr_with_temp_inject_outputs = snr_dc.snr_distance_plotter_wf_self_generating(m1, m2, f_low, dt, theta, distances, injected_fs, lgd_label='welch snr distance comparison')

# snr_outputs = snr_dc.snr_distance_plotter_wf_self_generating(m1, m2, f_low, dt, theta, distances, noise_psd, lgd_label='fft snr distance comparison')
