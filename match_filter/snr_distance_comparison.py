# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 01:22:43 2020

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt
import q_c_orbit_waveform_py2 as q_c_py2

from pycbc import types
from pycbc import psd
import pycbc_welch_function as welch_function

def snr_distance_plotter_wf_self_generating(mass1, mass2, f_lower_bound, dt, theta, distance_array, noise_psd):
    
    snr_values_pycbc_welch = []
    snr_values_fft = []
    snr_values_scipy_welch = []
    
    for r in distance_array:
        times, plus_strain, cross_strain = q_c_py2.obs_time_inspiral_strain(mass1, mass2, f_lower_bound, dt, r, theta)
        
        h_ts = types.timeseries.TimeSeries(plus_strain, dt)
        
        #find psd via welch-------------------------------------------------------------------------------------------
        h_fs_pycbc = welch_function.pycbc_welch(h_ts.copy(), 15) #15 is number of segments used
        
        #interpolate the larger df of the two to match
        h_fs_pycbc = psd.interpolate(h_fs_pycbc, noise_psd.delta_f) 
        
        #calculate snr
        psd_ratio_pycbc_welch = (4.0 * h_fs_pycbc  ) / (noise_psd) 
        snr_squared_pycbc_welch = psd_ratio_pycbc_welch.sum()
        snr_estimate_pycbc_welch = np.sqrt(snr_squared_pycbc_welch)
        
        #add to snr list
        snr_values_pycbc_welch.append(snr_estimate_pycbc_welch)

        
        #find psd via fft--------------------------------------------------------------------------------------------

        #take fft and calculate psd
        freqs = np.fft.fftfreq(np.size(h_ts))
        mask = freqs > 0
        raw_fft_h_ts = np.fft.fft(h_ts.copy())
        psd_of_h_ts = ( 2.0 * np.abs( raw_fft_h_ts / float( np.size(h_ts) ) ) )** 2.0
        
        #turn psd to frequencyseries with df of fftfreqs
        fft_psd = types.frequencyseries.FrequencySeries(psd_of_h_ts[mask], delta_f=(1.0/float(h_ts.duration)))

        fft_psd = psd.interpolate(fft_psd, noise_psd.delta_f)

        psd_ratio_fft = (4.0 * fft_psd) / (noise_psd[:-2])
        snr_squared_fft = psd_ratio_fft.sum()
        snr_estimate_fft = np.sqrt(snr_squared_fft)
        snr_values_fft.append(snr_estimate_fft)
        
        #find psd via scipy welch------------------------------------------------------------------------------------
        f_s = 10 #sampling frequency
        frequency_array, sp_welch_psd = welch_function.scipy_welch(h_ts.copy(), f_s, 15) #15 to match the pycbc welch segment #

        sp_welch_psd = types.frequencyseries.FrequencySeries(sp_welch_psd, delta_f=(frequency_array[1]-frequency_array[0]))

        sp_welch_psd = psd.interpolate(sp_welch_psd, noise_psd.delta_f)
        
        psd_ratio_sp_welch = (4.0 * sp_welch_psd) / (noise_psd)
        snr_squared_sp_welch = psd_ratio_sp_welch.sum()
        snr_estimate_sp_welch = np.sqrt(snr_squared_sp_welch)
        snr_values_scipy_welch.append(snr_estimate_sp_welch)
        
        
    plt.loglog(distance_array, snr_values_fft, label='fft')
    plt.loglog(distance_array, snr_values_pycbc_welch, label='pycbc welch')
    plt.loglog(distance_array, snr_values_scipy_welch, label='scipy welch')
    plt.grid()
    plt.xlabel('distance in pc')
    plt.ylabel('snr')
    plt.legend(loc="upper right")
    plt.show()
    
    return  snr_values_pycbc_welch, snr_values_fft, snr_values_scipy_welch

#Note to self, build function that takes in the waveform as an input parameter instead of generating it