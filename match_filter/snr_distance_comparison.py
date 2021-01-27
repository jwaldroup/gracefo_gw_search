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

def snr_distance_plotter_wf_self_generating(mass1, mass2, f_lower_bound, dt, theta, distance_array, noise_psd, lgd_label='legend key'):
    
    snr_values = []
    
    for r in distance_array:
        times, plus_strain, cross_strain = q_c_py2.strain_waveform_observer_time(mass1, mass2, 
                                                                                 f_lower_bound, dt, 
                                                                                 r, theta)
        
        h_ts = types.timeseries.TimeSeries(plus_strain, dt)
        
        #find psd via welch
        # h_fs = welch_function.pycbc_welch(h_ts, 1)
        # h_fs = psd.interpolate(h_fs, noise_psd.delta_f)
        
        #find psd via fft
        freqs = np.fft.fftfreq(np.size(h_ts))
        mask = freqs > 0
        raw_fft_h_ts = np.fft.fft(h_ts)
        h_fs = (2.0 * np.abs(raw_fft_h_ts / float(np.size(h_ts)) ) ) ** 2.0
        h_fs = types.frequencyseries.FrequencySeries(h_fs[mask], delta_f=(1.0/h_ts.duration))
        #h_fs = psd.interpolate(h_fs, noise_psd.delta_f)
        noise_psd = psd.interpolate(noise_psd, h_fs.delta_f)
        
        #first way 
        # signal_psd = np.abs(h_fs)
        # psd_ratio = (4.0 * signal_psd * noise_psd.delta_f) / noise_psd
        # snr_squared = psd_ratio.sum()
        
        # snr_estimate = np.sqrt(snr_squared)
        # snr_values.append(snr_estimate)
        
        #second way
        signal_psd = h_fs
        #psd_ratio = (4.0 * signal_psd* noise_psd.delta_f) / (noise_psd[:-1])
        psd_ratio = (4.0 * signal_psd) / noise_psd[:-2] #/(noise_psd[:-2])
        snr_squared = (psd_ratio).sum()
        #snr_squared = (psd_ratio * noise_psd.delta_f).sum()
        snr_estimate = np.sqrt(snr_squared)
        snr_values.append(snr_estimate)
        
    plt.loglog(distance_array, snr_values, label=lgd_label)
    plt.grid()
    plt.xlabel('distance in pc')
    plt.ylabel('snr')
    plt.legend(loc="upper right")
    plt.show()
    
    return snr_values

#Note to self, build function that takes in the waveform as an input parameter instead of generating it