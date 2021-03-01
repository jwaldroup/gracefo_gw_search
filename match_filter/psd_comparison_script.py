# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:48:10 2021

@author: john

PSD computed by fft and via pycbc.welch estimate comparison for simple sinusodial signals

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pycbc_welch_function as welch_function
from pycbc import psd
from pycbc import types

#td total number of samples
#n = 1000
n = 10000

#td time length seconds
#Lx = 100
Lx = 1000

#angular freq of fund frequency (2pi * fundfreq)
ang = 2.0 * np.pi / Lx

#nyquist frequency
s_freq = n / Lx
nyq = s_freq / 2

##create a signal for td
#instead of dt, third input parameter is number of samples
x = np.linspace(0, Lx, n)
y1 = 1.0*np.cos(5.0*ang*x)
y2 = 2.0*np.sin(10.0*ang*x)
y3 = 0.5*np.sin(20.0*ang*x)

#net signal
y = y1+y2+y3

##Calculate frequencies, FFT, remove complex conjugates of data,
freqs = np.fft.fftfreq(n)
mask = freqs > 0
fft_vals = np.fft.fft(y)

#calculate the fft PSD
fft_psd_1 = 2.0 * ( np.abs(fft_vals / float(n) ) ) ** 2.0

#checking psd
std_y = np.std(y)
var_y = std_y**2.0
print(var_y, np.sum(fft_psd_1[mask]))

#calculate the welch PSD
dt = float(Lx)/float(n)
segnum = 80
y_ts = types.timeseries.TimeSeries(y, delta_t=dt)
welch_psd = welch_function.pycbc_welch(y_ts, segnum)

test_welch_psd = welch_function.test_py_welch(y_ts, 256)

#calulate the scipy PSD
#freqs2, psd2 = welch_function.scipy_welch(y, s_freq, 80)

#test scipy psd for finding segmentation
freqs3, psd3 = welch_function.test_welch(y, s_freq, 256)

plt.plot(freqs[mask], fft_psd_1[mask], label='fft psd 1')
#plt.plot(welch_psd.sample_frequencies, welch_psd, label='pycbc welch')
plt.plot(freqs3, psd3, label='test scipy welch')
#plt.plot(freqs2, psd2, label='scipy welch')
plt.plot(test_welch_psd.sample_frequencies, test_welch_psd, label='test py welch')
plt.xlabel('frequencies (hz)')
plt.ylabel('psd (1/hz)')
plt.grid()
plt.legend(loc='upper right')
plt.show()
