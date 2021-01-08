# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 00:48:10 2021

@author: john

PSD computed by fft and via pycbc.welch estimate comparison for simple sinusodial signals

"""
import numpy as np
import matplotlib.pyplot as plt
import pycbc_welch_function as welch_function
from pycbc import psd
from pycbc import types

#td total number of samples
n = 1000

#td time length seconds
Lx = 100

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
fft_psd = 2.0 * np.abs(fft_vals / n) ** 2.0

#calculate the welch PSD
dt = float(Lx)/float(n)
y_ts = types.timeseries.TimeSeries(y, delta_t=dt)
welch_psd = welch_function.pycbc_welch(y_ts, 3)

plt.plot(freqs[mask], fft_psd[mask], label='fft')
plt.plot(welch_psd.sample_frequencies, welch_psd, label='welch')
plt.grid()
plt.legend(loc='upper right')
plt.show()
