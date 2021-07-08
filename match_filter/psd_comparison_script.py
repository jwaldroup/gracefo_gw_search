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
# from pycbc import psd
# from pycbc import types

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
freqs = np.fft.fftfreq(n, d=0.1)
mask = freqs > 0
fft_vals = np.fft.fft(y)

df = freqs[1]-freqs[0]
print(df)

#calculate the fft PSD
fft_psd_1 = 2.0 * ( np.abs(fft_vals / float(n)) ) ** 2.0

fft_psd_test = 2.0 * ( np.abs(fft_vals / float(n)) ** 2.0) / (df)

#checking psd
std_y = np.std(y)
var_y = std_y**2.0
print(var_y, np.sum(fft_psd_1[mask]))
print(var_y, np.sum(fft_psd_test)/2.0)

#redo below section to compare psd's

#welch method to compare
welch_ps_freqs, welch_ps = welch_function.test_welch_2(y, 10.0, int(np.size(y)))
welch_psd_freqs, welch_psd = welch_function.test_welch(y, 10.0, int(np.size(y)))

#plt.plot(welch_ps_freqs, welch_ps, label='welch ps')
#plt.plot(freqs[mask], fft_psd_1[mask], label= 'ps via fft')

plt.plot(welch_psd_freqs, welch_psd*1.5, label='welch psd') #1.5 is the hanning window correction
plt.plot(freqs[mask], fft_psd_test[mask], label='psd via fft')
plt.xlim(0, 0.1)
plt.grid()
plt.legend()
plt.show()