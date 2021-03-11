# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:06:23 2020

@author: john
"""


#Takes the Welch's method of a timeseries to create its PSD frequencyseries
def pycbc_welch(ts, segnum):
    
    from pycbc import psd
    import numpy as np
    
    seg_len = int(np.size(ts) / segnum) #higher number = more segments and increasing smoothing & decrease power
    seg_stride = int(seg_len / 2) #50% overlap
    
    noise_fs = psd.welch(ts, seg_len=seg_len,
                          seg_stride=seg_stride, avg_method='mean')
    return noise_fs

def pyc_welch(ts, seg_size):
    
    from pycbc import psd
    import numpy as np
    
    seg_len = int(seg_size)
    seg_stride = int(seg_len / 2)
    
    fs = psd.welch(ts, seg_len=seg_len, seg_stride=seg_stride, avg_method='mean')
   
    return fs


def test_welch(ts, fs, seg_size):
    
    from scipy import signal
    
    nperseg = int(seg_size)
    noverlap = nperseg // 2
    freqs, psd = signal.welch(ts, fs, nperseg=nperseg, noverlap=noverlap)
    
    return freqs, psd
    
def scipy_welch(ts, fs, seg_num):
    
    import numpy as np
    from scipy import signal
    
    nperseg = int( np.size(ts) / seg_num )
    noverlap = nperseg // 2
    

    freqs, psd = signal.welch(ts, fs, nperseg=nperseg, noverlap=noverlap) #fs is sampling frequency
    
    return freqs, psd