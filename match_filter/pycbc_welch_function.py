# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:06:23 2020

@author: john
"""


#Takes the Welch's method of a timeseries to create its PSD frequencyseries
def pycbc_welch(ts, segnum):
    
    from pycbc import psd
    
    seg_len = int(ts.duration) // segnum #higher number = more segments and increasing smoothing & decrease power
    seg_stride = seg_len // 2 #50% overlap
    
    noise_fs = psd.welch(ts, seg_len=seg_len,
                         seg_stride=seg_stride)
    return noise_fs