# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:34:24 2021

@author: John Waldroup

GRACE-FO SNR Analysis Script
"""

"""
Script Overview

1. read in gracefo model noise data from csv
2. initialize an array of masses to test and an array of distances
3. for loop over the array of distances that for each mass:
3.1 computes waveform and inject copy into noise 
3.2 take psd of waveform and noise data
3.3 calculates the snr estimate for that selected mass at that distance iteration
3.3.1 packages the snr vs distance vector into a 2d array of mass vs distance
4. for loop through mass vs distance array and returns a new array of mass vs distance
the snr is closest to the chosen value of ten
5. plot

"""
#imports
import numpy as np
import scipy as sp
import csv
import matplotlib.pyplot as plt

from pycbc.filter import highpass
from pycbc import types
from pycbc import psd

import pycbc_welch_function as welch_function
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder

#1 - read in gracefo model noise data from csv----------------------------------------