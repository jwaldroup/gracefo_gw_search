# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:04:17 2020

@author: john
"""
import numpy as np
import q_c_orbit_waveform_gen_functions as q_c_apx
import q_c_orbit_waveform_py2 as q_c_py2

#binary system parameters
m1 = 100.0 #solar mass multiples
m2 = 100.0
f_low = 0.1
r = 1.0 #in parsecs
dt = 0.1
theta = 0.0 

#generate waveform as seen by observer
#wf_t_array, hp, hc = q_c_apx.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)
wf_t_array, hp, hc = q_c_py2.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)
