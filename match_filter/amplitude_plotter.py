# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:36:14 2020

@author: john
"""

# Waveform amplitude plotting test

import numpy as np
import matplotlib.pyplot as plt
import q_c_orbit_waveform_py2 as q_c_py2

def amplitude_plotter(mass1, mass2, f_lower_bound, dt, theta, distance_array):
    
    #This function calculates the maximum magnitude of gravitational wave strain
    #using the analytic approx strain_waveform_observer_time, packages them into a list 
    #that it returns and plots 
    
    #It inputs the same parameters as q_c_orbit_waveform_py2 and an array of 
    #distances which are in units of parsecs
    
    amplitudes = []
    
    for r in distance_array:
        
        times, plus_strain, cross_strain = q_c_py2.strain_waveform_observer_time(mass1, mass2, 
                                                                                 f_lower_bound, dt, 
                                                                                 r, theta)
        amplitudes.append(max(plus_strain))
        
        
    plt.plot(distance_array, amplitudes, label='waveform amplitudes')
    #plt.plot(distance_array, (1.0/distance_array)*(amplitudes**-1.0), label='1/r')
    plt.grid()
    plt.xlabel('distance in pc')
    plt.ylabel('strain amplitude')
    plt.legend(loc="upper right")
    plt.show()
    
    return amplitudes


#Testing the function-----------------------------------------------------------------------------------

#binary system parameters
m1 = 500.0 #solar mass multiples
m2 = 500.0
f_low = 0.1
distances = np.arange(50, 5000, 50)
dt = 0.1
theta = 0.0 

test_amp = amplitude_plotter(m1, m2, f_low, dt, theta, distances)
        

    
        