# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:09:23 2020

@author: john
"""
import numpy as np

#Zero finder

#I want to iterate through an array from last element to first until I find the last zero value
#then I want it to truncate the array to include that last zero value and output it as an array

test = np.array([1.0, 5.3, 6.8, 7.9, 0.0, 3.2, 0.0, 7.2])

def last_zero_finder(test_array, abs_tol=1e-13):
    
    for i in range(np.size(test_array)):
        if np.isclose(test_array[-i], 0.0, atol=abs_tol) == True:
            
            output = test_array[0:(-i+1)] 
            break
        
        else:
            pass
        
    return output

result = last_zero_finder(test)

def first_zero_finder(test_array, abs_tol=1e-13):
    
    for i in range(np.size(test_array)):
        if np.isclose(test_array[i], 0.0, atol=abs_tol) == True:
            
            output = test_array[(i+1):] 
            break
        
        else:
            pass
    return output

def last_zero_finder_02(waveform):
    
    #Overview
    #input waveform
    #take diff of waveform which returns a array of size of waveform array - 1 element
    #take abs value of array to return another array
    #take argmax() of array to find the last inflection point as it should be the one with the greatest slope
    
    uncut = waveform.copy()
    max_loc = np.argmax( np.abs( np.diff(uncut) ) )
    truncated_waveform = uncut[0:(max_loc)]
    
    return truncated_waveform

def first_zero_finder_02(waveform):
    
    #Overview
    #input waveform
    #take second derivative of waveform which returns an array of size of the original array - 2
    #take abs value of array
    #take argmin() of array to find the first inflection point as this should correspond to the first zero
    
    uncut = waveform.copy()
    min_loc = np.argmin( np.abs( np.diff(uncut, n=2) ) )
    truncated_waveform = uncut[0:min_loc]
    
    return truncated_waveform
    



test = np.array([0,1,3,4,7,8,12,13])    

b = last_zero_finder_02(test)

x = np.linspace(15, 100, 1000)
test_2 = 2.0*x*np.sin(5.0*(2.0*np.pi/100.0)*x)

import scipy as sp
test_3 = 2.0*x*sp.signal.chirp(x, (2.0*np.pi/100.0), 100,(2.0*np.pi/100.0) )

test_2_output = first_zero_finder_02(test_2)
test_1_output = last_zero_finder_02(test_2)
test_3_output = first_zero_finder_02(test_3)


import matplotlib.pyplot as plt
# plt.plot(x, test_2, label='uncut')
# plt.plot(x[0:np.size(test_2_output)], test_2_output, label='cut')

plt.plot(x, test_3, label='uncut')
plt.legend()
plt.grid()

# plt.figure()
# plt.plot(x, test_2, label='uncut')
# plt.plot(x[0:np.size(test_1_output)], test_1_output, label='cut')
# plt.grid()
# plt.legend()

plt.show()
