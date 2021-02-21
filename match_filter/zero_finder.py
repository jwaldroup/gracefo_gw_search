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

result_2 = first_zero_finder(test)
    