# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 22:09:23 2020

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt

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

def last_zero_finder_03(waveform, component_mass, dt):
    
    #this one isn't working
    
    
    uncut = waveform.copy()
    
    #define constants and calculate chirp mass
    sol_mass = 1.989e30
    G = 6.67e-11
    c = 3.0e8 
    mass1 = sol_mass*component_mass
    m_chirp = ( ((mass1*mass1)**(3.0/5.0)) / ( (mass1+mass1)**(1.0/5.0) ))
    
    #calculate f_isco
    f_isco = ( 1.0/(6.0*(6.0**(1.0/2.0))*np.pi) ) * ( (c**3.0) / (G * 2.0 * mass1 ) )
    
    #calculate f_min of bandwidth in which one full cycle of the gravitational wave strain occurs
    f_min = (f_isco**(-5.0/3.0) + ( 32.0*(np.pi**(8.0/3.0)) ) * ( ((c**3.0) / (m_chirp * G))**(-5.0/3.0) ) )**(-3.0/5.0)
    
    #calculate wave period of cycle
    period = 1.0 / f_min
    
    #caluclate cutoff index and truncate waveform
    wavelength_index = int(period / dt )
    print(wavelength_index)
    cut_index = np.argmin( np.abs(uncut[(-wavelength_index):] ) )
    print(cut_index)
    truncated_waveform = uncut[0:(-wavelength_index+cut_index)]
    return truncated_waveform

def first_zero_finder_02(waveform, component_mass, f_low, dt):
    
    uncut = waveform.copy()
    
    #define constants and calculate chirp mass
    sol_mass = 1.989e30
    G = 6.67e-11
    c = 3.0e8 
    mass1 = sol_mass*component_mass
    m_chirp = ( ((mass1*mass1)**(3.0/5.0)) / ( (mass1+mass1)**(1.0/5.0) ))
    
    #calculate f_max of bandwidth in which one full cycle of the gravitational wave strain occurs
    f_max = (f_low**(-5.0/3.0) - ( 32.0*(np.pi**(8.0/3.0)) ) * ( ((c**3.0) / (m_chirp * G))**(-5.0/3.0) ))**(-3.0/5.0)
    
    #calculate wave period of the cycle
    period = 1.0 / f_max
    
    #find cutoff index and truncate waveform
    wavelength_end_index =  int(period / dt)
    min_index = np.argmin(np.abs(uncut[0:wavelength_end_index]))
    truncated_waveform = uncut[min_index:]
    
    return truncated_waveform
    
def first_zero_finder_03(waveform):
    
    #computes by finding index of waveform vector where the abs val of the second
    #derivative is minimal (ie closest to zero and therefore at an inflection point which is close to a root)
    uncut = waveform.copy()
    min_loc = np.argmin( np.abs( np.diff(uncut, n=2) ) )
    truncated_waveform = uncut[min_loc:]
    
    return truncated_waveform



# test = np.array([0,1,3,4,7,8,12,13])    

# b = last_zero_finder_02(test)

# x = np.linspace(15, 185, 1000)
# test_2 = 2.0*x*np.sin(5.0*(2.0*np.pi/100.0)*x)

# from scipy import signal
# test_3 = 2.0*x*signal.chirp(x, (2.0*np.pi/100.0), 100,(2.0*np.pi/100.0) )

# test_2_output = first_zero_finder_02(test_3, (2.0*np.pi/100.0), 170.0/1000.0)
# test_1_output = last_zero_finder_02(test_2)
# #test_3_output = first_zero_finder_02(test_3)
# test_4_op = last_zero_finder_02(test_3)

# import matplotlib.pyplot as plt
# plt.plot(x, test_3, label='uncut')
# plt.plot(x[0:np.size(test_2_output)], test_2_output, label='cut')
# plt.grid()
# plt.legend()


# plt.figure()
# plt.plot(x, test_3, label='uncut')
# plt.plot(x[0:np.size(test_4_op)], test_4_op, label='cut')
# plt.legend()
# plt.grid()

# #plt.plot(x, test_2, label='uncut')
# #plt.plot(x[0:np.size(test_1_output)], test_1_output, label='cut')

# plt.show()

#Testing on actual generated waveforms

import q_c_orbit_waveform_py2 as q_c_py2

m1 = 1000
m2 = m1
f_lower = 0.1
dt = 0.1
r = 1000.0
theta = 0

obs_t, hp, hc = q_c_py2.obs_time_inspiral_strain(m1, m2, f_lower, dt, r, theta)

hp_cut = last_zero_finder_02(hp)
#hp_cut = last_zero_finder_03(hp, m1, dt)
obs_t_cut = obs_t[0:np.size(hp_cut)]

#hp_cut = first_zero_finder_03(hp_cut)
hp_cut = first_zero_finder_02(hp_cut, m1, f_lower, dt)
obs_t_cut = obs_t_cut[-np.size(hp_cut):]

plt.plot(obs_t, hp, label='uncut')
plt.plot(obs_t_cut, hp_cut, label='cut')
plt.grid()
plt.legend(loc='upper left')
plt.show()