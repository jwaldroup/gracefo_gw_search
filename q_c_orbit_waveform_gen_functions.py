# -*- coding: utf-8 -*-
"""

GW Waveform Approximants for Quasi-Circular Orbits 

Created on Tue Oct 13 18:33:20 2020

@author: john
"""

#Imports
import numpy as np

#Chirp Mass Calculator - enter masses as solar mass multiples (floats)
def chirp_mass(m1, m2):
    sol_mass = 1.989*10e30
    mass1 = sol_mass*m1
    mass2 = sol_mass*m2
    
    m_chirp = ( ((mass1*mass2)**(3/5)) / ( (mass1+mass2)**(1/5) ))
    return m_chirp

#Creates basic sinusoidal waveform of non-changing frequency
def constant_f_strain_waveform(m1, m2, f_gw, duration, dt, r, theta, phi):
    
    #parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67*10e-11
    c = 3.0*10e8
    distance = r*(3.0857e16)
    A = (4.0/distance)*((G*M_c) / (c**2))**(5/3)
    
    #define a array of time values
    t = np.arange(0, duration, dt)
    
    #compute retarded time
    t_ret = t - (distance/c)
    
    #compute strain amplitudes
    h_plus = A*(((np.pi*f_gw)/c)**(2/3)) * (1+(np.cos(theta))**2) * 0.5 * np.cos(2*np.pi*f_gw*t_ret+2*phi)
    h_cross = A*(((np.pi*f_gw)/c)**(2/3)) * np.cos(theta) * np.sin(2*np.pi*f_gw*t_ret+2*phi)
    
    return t, h_plus, h_cross

def strain_waveform_in_retarded_time(m1, m2, f_lower, dt, r, theta):
    
    #All equations below are from Maggiore's Gravitational Waves text - 2008
    
    #Unchanging parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67*10e-11
    c = 3.0*1e8
    distance = r*(3.0857e16)
    A = (4.0/distance)*((G*M_c) / (c**2))**(5/3)
    
    #1 - find time to coalescence using equation 4.21, given a lower frequency limit
    time_until_coalescence = 2.18*((1.21*1.989*10e30)/M_c)**(5/3) * (100/f_lower)**(8/3)
    print('duration (s):', time_until_coalescence, '(min):', time_until_coalescence/60)
    print('df:', 1.0/time_until_coalescence)
    
    #2 - calculate time vector, retarded time vector, and tau vector as defined by pg 170, footnote 3
    t = np.arange(0, int(time_until_coalescence), dt)
    t_ret = t - (distance/c)
    tau = int(time_until_coalescence - (distance/c)) - t_ret
    
    #3 - find varying frequency as a function of tau (defined above) from equation 4.20
    f_gw = (1/np.pi)*((5/256)*(1/tau))**(3/8) * ((G*M_c)/(c**3))**(-5/8)
    
    #4 - find Phi from equation 4.30
    Phi = -2.0 * ( ( (5.0*G*M_c)/(c**3) )**(-5/8) ) * (tau**(5/8) )
    
    #5 - calculate plus and cross polarizations from equations 4.29 
    h_plus = A*(((np.pi*f_gw)/c)**(2/3)) * (1+(np.cos(theta))**2) * 0.5 * np.cos(Phi)
    h_cross = A*(((np.pi*f_gw)/c)**(2/3)) * np.cos(theta) * np.sin(Phi)
    
    return t, h_plus, h_cross

def strain_waveform_observer_time(m1, m2, f_lower, dt, r, theta):
    
    #Unchanging parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67*10e-11
    c = 3.0*1e8
    distance = r*(3.0857e16)
    A = (1.0/distance)*((G*M_c) / (c**2))**(5/4)
    
    #1 - find time to coalescence using equation 4.21, given a lower frequency limit
    time_until_coalescence = 2.18*((1.21*1.989*10e30)/M_c)**(5/3) * (100/f_lower)**(8/3)
    print('duration (s):', time_until_coalescence, '(min):', time_until_coalescence/60)
    print('df:', 1.0/time_until_coalescence)
    
    #2 - find tau now as a function of observer time instead of retarded time (pg 170)
    t = np.arange(0, int(time_until_coalescence), dt)
    tau = int(time_until_coalescence) - t
    
    #3 - find Phi from equation 4.30
    Phi = -2.0 * ( ( (5.0*G*M_c)/(c**3) )**(-5/8) ) * (tau**(5/8) )
    
    #4 - calculate plus and cross polarizations from equations 4.29
    h_plus = A*((5.0/(c*tau))**(1/4)) * (1+(np.cos(theta)**2)) * 0.5 * np.cos(Phi)
    h_cross = A*((5.0/(c*tau))**(1/4)) * (np.cos(theta)) * np.sin(Phi)
    
    return t, h_plus, h_cross
