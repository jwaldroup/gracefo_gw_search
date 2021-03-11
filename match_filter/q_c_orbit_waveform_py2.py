# -*- coding: utf-8 -*-
"""
Quasi-circular orbit functions adapted for python 2 order of operations syntax

version 10/22/20 4:54pm

@author: john
"""

#Imports
import numpy as np

#Chirp Mass Calculator - enter masses as solar mass multiples (floats)
def chirp_mass(m1, m2):
    sol_mass = 1.989e30
    mass1 = sol_mass*m1
    mass2 = sol_mass*m2
    
    m_chirp = ( ((mass1*mass2)**(3.0/5.0)) / ( (mass1+mass2)**(1.0/5.0) ))
    return m_chirp

#Creates basic sinusoidal waveform of non-changing frequency
def constant_f_strain_waveform(m1, m2, f_gw, duration, dt, r, theta, phi):
    
   #Unchanging parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67e-11
    c = 3.0e8
    distance = r*(3.0857e16)
    A = (4.0/distance)*((G*M_c) / (c**2.0))**(5.0/3.0)
    
    #define a array of time values
    t = np.arange(0, duration, dt)
    
    #compute retarded time
    t_ret = t - (distance/c)
    
    #compute strain amplitudes
    h_plus = A*(((np.pi*f_gw)/c)**(2.0/3.0)) * (1+(np.cos(theta))**2.0) * 0.5 * np.cos(2.0*np.pi*f_gw*t_ret+2.0*phi)
    h_cross = A*(((np.pi*f_gw)/c)**(2.0/3.0)) * np.cos(theta) * np.sin(2.0*np.pi*f_gw*t_ret+2.0*phi)
    
    print('constant frequency wave:', 'time array size:', np.size(t), 'duration (s):', duration, 'frequency (hz):', f_gw)
    
    return t, h_plus, h_cross

def strain_waveform_in_retarded_time(m1, m2, f_lower, dt, r, theta):
    
    #All equations below are from Maggiore's Gravitational Waves text - 2008
    
    #Unchanging parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67e-11
    c = 3.0e8
    distance = r*(3.0857e16)
    A = (4.0/distance)*((G*M_c) / (c**2))**(5/3)
    
    #1 - find time to coalescence using equation 4.21, given a lower frequency limit
    time_until_coalescence = 2.18*((1.21*1.989*10e30)/M_c)**(5.0/3.0) * (100.0/f_lower)**(8.0/3.0)
    print('duration (s):', time_until_coalescence, '(min):', time_until_coalescence/60)
    print('df:', 1.0/time_until_coalescence)
    
    #2 - calculate time vector, retarded time vector, and tau vector as defined by pg 170, footnote 3
    t = np.arange(0, int(time_until_coalescence), dt)
    t_ret = t - (distance/c)
    tau = int(time_until_coalescence - (distance/c)) - t_ret
    
    #3 - find varying frequency as a function of tau (defined above) from equation 4.20
    f_gw = (1.0/np.pi)*((5.0/256.0)*(1.0/tau))**(3.0/8.0) * ((G*M_c)/(c**3.0))**(-5.0/8.0)
    
    #4 - find Phi from equation 4.30
    Phi = -2.0 * ( ( (5.0*G*M_c)/(c**3.0) )**(-5.0/8.0) ) * (tau**(5.0/8.0) )
    
    #5 - calculate plus and cross polarizations from equations 4.29 
    h_plus = A*(((np.pi*f_gw)/c)**(2.0/3.0)) * (1+(np.cos(theta))**2.0) * 0.5 * np.cos(Phi)
    h_cross = A*(((np.pi*f_gw)/c)**(2.0/3.0)) * np.cos(theta) * np.sin(Phi)
    
    return t, h_plus, h_cross

def strain_waveform_observer_time(m1, m2, f_lower, dt, r, theta):
    
    #Unchanging parameters to define
    M_c = chirp_mass(m1,m2)
    G = 6.67e-11
    c = 3.0e8
    distance = r*(3.0857e16)
    A = (1.0/distance)*((G*M_c) / (c**2.0))**(5.0/4.0)
    
    #1 - find time to coalescence using equation 4.21, given a lower frequency limit
    time_until_coalescence = 2.18*((1.21*1.989*10e30)/M_c)**(5.0/3.0) * (100.0/float(f_lower))**(8.0/3.0)
    
    #2 - find tau now as a function of observer time instead of retarded time (pg 170)
    t = np.arange(0, int(time_until_coalescence), dt)
    tau = int(time_until_coalescence) - t
    
    #3 - find Phi from equation 4.30
    Phi = -2.0 * ( ( (5.0*G*M_c)/(c**3.0) )**(-5.0/8.0) ) * (tau**(5.0/8.0) )
    
    #4 - calculate plus and cross polarizations from equations 4.31 and 4.32
    h_plus = A*((5.0/(c*tau))**(1.0/4.0)) * (1+(np.cos(theta)**2.0)) * 0.5 * np.cos(Phi)
    h_cross = A*((5.0/(c*tau))**(1.0/4.0)) * (np.cos(theta)) * np.sin(Phi)
    
    #print('Strain waveform observer time parameters', 'time array size:', np.size(t), 'duration (s):', time_until_coalescence, '(min):', time_until_coalescence/60,
    #      'df:', 1.0/time_until_coalescence)
    #print(time_until_coalescence)
    
    return t, h_plus, h_cross

def obs_time_inspiral_strain(m1, m2, f_lower, dt, r, theta):
    
    #define some important quantities
    sol_mass = 1.989e30
    G = 6.67e-11
    c = 3.0e8
    distance = r*(3.0857e16)
    M_c = chirp_mass(m1,m2)
    m_tot = float( sol_mass * (m1 + m2) )
    m1_kg = sol_mass * m1
    m2_kg = sol_mass * m2
    
    #define some scaling factors for later equation simplification
    A = (1.0/distance) * (((G*M_c) / (c**2.0))**(5.0/4.0))
    B = (6.0 * (6.0 **(1.0/2.0)) * (5.0**(3.0/8.0)) ) / (256.0**(3.0/8.0))
    
    #1 - find cutoff time (ie time at which isco is reached and tau min is reached) from eqts 4.39 and 4.19
    num = B * (G**(3.0/8.0)) * (m_tot**(9.0/8.0)) 
    den = c**(9.0/8.0) * ( (m1_kg * m2_kg)**(3.0/8.0) )
    tau_min = ( num / den )**(8.0/3.0)
    
    #2 - find time to coalescence using equation 4.21, given a lower frequency limit
    time_until_coalescence = 2.18*((1.21*1.989e30)/M_c)**(5.0/3.0) * (100.0/float(f_lower))**(8.0/3.0)
    
    #3 - find tau now as a function of observer time instead of retarded time (pg 170)
    obs_t = np.arange(0, int(time_until_coalescence - tau_min), dt)
    tau = int(time_until_coalescence) - obs_t #don't subtract cutoff_time here to ensure tau does not go to zero)
    
    #4 - find Phi from equation 4.30
    Phi = -2.0 * ( ( (5.0*G*M_c)/(c**3.0) )**(-5.0/8.0) ) * (tau**(5.0/8.0) )
    
    #5 - calculate plus and cross polarizations from equations 4.31 and 4.32
    h_plus = A*((5.0/(c*tau))**(1.0/4.0)) * (1+(np.cos(theta)**2.0)) * 0.5 * np.cos(Phi)
    h_cross = A*((5.0/(c*tau))**(1.0/4.0)) * (np.cos(theta)) * np.sin(Phi)
    
    #sanity check
    #print(tau_min)
    
    return obs_t, h_plus, h_cross
