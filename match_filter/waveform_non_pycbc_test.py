# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:04:17 2020

version 10/22/20 5:54pm

@author: john
"""
import numpy as np
import matplotlib.pyplot as plt
#import q_c_orbit_waveform_gen_functions as q_c_apx
import q_c_orbit_waveform_py2 as q_c_py2
import zero_finder

#binary system parameters
m1 = 500.0 #solar mass multiples
m2 = 500.0
f_low = 0.1
r = 1000.0 #in parsecs
dt = 0.1
theta = 0.0 

#generate waveform as seen by observer
#wf_t_array, hp, hc = q_c_apx.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)
wf_t_array, hp, hc = q_c_py2.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)
obs_t, h_plus, h_cross = q_c_py2.obs_time_inspiral_strain(m1, m2, f_low, dt, r, theta)

#t_array, hp2, hc2 = q_c_py2.strain_waveform_in_retarded_time(m1, m2, f_low, dt, r, theta)

plt.plot(wf_t_array, hp, label='waveform with no cutoff')
plt.plot(obs_t, h_plus, label='inspiral only waveform (with cutoff)')
plt.xlabel('time (s)')
plt.ylabel('amplitude strain')
plt.legend()
plt.grid()
plt.show()

#plt.figure()
#plt.plot(wf_t_array, hp, label='full waveform')
#plt.xlabel('time (s)')
#plt.ylabel('amplitude strain')
#plt.legend()
#plt.grid()
#plt.show()

#print('wf max values without and then with cutoff', np.max(hp), np.max(h_plus))

# f_gw = 0.1
# duration = int(5 * 1.0/f_gw)
# phi = 0# - 0.7479721
# t, h_plus, h_cross = q_c_py2.constant_f_strain_waveform(m1, m2, f_gw, duration, dt, r, theta, phi)

# print(phi, np.mean(h_plus))

# plt.figure()
# plt.plot(t, h_plus, label='constant frequency')
# plt.xlabel('time (s)')
# plt.ylabel('amplitude strain')
# #plt.xlim(11900,12000)
# plt.legend()
# plt.show()

# #Testing the zero finder with actual strain array---------------------------------------------------
# m1 = 500.0 #solar mass multiples
# m2 = 500.0
# f_low = 0.1
# r = 10000.0 #in parsecs
# dt = 0.1
# theta = 0.0 

# #generate waveform as seen by observer
# t_array, hp, hc = q_c_py2.strain_waveform_observer_time(m1, m2, f_low, dt, r, theta)
# print(hp, np.shape(hp))
# print(hp[-10:])

# #truncate such that graph ends by going smoothly to zero
# #hp = zero_finder.last_zero_finder(hp)
# #hp = zero_finder.first_zero_finder(hp)
# ##t_array = t_array[0:np.size(hp)]

# print(hp, np.shape(hp))

# plt.plot(hp, label='generated waveform')
# plt.xlabel('time (s)')
# plt.ylabel('amplitude strain')
# plt.legend(loc="upper left")
# plt.show()
