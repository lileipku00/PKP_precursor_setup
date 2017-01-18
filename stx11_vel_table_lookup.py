import numpy as np
import seis_vel as sv
import os
import subprocess as subp
import matplotlib.pyplot as plt
import matplotlib
import Scatterer_Position as sp
import sys
import h5py
from scipy import interpolate
# matplotlib.use('PDF')

# FIRST ARUGMENT IS TEMPERATURE COORDINATES
# SECOND ARGUEMNT IS VELOCITY LOOKUP TABLE
TandC_array = np.loadtxt(sys.argv[1])

# Velocity lookup array
# P (bar) T(K) vp(km/s) vs(km/s) rho(kg/me) alpha (1/K) beta (1/bar) cp(J/K/kg) vpan, (km/s) vsan(km/s) Qp Qs
Velocity_lookup = np.loadtxt(sys.argv[2],skiprows=13)
# convert bar to pascal
Velocity_lookup[:,0] *= 100000.

# Make XY into polar coordinates with theta clockwise from northpole
Theta_array = np.degrees(np.arctan2(TandC_array[:,0],TandC_array[:,1]))
for ii in xrange(0,len(Theta_array)):
    if ii <= 0:
        Theta_array[ii] = Theta_array[ii]+360

# Determine Radius
Radius_array = np.sqrt(TandC_array[:,0]**2+TandC_array[:,1]**2)

Coord_array = np.vstack((Radius_array,Theta_array)).transpose()
Polar_Temp_array = np.round(np.hstack((Coord_array,TandC_array[:,2][:,None])),decimals=1)

# Place unique radius and theta values into array.
Radius_array = np.unique(Polar_Temp_array[:,0])
Theta_array = np.unique(Polar_Temp_array[:,1])

Temp_array = np.zeros((len(Radius_array),len(Theta_array)))

for ii in range(0,len(Polar_Temp_array)):
    rad = np.argmin(np.abs(Radius_array-Polar_Temp_array[ii,0]))
    th = np.argmin(np.abs(Theta_array-Polar_Temp_array[ii,1]))
    Temp_array[rad,th] = Polar_Temp_array[ii,2]

Radius_array = np.linspace(3390,6371,num=200)

# Iteratively find pressure conditions throughout array. Use PT conditions to find
# Density which is used for lithostatic pressure calculations. P=0 at surface.

#Allocate arrays for pressure, velocity and density. Make same shape as Temp_array
Density_array = np.zeros(Temp_array.shape)
Pressure_array = np.zeros(Temp_array.shape)
Vp_array = np.zeros(Temp_array.shape)
Vs_array = np.zeros(Temp_array.shape)

#Flip upside down to make for loop indexing more intuitive
Temp_array = np.flipud(Temp_array)
Radius_array = np.flipud(Radius_array)
Temp_array[:,-1] = (Temp_array[:,-2]+Temp_array[:,0])/2.

for ii in range(0,Temp_array.shape[1]):
    Pressure_array[0,ii] = 0
    Density_array[0,ii] = Velocity_lookup[np.where(np.logical_and(Velocity_lookup[:,0] == 0.,
                          Velocity_lookup[:,1] == Velocity_lookup[np.argmin(np.abs(Temp_array[0,ii]-Velocity_lookup[:,1])),1])),4]
    Vp_array[0,ii] = Velocity_lookup[np.where(np.logical_and(Velocity_lookup[:,0] == 0.,
                          Velocity_lookup[:,1] == Velocity_lookup[np.argmin(np.abs(Temp_array[0,ii]-Velocity_lookup[:,1])),1])),2]
    Vs_array[0,ii] = Velocity_lookup[np.where(np.logical_and(Velocity_lookup[:,0] == 0.,
                          Velocity_lookup[:,1] == Velocity_lookup[np.argmin(np.abs(Temp_array[0,ii]-Velocity_lookup[:,1])),1])),3]
#Now compute values iteratively starting on the second row of the array.
for idx,ii in enumerate(xrange(1,Temp_array.shape[0])):
#    print str(float(idx)*100/float(Temp_array.shape[0]))+'% Complete'
    for jj in range(0,Temp_array.shape[1]):
        Pressure_array[ii,jj] = Pressure_array[ii-1,jj]+Density_array[ii-1,jj]*\
                np.round((np.abs(Radius_array[ii]-Radius_array[ii-1])),decimals=3)*1000.*\
                9.81
        vel_index = np.where(np.logical_and(
                          Velocity_lookup[:,0] == Velocity_lookup[np.argmin(np.abs(Pressure_array[ii,jj]-Velocity_lookup[:,0])),0],
                          Velocity_lookup[:,1] == Velocity_lookup[np.argmin(np.abs(Temp_array[ii,jj]-Velocity_lookup[:,1])),1]))[0]
        Density_array[ii,jj] = Velocity_lookup[vel_index,4]
	Vp_array[ii,jj] = Velocity_lookup[vel_index,2]
        Vs_array[ii,jj] = Velocity_lookup[vel_index,3]

h5f = h5py.File(sys.argv[1].split('/')[0]+'_STX11.h5','w')

h5f.create_dataset('Original_Profile/Temp_Original', data = Temp_array)
h5f.create_dataset('Original_Profile/Density_Original', data = Density_array)
h5f.create_dataset('Original_Profile/Pressure_Original', data = Pressure_array)
h5f.create_dataset('Original_Profile/Vp_Original', data = Vp_array)
h5f.create_dataset('Original_Profile/Vs_Original', data = Vs_array)

h5f.close()


