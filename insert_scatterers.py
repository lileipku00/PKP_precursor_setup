#!/usr/bin/env python
import numpy as np
import Scatterer_Position as sp
import hetero_structure as hs
import sys

#sys.argv[1] = name of output file
#sys.argv[2] = .csv file
#sys.argv[3] = MORB data file

vel_structure = hs.Mantle_Structure(sys.argv[1],sys.argv[2])

#interp_radius = np.linspace(3500.4,6356.6,num=4000)
#interp_theta = np.linspace(0,180,num=1800)
interp_radius = np.linspace(3480.4,6371.6,num=1000)
interp_theta = np.linspace(0,180,num=300)

vel_structure.array_interp2D(interp_radius,interp_theta)

# Open file containing locations of MORB tracers
'''
MORB = np.loadtxt(sys.argv[3])

MORB_pol = np.zeros(MORB.shape)

MORB_pol[:,0] = np.sqrt(MORB[:,0]**2+MORB[:,1]**2)
MORB_pol[:,1] = np.degrees(np.arctan2(MORB[:,0],MORB[:,1]))

vp_reference = vel_structure.vp_2D
vs_reference = vel_structure.vs_2D
rho_reference = vel_structure.rho_2D

# use add_hetero_point to add a heterogeneous point in velocity structure
# corresponding to each point in MORB_output.dat

for ii in range(0,len(MORB_pol)):
    rand_pert = abs(np.random.normal(0.001,0.0005))
    r = np.argmin(np.abs(vel_structure.radius-MORB_pol[ii,0]))
    th = np.argmin(np.abs(vel_structure.theta-MORB_pol[ii,1]))
    new_vp = vp_reference[r,th]*(1+rand_pert)
    new_vs = vs_reference[r,th]*(1+rand_pert)
    new_rho = rho_reference[r,th]*(1+0.00)

    vel_structure.add_hetero_point(MORB_pol[ii,0],MORB_pol[ii,1],new_vp,
            new_vs,new_rho,10.)
'''
#vel_structure.output_2D_sph()
