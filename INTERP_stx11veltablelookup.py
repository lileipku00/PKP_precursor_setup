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

#########################################################################################
def interpolate_mantle(Radius_len,Theta_len,h5_file):
#########################################################################################
    
    '''
    Interpolate mantle velocity profile to a desired mesh fineness. This is essential for resolving
    scattering heterogeneities on the order of tens of kilometers.
    '''
    print 'Beginning interpolate_mantle'

    h5f = h5py.File(h5_file,'r+')
    New_Radius = np.linspace(3481,6371,num=h5f['Original_Profile/Density_Original'][...].shape[0])
    New_Theta = np.linspace(0,359,num=h5f['Original_Profile/Density_Original'][...].shape[1])
    
    f_rho = interpolate.interp2d(New_Theta, 
            New_Radius, h5f['Original_Profile/Density_Original'][...])
    f_vp = interpolate.interp2d(New_Theta,
            New_Radius, h5f['Original_Profile/Vp_Original'][...])
    f_vs = interpolate.interp2d(New_Theta,
            New_Radius, h5f['Original_Profile/Vs_Original'][...])
    
    New_Radius = np.linspace(3481,6371,num=Radius_len)
    New_Theta = np.linspace(0,359,num=Theta_len)

    Theta_array, Radius_array = np.meshgrid(New_Theta,New_Radius) 
    Radius_array = np.flipud(Radius_array)

    h5f.create_dataset('Interp_Profile/Radius_Interp',  data = Radius_array,chunks=True)  
    h5f.create_dataset('Interp_Profile/Theta_Interp',   data = Theta_array,chunks=True) 
    h5f.create_dataset('Interp_Profile/Vp_Interp',      data = f_vp(New_Theta, New_Radius),chunks=True)
    h5f.create_dataset('Interp_Profile/Vs_Interp',      data = f_vs(New_Theta, New_Radius),chunks=True)
    h5f.create_dataset('Interp_Profile/Density_Interp', data = f_rho(New_Theta, New_Radius),chunks=True)
   
    Vp_Average = np.empty(h5f['Interp_Profile/Vp_Interp'].shape)
    Vs_Average = np.empty(h5f['Interp_Profile/Vs_Interp'].shape)
    Density_Average = np.empty(h5f['Interp_Profile/Density_Interp'].shape)
    for ii in xrange(0,Vp_Average.shape[0]):
        Vp_Average[ii,:] = np.mean(h5f['Interp_Profile/Vp_Interp'][ii,:])
        Vs_Average[ii,:] = np.mean(h5f['Interp_Profile/Vs_Interp'][ii,:])
        Density_Average[ii,:] = np.mean(h5f['Interp_Profile/Density_Interp'][ii,:])

    h5f.create_dataset('Average_Profile/Vp_Interp',      data = Vp_Average, chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Interp',      data = Vs_Average, chunks=True)
    h5f.create_dataset('Average_Profile/Density_Interp', data = Density_Average, chunks=True)

    #prepare_radius = np.transpose([np.reshape(Radius_array,Radius_array.size)])
    #prepare_theta = np.transpose([np.reshape(Theta_array,Radius_array.size)])
    #prepare_rho = np.transpose([np.reshape(h5f['Interp_Profile/Density_Interp'][...],Radius_array.size)])
    #prepare_vp = np.transpose([np.reshape(h5f['Interp_Profile/Vp_Interp'][...],Radius_array.size)])
    #prepare_vs = np.transpose([np.reshape(h5f['Interp_Profile/Vs_Interp'][...],Radius_array.size)])
    
    #file.write('Radius (km) theta (deg from NP) Vp (km/s) Vs (km/s) rho (kg/m^3) \n')
    #h5f.create_dataset('AxiSEM_Thermal_Input', 
    #          data = np.hstack((prepare_radius,prepare_theta,
    #          prepare_vp,prepare_vs,prepare_rho)))
    h5f.close()

#########################################################################################
def MORB_coord_bin(MORB_polar,bin_size):
#########################################################################################
    '''
    Average the radial Distribution of MORB tracers.
    MORB_polar : np.array with radius/theta values
    bin_size : radius incriment at which number of MORB tracers contained is computed
    '''
    ibins = int((MORB_polar[:,0].max()-MORB_polar[:,0].min())/bin_size)+1
    bin_contains = np.empty((ibins,2))
    uniform_coord_rad = []
    uniform_coord_theta = []
    print ibins

    for ii in xrange(0,ibins):
        bin_min = MORB_polar[:,0].min()+(ii*bin_size)
        bin_max = MORB_polar[:,0].min()+((ii+1)*bin_size)
        bin_contains[ii,0] = ii
        bin_contains[ii,1] = np.where((MORB_polar[:,0] > bin_min) & (MORB_polar[:,0] < bin_max))[0].size
        rand_theta_coord = np.random.uniform(low = 0, high = 359, size = int(bin_contains[ii,1]))
        rand_radius_coord = np.random.uniform(low = bin_min, high = bin_max, size = int(bin_contains[ii,1]))
        for jj in xrange(0,int(bin_contains[ii,1])):
            uniform_coord_rad.append(rand_radius_coord[jj])
            uniform_coord_theta.append(rand_theta_coord[jj])
    return np.array(uniform_coord_rad), np.array(uniform_coord_theta),bin_contains

#########################################################################################
def scatter_position(MORB_FILE,h5_interp_file,scale_length,vel_perturb):
#########################################################################################
    '''
    Determine location of MORB tracers and insert into velocity profile
    '''
    print 'Beginning scatter_position'
    h5f = h5py.File(h5_interp_file,'r+')
    cart_morb_array = np.around(np.loadtxt(MORB_FILE),2)
    pol_morb_array = np.empty(cart_morb_array.shape)
    pol_morb_array[:,0] = np.around(pow((cart_morb_array[:,0]**2+cart_morb_array[:,1]**2),0.5),3)
    pol_morb_array[:,1] = np.arctan2(cart_morb_array[:,0],cart_morb_array[:,1])*180./np.pi

    #Linearly stretch locations of tracers to make them fit in mantle profile
    max_rad = h5f['Interp_Profile/Radius_Interp'][:,0].max()
    min_rad = h5f['Interp_Profile/Radius_Interp'][:,0].min()
    max_morb = pol_morb_array[:,0].max()
    min_morb = pol_morb_array[:,0].min()
    slope = (max_rad-min_rad)/(max_morb-min_morb)
    y_int = min_rad-(slope*min_morb)
    
    pol_morb_array[:,0] = slope*pol_morb_array[:,0]+y_int

    for ii in xrange(0,pol_morb_array.shape[0]):
        if pol_morb_array[ii,1] < 0.:
           pol_morb_array[ii,1] = np.around((360.+pol_morb_array[ii,1]),3)

    pol_morb_array[:,1] += 180.
    for ii in xrange(0,pol_morb_array.shape[0]):
        if pol_morb_array[ii,1] >= 360.:
            pol_morb_array[ii,1] = pol_morb_array[ii,1]-360.
    h5f.create_dataset('Scatter_Profile/MORB_Polar_Coord',   data = pol_morb_array,chunks=True) 
    
    #DETERMINE AVERAGE MORB DISTRIBUTION IN POLAR COORDINATES
    avg_rad, avg_theta, bin_contains = MORB_coord_bin(h5f['Scatter_Profile/MORB_Polar_Coord'][...],20)
    avg_pol_morb_array = np.transpose(np.vstack((avg_rad,avg_theta)))

    h5f.create_dataset('Average_Profile/MORB_Radial_Bin',       data = bin_contains)
    h5f.create_dataset('Average_Profile/MORB_Average_Coord',    data = avg_pol_morb_array)
    
    #INSERT EVERY Nth MORB Tracer
    morb_insert = pol_morb_array[::4,:]
    avg_morb_insert = avg_pol_morb_array[::4,:]

    Radius = h5f['Interp_Profile/Radius_Interp'][:,0]
    Theta = h5f['Interp_Profile/Theta_Interp'][0,:]
    
    h5f.create_dataset('Scatter_Profile/Density_Scatter',data= h5f['Interp_Profile/Density_Interp'][...],chunks=True)
    h5f.create_dataset('Scatter_Profile/Vp_Scatter_1.0FAST',data= h5f['Interp_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Scatter_Profile/Vs_Scatter_1.0FAST',data= h5f['Interp_Profile/Vs_Interp'][...],chunks=True)
    h5f.create_dataset('Scatter_Profile/Vp_Scatter_0.1FAST',data= h5f['Interp_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Scatter_Profile/Vs_Scatter_0.1FAST',data= h5f['Interp_Profile/Vs_Interp'][...],chunks=True)

    print 'Constructing geodynamic scattering profiles'
    for ii in xrange(0,morb_insert.shape[0]):
        #print str(float(ii/float(morb_insert.shape[0]))*100.)
        closest_rad   = np.argmin(np.abs(morb_insert[ii,0]-Radius))
        rad_hetero    = np.argmin(np.abs((Radius[closest_rad]-scale_length)-Radius))

        closest_theta = np.argmin(np.abs(morb_insert[ii,1]-Theta))
        theta_hetero  = np.argmin(np.abs((float(scale_length)/Radius[closest_rad]*180./np.pi+Theta[closest_theta])-Theta))

        h5f['Scatter_Profile/Density_Scatter'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Interp_Profile/Density_Interp'][closest_rad,closest_theta]*0.07+
                      h5f['Interp_Profile/Density_Interp'][closest_rad,closest_theta])

        h5f['Scatter_Profile/Vp_Scatter_1.0FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Interp_Profile/Vp_Interp'][closest_rad,closest_theta]*vel_perturb+ 
                      h5f['Interp_Profile/Vp_Interp'][closest_rad,closest_theta])
        h5f['Scatter_Profile/Vs_Scatter_1.0FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Interp_Profile/Vs_Interp'][closest_rad,closest_theta]*vel_perturb +
                      h5f['Interp_Profile/Vs_Interp'][closest_rad,closest_theta])

        h5f['Scatter_Profile/Vp_Scatter_0.1FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Interp_Profile/Vp_Interp'][closest_rad,closest_theta]*0.001 +  
                      h5f['Interp_Profile/Vp_Interp'][closest_rad,closest_theta])
        h5f['Scatter_Profile/Vs_Scatter_0.1FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Interp_Profile/Vs_Interp'][closest_rad,closest_theta]*0.001 +
                      h5f['Interp_Profile/Vs_Interp'][closest_rad,closest_theta])
    
    h5f.create_dataset('Average_Profile/Density_Scatter',data= h5f['Average_Profile/Density_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vp_Scatter_1.0FAST',data= h5f['Average_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Scatter_1.0FAST',data= h5f['Average_Profile/Vs_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vp_Scatter_0.1FAST',data= h5f['Average_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Scatter_0.1FAST',data= h5f['Average_Profile/Vs_Interp'][...],chunks=True)

    h5f.create_dataset('Average_Profile/Density_Scatter_200km',data= h5f['Average_Profile/Density_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vp_Scatter_200km',data= h5f['Average_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Scatter_200km',data= h5f['Average_Profile/Vs_Interp'][...],chunks=True)

    h5f.create_dataset('Average_Profile/Density_Scatter_400km',data= h5f['Average_Profile/Density_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vp_Scatter_400km',data= h5f['Average_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Scatter_400km',data= h5f['Average_Profile/Vs_Interp'][...],chunks=True)

    h5f.create_dataset('Average_Profile/Density_Scatter_600km',data= h5f['Average_Profile/Density_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vp_Scatter_600km',data= h5f['Average_Profile/Vp_Interp'][...],chunks=True)
    h5f.create_dataset('Average_Profile/Vs_Scatter_600km',data= h5f['Average_Profile/Vs_Interp'][...],chunks=True)

    print 'Constructing averaged scattering profiles'
    for ii in xrange(0,avg_morb_insert.shape[0]):
        #print str(float(ii/float(morb_insert.shape[0]))*100.)
        closest_rad   = np.argmin(np.abs(avg_morb_insert[ii,0]-Radius))
        rad_hetero    = np.argmin(np.abs((Radius[closest_rad]-scale_length)-Radius))

        closest_theta = np.argmin(np.abs(avg_morb_insert[ii,1]-Theta))
        theta_hetero  = np.argmin(np.abs((float(scale_length)/Radius[closest_rad]*180./np.pi+Theta[closest_theta])-Theta))

        h5f['Average_Profile/Density_Scatter'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta]*float(h5f.filename.split('_')[1])/100.+
                      h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta])

        h5f['Average_Profile/Vp_Scatter_1.0FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta]*vel_perturb+ 
                      h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta])
        h5f['Average_Profile/Vs_Scatter_1.0FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta]*vel_perturb +
                      h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta])

        h5f['Average_Profile/Vp_Scatter_0.1FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta]*0.001 +  
                      h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta])
        h5f['Average_Profile/Vs_Scatter_0.1FAST'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta]*0.001 +
                      h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta])

        if avg_morb_insert[ii,0] < 3680.:
            h5f['Average_Profile/Density_Scatter_200km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta]*float(h5f.filename.split('_')[1])/100.+
                      h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vp_Scatter_200km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta]*0.01 +  
                      h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vs_Scatter_200km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta]*0.01 +
                      h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta])
        if 3680. < avg_morb_insert[ii,0] < 3880.:
            h5f['Average_Profile/Density_Scatter_400km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta]*float(h5f.filename.split('_')[1])/100.+
                      h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vp_Scatter_400km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta]*0.01 +  
                      h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vs_Scatter_400km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta]*0.01 +
                      h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta])
        if 3880. < avg_morb_insert[ii,0] < 4080.:
            h5f['Average_Profile/Density_Scatter_600km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta]*float(h5f.filename.split('_')[1])/100.+
                      h5f['Average_Profile/Density_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vp_Scatter_600km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta]*0.01 +  
                      h5f['Average_Profile/Vp_Interp'][closest_rad,closest_theta])
            h5f['Average_Profile/Vs_Scatter_600km'][closest_rad:rad_hetero,closest_theta:theta_hetero] = \
                      (h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta]*0.01 +
                      h5f['Average_Profile/Vs_Interp'][closest_rad,closest_theta])

    h5f.close()
   
#########################################################################################
def rotation_slice(h5f_file):
#########################################################################################
    print 'Beginning rotation_slice'
    h5f = h5py.File(h5f_file,'r+')
    rotation_list = [0,60,120,180,240,300]
    #EPD elements per degree of latitude
    EPD = h5f['Scatter_Profile/Density_Scatter'].shape[1]/360.
    for ii in rotation_list:
        print str(ii)+'_degree'
        h5f.create_dataset(str(ii)+'_Deg_Rotation/Thermal/Vp',data = np.roll(h5f['Interp_Profile/Vp_Interp'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/Thermal/Vs',data = np.roll(h5f['Interp_Profile/Vs_Interp'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/Thermal/Density',data = np.roll(h5f['Interp_Profile/Density_Interp'],int(EPD*ii),axis=1))

        h5f.create_dataset(str(ii)+'_Deg_Rotation/1.0FAST/Vp',data=np.roll(h5f['Scatter_Profile/Vp_Scatter_1.0FAST'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/1.0FAST/Vs',data=np.roll(h5f['Scatter_Profile/Vs_Scatter_1.0FAST'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/1.0FAST/Density',data=np.roll(h5f['Scatter_Profile/Density_Scatter'],int(EPD*ii),axis=1))

        h5f.create_dataset(str(ii)+'_Deg_Rotation/0.1FAST/Vp',data=np.roll(h5f['Scatter_Profile/Vp_Scatter_0.1FAST'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/0.1FAST/Vs',data=np.roll(h5f['Scatter_Profile/Vs_Scatter_0.1FAST'],int(EPD*ii),axis=1))
        h5f.create_dataset(str(ii)+'_Deg_Rotation/0.1FAST/Density',data=np.roll(h5f['Scatter_Profile/Density_Scatter'],int(EPD*ii),axis=1))
    h5f.close()

#########################################################################################
def output_datafiles(h5f_file):
#########################################################################################
    print 'Beginning output_datafiles'
    h5f = h5py.File(h5f_file,'r+')
    rotation_list = [0,60,120,180,240,300]

    rad  = h5f['Interp_Profile/Radius_Interp'][...]
    radshape = rad[:,0:rad.shape[1]/2].shape
    radsize = rad[:,0:rad.shape[1]/2].size
    print radshape

    theta = h5f['Interp_Profile/Theta_Interp'][...]

    rad = np.reshape(rad[:,0:radshape[1]],(rad[:,0:radshape[1]].size,1))
    theta = np.reshape(theta[:,0:radshape[1]],(rad[:,0:radshape[1]].size,1))
    Av_Density_1D = np.reshape(h5f['Average_Profile/Density_Interp'][:,0:radshape[1]],(radsize,1))
    Av_Density = np.reshape(h5f['Average_Profile/Density_Scatter'][:,0:radshape[1]],(radsize,1))
    
    # Write 1D Scatter
    Av_Vp = np.reshape(1000*h5f['Average_Profile/Vp_Interp'][:,0:radshape[1]],(radsize,1))
    Av_Vs = np.reshape(1000*h5f['Average_Profile/Vs_Interp'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_1D.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Av_Vp,Av_Vs,Av_Density_1D)),delimiter='   ',fmt='%1.3f')

    # Write 0.1 Scatter
    Av_Vp_01 = np.reshape(1000*h5f['Average_Profile/Vp_Scatter_0.1FAST'][:,0:radshape[1]],(radsize,1))
    Av_Vs_01 = np.reshape(1000*h5f['Average_Profile/Vs_Scatter_0.1FAST'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_0.1FAST.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Av_Vp_01,Av_Vs_01,Av_Density)),delimiter='   ',fmt='%1.3f')

    # Write 1.0 Scatter
    Av_Vp_10 = np.reshape(1000*h5f['Average_Profile/Vp_Scatter_1.0FAST'][:,0:radshape[1]],(radsize,1))
    Av_Vs_10 = np.reshape(1000*h5f['Average_Profile/Vs_Scatter_1.0FAST'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_1.0FAST.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Av_Vp_10,Av_Vs_10,Av_Density)),delimiter='   ',fmt='%1.3f')

    # Write 200km Scatter
    Vp_200 = np.reshape(1000*h5f['Average_Profile/Vp_Scatter_200km'][:,0:radshape[1]],(radsize,1))
    Vs_200 = np.reshape(1000*h5f['Average_Profile/Vs_Scatter_200km'][:,0:radshape[1]],(radsize,1))
    Density_200 = np.reshape(h5f['Average_Profile/Density_Scatter_200km'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_200km.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Vp_200,Vs_200,Density_200)),delimiter='   ',fmt='%1.3f')

    # Write 400km Scatter
    Vp_400 = np.reshape(1000*h5f['Average_Profile/Vp_Scatter_400km'][:,0:radshape[1]],(radsize,1))
    Vs_400 = np.reshape(1000*h5f['Average_Profile/Vs_Scatter_400km'][:,0:radshape[1]],(radsize,1))
    Density_400 = np.reshape(h5f['Average_Profile/Density_Scatter_400km'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_400km.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Vp_400,Vs_400,Density_400)),delimiter='   ',fmt='%1.3f')

    # Write 600km Scatter
    Vp_600 = np.reshape(1000*h5f['Average_Profile/Vp_Scatter_600km'][:,0:radshape[1]],(radsize,1))
    Vs_600 = np.reshape(1000*h5f['Average_Profile/Vs_Scatter_600km'][:,0:radshape[1]],(radsize,1))
    Density_600 = np.reshape(h5f['Average_Profile/Density_Scatter_600km'][:,0:radshape[1]],(radsize,1))
    with open(str(h5f.filename.split('_')[1])+'_AVG_600km.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,np.hstack((rad,theta,Vp_600,Vs_600,Density_600)),delimiter='   ',fmt='%1.3f')

    '''
    #Start the rotations
    for ii in rotation_list:
        Vp_10 = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/1.0FAST/Vp'][:,0:radshape[1]],(radsize,1))
        Vs_10 = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/1.0FAST/Vs'][:,0:radshape[1]],(radsize,1))
        Density_10 = np.reshape(h5f[str(ii)+'_Deg_Rotation/1.0FAST/Density'][:,0:radshape[1]],(radsize,1))
        with open(str(ii)+'_DEG_1.0FAST.sph','w') as f_handle:
            f_handle.write(str(rad.size)+'\n')
            np.savetxt(f_handle,np.hstack((rad,theta,Vp_10,Vs_10,Density_10)),delimiter='   ',fmt='%1.3f')

        Vp_01 = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/0.1FAST/Vp'][:,0:radshape[1]],(radsize,1))
        Vs_01 = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/0.1FAST/Vs'][:,0:radshape[1]],(radsize,1))
        #Density_01 = np.reshape(h5f[str(ii)+'_Deg_Rotation/0.1FAST/Density'][:,0:radshape[1]],(radsize,1))
        with open(str(ii)+'_DEG_0.1FAST.sph','w') as f_handle:
            f_handle.write(str(rad.size)+'\n')
            np.savetxt(f_handle,np.hstack((rad,theta,Vp_01,Vs_01,Density_10)),delimiter='   ',fmt='%1.3f')

        Vp_therm = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/Thermal/Vp'][:,0:radshape[1]],(radsize,1))
        Vs_therm = np.reshape(1000*h5f[str(ii)+'_Deg_Rotation/Thermal/Vs'][:,0:radshape[1]],(radsize,1))
        Density_therm = np.reshape(h5f[str(ii)+'_Deg_Rotation/Thermal/Density'][:,0:radshape[1]],(radsize,1))
        with open(str(ii)+'_DEG_THERMAL.sph','w') as f_handle:
            f_handle.write(str(rad.size)+'\n')
            np.savetxt(f_handle,np.hstack((rad,theta,Vp_therm,Vs_therm,Density_therm)),delimiter='   ',fmt='%1.3f')
      '''
interpolate_mantle(1000,20000,'EED_000_STX11.h5')
scatter_position('EED_000/MORB.dat','EED_000_STX11.h5',16.,0.01)
rotation_slice('EED_000_STX11.h5')
output_datafiles('EED_000_STX11.h5')



