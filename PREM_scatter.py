import h5py
import numpy as np
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt
import h5py

#############################################################################################
def PREM_interp():
#############################################################################################
    '''

    '''
    PREM = np.loadtxt('PREM_1s.csv',delimiter=',')

    prem_radius = PREM[:,0]
    rho_1D =  PREM[:,2]
    vs_1D  =  PREM[:,5]
    vp_1D  =  PREM[:,3]

    #Interpolate top to smooth out
    rad_top = np.linspace(PREM[0,0],PREM[27,0],num=100)
    rho_top = np.linspace(PREM[0,2],PREM[27,2],num=100)
    vs_top = np.linspace(PREM[0,5],PREM[27,5],num=100)
    vp_top = np.linspace(PREM[0,3],PREM[27,3],num=100)

    prem_radius = np.hstack((rad_top,prem_radius[27:]))
    rho_1D = np.hstack((rho_top,rho_1D[27:]))
    vs_1D = np.hstack((vs_top,vs_1D[27:]))
    vp_1D = np.hstack((vp_top,vp_1D[27:]))

    f_rho = scipy.interpolate.interp1d(prem_radius,rho_1D)
    f_vp = scipy.interpolate.interp1d(prem_radius,vp_1D)
    f_vs = scipy.interpolate.interp1d(prem_radius,vs_1D)

    interp_rad = np.linspace(prem_radius.max(),prem_radius.min(),num = 1000)
    interp_theta = np.linspace(0,179,num=10000)

    interp_rho = f_rho(interp_rad)
    interp_vp = f_vp(interp_rad)
    interp_vs = f_vs(interp_rad)

    x,rho_2D = np.meshgrid(interp_theta,interp_rho)
    x,vp_2D = np.meshgrid(interp_theta,interp_vp)
    x,vs_2D = np.meshgrid(interp_theta,interp_vs)

    #remove crust
    rho_2D[:10,:]=rho_2D[10,:]
    vp_2D[:10,:]=vp_2D[10,:]
    vs_2D[:10,:]=vs_2D[10,:]

    return rho_2D,vp_2D,vs_2D

#############################################################################################
def MORB_profile(min_morb,bin_num):
#############################################################################################

    rad_bin = np.linspace(3481,6371,num=bin_num)
    max_morb = rad_bin.max()*min_morb/rad_bin.min()
    morb_bin = np.linspace(min_morb,max_morb,num=rad_bin.size)

    return rad_bin,morb_bin

#############################################################################################
def single_scatter_coord(rad,number,shift):
#############################################################################################
    '''
    Use only to make a single layer of scatterers at certain depth
    shift in kilomters
    '''
    degshift = np.degrees(shift/float(rad))
    theta_array = np.linspace(90+degshift,185+degshift,num=number)
    rad_array = np.linspace(1,1,num=number)*rad

    return np.hstack((np.transpose([rad_array]),np.transpose([theta_array])))

#############################################################################################
def scatter_coord(rad_min,rad_max,rho_2D):
#############################################################################################

    rad_bin, morb_bin = MORB_profile(90,100)

    bin_min = np.argmin(np.abs(rad_min-rad_bin))
    bin_max = np.argmin(np.abs(rad_max-rad_bin))

    rad_list = []
    theta_list = []

    for idx, ii in enumerate(xrange(bin_min,bin_max)):
        rad_min = rad_bin[ii]
        rad_max = rad_bin[ii+1]
        rad_values = (rad_max-rad_min)*np.random.random(int(morb_bin[idx]))+rad_min
        theta_values = 179*np.random.random(morb_bin[idx])
        rad_list.append(rad_values)
        theta_list.append(theta_values)

        rad_array = np.concatenate(np.array(rad_list))
        theta_array = np.concatenate(np.array(theta_list))

    return np.hstack((np.transpose([rad_array]),np.transpose([theta_array])))

#############################################################################################
def add_scatter(scatter_array,scale_length,rho_2D,vp_2D,vs_2D):
#############################################################################################

    rho_copy = np.copy(np.flipud(rho_2D))
    vp_copy = np.copy(np.flipud(vp_2D))
    vs_copy = np.copy(np.flipud(vs_2D))

    rho_orig = np.copy(np.flipud(rho_2D))
    vp_orig = np.copy(np.flipud(vp_2D))
    vs_orig = np.copy(np.flipud(vs_2D))

    Radius = np.linspace(3481,6371,num=rho_2D.shape[0])
    Theta = np.linspace(0,179,num=rho_2D.shape[1])

    thetamesh,radmesh = np.meshgrid(Theta,Radius)

    radius_values = scatter_array[:,0]
    theta_values = scatter_array[:,1]

    for idx, ii in enumerate(radius_values):
        closest_rad  = np.argmin(np.abs(ii-Radius))
        rad_hetero   = np.argmin(np.abs((Radius[closest_rad]+scale_length)-Radius))

        closest_theta = np.argmin(np.abs(theta_values[idx]-Theta))
        theta_hetero  = np.argmin(np.abs(((np.degrees(scale_length/float(Radius[closest_rad]))+Theta[closest_theta])-Theta)))

        rho_copy[closest_rad:rad_hetero,closest_theta:theta_hetero] = rho_orig[closest_rad,closest_theta]*(1+0.03)
        vp_copy[closest_rad:rad_hetero,closest_theta:theta_hetero] = vp_orig[closest_rad,closest_theta]*(1+0.001)
        vs_copy[closest_rad:rad_hetero,closest_theta:theta_hetero] = vs_orig[closest_rad,closest_theta]*(1+0.001)

    return np.flipud(rho_copy), np.flipud(vp_copy), np.flipud(vs_copy),thetamesh,np.flipud(radmesh)

#########################################################################################
def write_h5(name,rho,vp,vs,thetamesh,radmesh,array):
#########################################################################################
    '''
    Write data set to h5py file.

    name: str. Name of subgroup that will be created
    '''
    f = h5py.File(name+'.h5','w')
    f.create_dataset('/rho',data = rho)
    f.create_dataset('/vp',data = vp)
    f.create_dataset('/vs',data = vs)
    f.create_dataset('/radmesh',data = radmesh)
    f.create_dataset('/thetamesh',data = thetamesh)
    f.create_dataset('/scatter',data = array)
    f.close()

#########################################################################################
def output_datafiles(h5f_file,name):
#########################################################################################
    '''
    write files for AxiSEM to read.

    h5f_file: name of h5py file.
    name : name of subgroup of h5py.file
    '''

    print 'Beginning output_datafiles'

    h5f = h5py.File(h5f_file,'r')

    rad = h5f['/radmesh'][...]
    theta = h5f['/thetamesh'][...]

    rad = np.reshape(rad,(rad.size,1))
    theta = np.reshape(theta,(rad.size,1))
    rho = np.reshape(h5f['/rho'][...]*1000.,(rad.size,1))
    vp = np.reshape(h5f['/vp'][...]*1000.,(rad.size,1))
    vs = np.reshape(h5f['/vs'][...]*1000.,(rad.size,1))
    h5f.close()

    data = np.hstack((rad,theta,vp,vs,rho))
    data = data[np.logical_not(data[:,1] < 90)]
    data = data[data[:,0] < 5873]
    with open('PREM_'+name+'.sph','w') as f_handle:
        f_handle.write(str(rad.size)+'\n')
        np.savetxt(f_handle,data,delimiter='   ',fmt='%1.3f')


radmin = 3481
radmax = 3481

rad = 3481
num = 75
shift = 50


layer_shift = [[3481,60,0],
               [3681,63,10],
               [3881,66,20],
               [4081,70,30],
               [4281,73,40],
               [4481,77,50],
               [4681,81,60]]

rho,vp,vs = PREM_interp()
#scatter_array = scatter_coord(radmin,radmax,rho)
#scatter_array = single_scatter_coord(rad,num,shift)

'''
full_scatter_array1 = np.zeros((1,2))
full_scatter_array2 = np.zeros((1,2))
full_scatter_array3 = np.zeros((1,2))
full_scatter_array = np.zeros((1,2))

for idx,ii in enumerate(layer_shift):
    layer_array1 = single_scatter_coord(ii[0],ii[1]+int(full_list[0][idx]),ii[2])
    layer_array2 = single_scatter_coord(ii[0],ii[1]+int(full_list[1][idx]),ii[2])
    layer_array3 = single_scatter_coord(ii[0],ii[1]+int(full_list[2][idx]),ii[2])
    layer_array = single_scatter_coord(ii[0],ii[1],ii[2])
    full_scatter_array1 = np.vstack((full_scatter_array1,layer_array1))
    full_scatter_array2 = np.vstack((full_scatter_array2,layer_array2))
    full_scatter_array3 = np.vstack((full_scatter_array3,layer_array3))
    full_scatter_array = np.vstack((full_scatter_array,layer_array))

full_scatter_array1 = full_scatter_array1[1::,:]
full_scatter_array2 = full_scatter_array2[1::,:]
full_scatter_array3 = full_scatter_array3[1::,:]
full_scatter_array = full_scatter_array[1::,:]
'''

'''
r,p,s,thetamesh,radmesh = add_scatter(full_scatter_array,20,rho,vp,vs)
write_h5('full_baseline',r,p,s,thetamesh,radmesh)

r,p,s,thetamesh,radmesh = add_scatter(full_scatter_array1,20,rho,vp,vs)
write_h5('EED_000',r,p,s,thetamesh,radmesh)

r,p,s,thetamesh,radmesh = add_scatter(full_scatter_array2,20,rho,vp,vs)
write_h5('EED_007',r,p,s,thetamesh,radmesh)

r,p,s,thetamesh,radmesh = add_scatter(full_scatter_array3,20,rho,vp,vs)
write_h5('EED_009',r,p,s,thetamesh,radmesh)
'''

def rotate_brandenburg(file,rotation,skip):
    EED_000 = np.loadtxt(file)
    degree_rotation = rotation
    x = EED_000[:,0]
    y = EED_000[:,1]
    theta = np.transpose([np.degrees(np.arctan2(x,y))])[::skip]
    theta += degree_rotation
    radius = np.transpose([np.sqrt(x**2+y**2)])[::skip]
    array = []
    for ii in range(0,len(theta)):
        if theta[ii][0] < 90. or theta[ii][0] > 180 or theta[ii][0] < 0 or radius[ii][0] > 5711.:
            continue
        else:
            array.append([radius[ii][0],theta[ii][0]])
    array = np.array(array)
    return array


array = rotate_brandenburg('EED_000/MORB.dat',46,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_000_46',r,p,s,thetamesh,radmesh,array)

array = rotate_brandenburg('EED_000/MORB.dat',126,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_000_126',r,p,s,thetamesh,radmesh,array)

array = rotate_brandenburg('EED_007/MORB.dat',156,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_007_156',r,p,s,thetamesh,radmesh,array)

array = rotate_brandenburg('EED_007/MORB.dat',46,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_007_46',r,p,s,thetamesh,radmesh,array)

array = rotate_brandenburg('EED_009/MORB.dat',176,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_009_176',r,p,s,thetamesh,radmesh,array)

array = rotate_brandenburg('EED_009/MORB.dat',146,3)
r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
write_h5('EED_009_146',r,p,s,thetamesh,radmesh,array)

#array = rotate_brandenburg('EED_007/MORB.dat',120,3)
#r,p,s,thetamesh,radmesh = add_scatter(array,20,rho,vp,vs)
#write_h5('EED_007_120',r,p,s,thetamesh,radmesh,array)

#output_datafiles('EED_007_40.h5','EED_007_40')
output_datafiles('EED_000_46.h5','EED_000_46')
output_datafiles('EED_000_126.h5','EED_000_126')
output_datafiles('EED_007_156.h5','EED_007_156')
output_datafiles('EED_007_46.h5','EED_007_46')
output_datafiles('EED_009_176.h5','EED_009_176')
output_datafiles('EED_009_146.h5','EED_009_146')
#output_datafiles('EED_007_43.h5','EED_007_43')

#write_h5('Single_'+str(rad)+'_'+str(num),r,p,s,thetamesh,radmesh)
#write_h5('Single_'+str(rad)+'_'+str(num),r,p,s,thetamesh,radmesh)
#write_h5('full_scatter',r,p,s,thetamesh,radmesh)
#output_datafiles(str(radmin)+'_'+str(radmax)+'.h5','Single_'+str(rad)+'_'+str(num)+'shift_'+str(shift))

'''
output_datafiles('EED_000.h5','000_full_scatter')
output_datafiles('EED_007.h5','007_full_scatter')
output_datafiles('EED_009_170.h5','009_full_scatter')
output_datafiles('full_baseline.h5','full_baseline')
'''

