import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys
import matplotlib as mpl
from scipy.interpolate import interp1d

#############################################################################################
def main(infile):
#############################################################################################
    MORB_cart = np.loadtxt(infile)
    MORB_polar = polar_convert(MORB_cart)
    bin_array = bin_morb(MORB_polar,20) 
    return bin_array #return rad, theta 
#############################################################################################
def polar_convert(MORB_cart):
#############################################################################################
    '''
    MORB_cart :np. array of first column is X coordinate, second is Y
    Returns : np.array of radius/theta values
    '''
    MORB_polar = np.empty(MORB_cart.shape)
    MORB_polar[:,0] = np.sqrt(MORB_cart[:,0]**2+MORB_cart[:,1]**2)
    for ii in xrange(0,MORB_cart.shape[0]):
        deg = np.degrees(np.arctan2(MORB_cart[ii,0],MORB_cart[ii,1]))
        if deg < 0.:
            MORB_polar[ii,1] = deg+360
        else:
            MORB_polar[ii,1] = deg

    return MORB_polar

#############################################################################################
def bin_morb(MORB_polar,bin_size):
#############################################################################################
    bins = np.linspace(3481,6371,num=int(2891./bin_size))
    reduce_morb = MORB_polar[::1,0]
    bin_array = np.zeros((len(bins[:-1]),2))
    for idx, ii in enumerate(bins[:-1]):
        count = 0
        for jj in reduce_morb:
            if jj > bins[idx] and jj < bins[idx+1]:
                count += 1
        bin_array[idx,0] = ii
        bin_array[idx,1] = count
    return bin_array


#############################################################################################
def PREM_interp():
#############################################################################################
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

    f_rho = interp1d(prem_radius,rho_1D)
    f_vp = interp1d(prem_radius,vp_1D)
    f_vs = interp1d(prem_radius,vs_1D)

    interp_rad = np.linspace(prem_radius.max(),prem_radius.min(),num = 1000)
    interp_theta = np.linspace(90,179,num=5000)

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
def add_scatter(scatter_array,scale_length,rho_2D,vp_2D,vs_2D):
#############################################################################################

    rho_copy = np.copy(np.flipud(rho_2D))
    vp_copy = np.copy(np.flipud(vp_2D))
    vs_copy = np.copy(np.flipud(vs_2D))

    rho_orig = np.copy(np.flipud(rho_2D))
    vp_orig = np.copy(np.flipud(vp_2D))
    vs_orig = np.copy(np.flipud(vs_2D))

    Radius = np.linspace(3481,6371,num=rho_2D.shape[0])
    Theta = np.linspace(90,179,num=rho_2D.shape[1])

    thetamesh,radmesh = np.meshgrid(Theta,Radius)

    radius_values = scatter_array[:,0]
    theta_values = scatter_array[:,1]

    for idx, ii in enumerate(radius_values):
        closest_rad  = np.argmin(np.abs(ii-Radius))
        rad_hetero   = np.argmin(np.abs((Radius[closest_rad]+scale_length)-Radius))
        closest_theta = np.argmin(np.abs(theta_values[idx]-Theta))
        theta_hetero  = np.argmin(np.abs(((np.degrees(scale_length
                        /float(Radius[closest_rad]))+Theta[closest_theta])-Theta)))
        rho_copy[closest_rad:rad_hetero,closest_theta:theta_hetero] = rho_orig[closest_rad,closest_theta]*(1+0.03)
        vp_copy[closest_rad:rad_hetero,closest_theta:theta_hetero] = vp_orig[closest_rad,closest_theta]*(1+0.001)*np.nan
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
    with open(name+'.sph','w') as f_handle:
        f_handle.write(str(data.shape[0])+'\n')
        np.savetxt(f_handle,data,delimiter='   ',fmt='%1.3f')

'''
NOW STARTS THE WORK
'''

rho_2D, vp_2D, vs_2D = PREM_interp()

#distance_list = [3481,3681,3881,4081,4281,4481]
#morb_list = ['EED_000/EED_000_MORB_output_0DEG.dat',
#             'EED_007/EED_007_0DEG_MORB_output.dat',
#             'EED_009/EED_009_0DEG_MORB_output.dat']

# THIS IS FOR TOTAL PERCENT COMPOSITION
'''
array_9 = main('EED_009/EED_009_0DEG_MORB_output.dat')
array_7 = main('EED_007/EED_007_0DEG_MORB_output.dat')
array_0 = main('EED_000/EED_000_MORB_output_0DEG.dat')

new_rad_9 = np.linspace(array_9[:,0].min(),array_9[:,0].max(),num=28)
new_rad_7 = np.linspace(array_7[:,0].min(),array_7[:,0].max(),num=28)
new_rad_0 = np.linspace(array_0[:,0].min(),array_0[:,0].max(),num=28)

f_9 = interp1d(array_9[:,0],array_9[:,1])
f_7 = interp1d(array_7[:,0],array_7[:,1])
f_0 = interp1d(array_0[:,0],array_0[:,1])

new_scat_9 = f_9(new_rad_9)/f_9(new_rad_9).sum()*30000
new_scat_7 = f_7(new_rad_7)/f_7(new_rad_7).sum()*30000
new_scat_0 = f_0(new_rad_0)/f_0(new_rad_0).sum()*30000

new_rad_9 = new_rad_9[0:-6]
new_rad_7 = new_rad_7[0:-6]
new_rad_0 = new_rad_0[0:-6]
new_scat_9 = new_scat_9[0:-6]
new_scat_7 = new_scat_7[0:-6]
new_scat_0 = new_scat_0[0:-6]
'''

# THIS IS FOR ACTUAL RUNS
f_000_120 = h5py.File('EED_000_120.h5','r')
f_000_40 = h5py.File('EED_000_40.h5','r')
f_007_150 = h5py.File('EED_007_150.h5','r')
f_007_40 = h5py.File('EED_007_40.h5','r')

f_uniform_000 = h5py.File('uniform_EED_000_1.h5','r')
f_uniform_007 = h5py.File('uniform_EED_007_1.h5','r')
scat_000 = f_uniform_000['scatter'][...]
scat_007 = f_uniform_007['scatter'][...]

scat_000_120 = f_000_120['scatter'][...]
scat_000_40 = f_000_40['scatter'][...]
scat_007_150 = f_007_150['scatter'][...]
scat_007_40 = f_007_40['scatter'][...]

scale_length=20

r,p,s,thetamesh,radmesh = add_scatter(scat_007,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_007',r,p,s,thetamesh,radmesh,scat_007)

r,p,s,thetamesh,radmesh = add_scatter(scat_000,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_000',r,p,s,thetamesh,radmesh,scat_000)

r,p,s,thetamesh,radmesh = add_scatter(scat_000_120,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_000_120',r,p,s,thetamesh,radmesh,scat_000_120)

r,p,s,thetamesh,radmesh = add_scatter(scat_000_40,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_000_40',r,p,s,thetamesh,radmesh,scat_000_40)

r,p,s,thetamesh,radmesh = add_scatter(scat_007_150,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_007_150',r,p,s,thetamesh,radmesh,scat_007_150)

r,p,s,thetamesh,radmesh = add_scatter(scat_007_40,scale_length,
                                      rho_2D,vp_2D,vs_2D)
write_h5('scat_007_40',r,p,s,thetamesh,radmesh,scat_007_40)

def make_scatter_array(new_scat_0,new_scat_7,new_scat_9,**kwargs):
    kwargs.get('name1','name1')
    kwargs.get('name2','name2')
    kwargs.get('name3','name3')

    scat_9_average = []
    scat_7_average = []
    scat_0_average = []

    for idx, ii in enumerate(new_scat_9):
        radius = new_rad_9[idx]
        for jj in range(int(ii)):
            #rad_offset = np.random.uniform(0,100)
            rad_offset = np.random.normal(50,50)
            theta = np.random.uniform(90,180)
            scat_9_average.append([radius+rad_offset,theta])
        for jj in range(int(new_scat_7[idx])):
            #rad_offset = np.random.uniform(0,100)
            rad_offset = np.random.normal(50,50)
            theta = np.random.uniform(90,180)
            scat_7_average.append([radius+rad_offset,theta])
        for jj in range(int(new_scat_0[idx])):
            #rad_offset = np.random.uniform(0,100)
            rad_offset = np.random.normal(50,50)
            theta = np.random.uniform(90,180)
            scat_0_average.append([radius+rad_offset,theta])

    scat_9_array = np.array(scat_9_average)
    scat_7_array = np.array(scat_7_average)
    scat_0_array = np.array(scat_0_average)


#make_scatter_array(scat_000_120,scat_000_40,scat_007_150,name1='000_120_nan',
#                   name2='000_40)nan',name3='007_150_nan')

'''
scale_length=20

scat_0_array, scat_7_array, scat_9_array = make_scatter_array(new_scat_9,new_scat_7,new_scat_0)
r,p,s,thetamesh,radmesh = add_scatter(scat_0_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('EED_000_count_3',r,p,s,thetamesh,radmesh,scat_9_array)
r,p,s,thetamesh,radmesh = add_scatter(scat_7_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('EED_007_count_3',r,p,s,thetamesh,radmesh,scat_9_array)


####### Case 1
scat_0_array, scat_7_array, scat_9_array = make_scatter_array(new_scat_9,new_scat_7,new_scat_0)

r,p,s,thetamesh,radmesh = add_scatter(scat_9_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_009_10km_1',r,p,s,thetamesh,radmesh,scat_9_array)
output_datafiles('uniform_EED_009_10km_1.h5','uniform_EED_009_10km_0.1_1')

r,p,s,thetamesh,radmesh = add_scatter(scat_7_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_007_10km_1',r,p,s,thetamesh,radmesh,scat_7_array)
output_datafiles('uniform_EED_007_10km_1.h5','uniform_EED_007_10km_0.1_1')

r,p,s,thetamesh,radmesh = add_scatter(scat_0_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_000_10km_1',r,p,s,thetamesh,radmesh,scat_0_array)
output_datafiles('uniform_EED_000_10km_1.h5','uniform_EED_000_10km_0.1_1')

####### Case 2
scat_0_array, scat_7_array, scat_9_array = make_scatter_array(new_scat_9,new_scat_7,new_scat_0)

r,p,s,thetamesh,radmesh = add_scatter(scat_9_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_009_10km_2',r,p,s,thetamesh,radmesh,scat_9_array)
output_datafiles('uniform_EED_009_10km_2.h5','uniform_EED_009_10km_0.1_2')

r,p,s,thetamesh,radmesh = add_scatter(scat_7_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_007_10km_2',r,p,s,thetamesh,radmesh,scat_7_array)
output_datafiles('uniform_EED_007_10km_2.h5','uniform_EED_007_10km_0.1_2')

r,p,s,thetamesh,radmesh = add_scatter(scat_0_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_000_10km_2',r,p,s,thetamesh,radmesh,scat_0_array)
output_datafiles('uniform_EED_000_10km_2.h5','uniform_EED_000_10km_0.1_2')

####### Case 3
scat_0_array, scat_7_array, scat_9_array = make_scatter_array(new_scat_9,new_scat_7,new_scat_0)

r,p,s,thetamesh,radmesh = add_scatter(scat_9_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_009_10km_3',r,p,s,thetamesh,radmesh,scat_9_array)
output_datafiles('uniform_EED_009_10km_3.h5','uniform_EED_009_10km_0.1_3')

r,p,s,thetamesh,radmesh = add_scatter(scat_7_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_007_10km_3',r,p,s,thetamesh,radmesh,scat_7_array)
output_datafiles('uniform_EED_007_10km_3.h5','uniform_EED_007_10km_0.1_3')

r,p,s,thetamesh,radmesh = add_scatter(scat_0_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_000_10km_3',r,p,s,thetamesh,radmesh,scat_0_array)
output_datafiles('uniform_EED_000_10km_3.h5','uniform_EED_000_10km_0.1_3')

####### Case 4
scat_0_array, scat_7_array, scat_9_array = make_scatter_array(new_scat_9,new_scat_7,new_scat_0)

r,p,s,thetamesh,radmesh = add_scatter(scat_9_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_009_10km_4',r,p,s,thetamesh,radmesh,scat_9_array)
output_datafiles('uniform_EED_009_10km_4.h5','uniform_EED_009_10km_0.1_4')

r,p,s,thetamesh,radmesh = add_scatter(scat_7_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_007_10km_4',r,p,s,thetamesh,radmesh,scat_7_array)
output_datafiles('uniform_EED_007_10km_4.h5','uniform_EED_007_10km_0.1_4')

r,p,s,thetamesh,radmesh = add_scatter(scat_0_array,scale_length,rho_2D,vp_2D,vs_2D)
write_h5('uniform_EED_000_10km_4',r,p,s,thetamesh,radmesh,scat_0_array)
output_datafiles('uniform_EED_000_10km_4.h5','uniform_EED_000_10km_0.1_4')

'''

'''
fig,ax = plt.subplots(figsize=(10,13))
ax.set_xlabel('Normalized tracer concentration')
ax.set_ylabel('Radius (km)')
ax.set_xlim((0,1.1))
ax.set_ylim((3481,6371))
ax.grid()
for ii in morb_list:
    conc_list = []
    pt = str(int(ii.split('_')[2]))
    bin_array = main(ii)
    bin_array[:,1] = bin_array[:,1]/bin_array[:,1].max()
    ax.plot(bin_array[:,1],bin_array[:,0],lw=2,label=pt+' % EED')
    for jj in distance_list:
        arg = np.argmin(np.abs(bin_array[:,0]-jj))
        conc_list.append(bin_array[arg,1])


ax.legend(loc=1)
plt.show()
'''


