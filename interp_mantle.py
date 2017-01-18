from scipy import interpolate
import numpy as np

density_load = np.loadtxt('Density_stx11.dat')
vp_load = np.loadtxt('Vp_stx11.dat')
vs_load = np.loadtxt('Vs_stx11.dat')

radius = np.linspace(3390,6371,num=density_load.shape[0])
theta = np.linspace(0,359,num=density_load.shape[1])

f_rho = interpolate.interp2d(theta,radius,density_load)
f_vp = interpolate.interp2d(theta,radius,vp_load)
f_vs = interpolate.interp2d(theta,radius,vs_load)

new_radius = np.linspace(3390,6371,num=2890)
new_theta = np.linspace(0,359,num=36000)
rad_array,theta_array = np.meshgrid(new_theta,new_radius)
size = rad_array.size
new_rad = np.transpose([np.reshape(rad_array,size)])
new_the = np.transpose([np.reshape(theta_array,size)])
new_rho = np.transpose([np.reshape(f_rho(new_theta,new_radius),size)])
new_vp = np.transpose([np.reshape(f_vp(new_theta,new_radius),size)])
new_vs = np.transpose([np.reshape(f_vs(new_theta,new_radius),size)])

final = np.hstack((new_rad, new_the, new_vp, new_vs, new_rho))

np.savetxt('test',final,fmt='%1.3f')
