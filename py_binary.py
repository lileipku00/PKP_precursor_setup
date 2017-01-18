#!/usr/bin/env python

import os
import subprocess
import numpy as np
import obspy

# Samuel Haugland May 2015
# Use Jeroen's axisem_2xh script to convert ASCII seismogram files into xh binary
# files.

ASCII_seis_list = os.listdir('.')

for idx, ii in enumerate(ASCII_seis_list):
    if ii.find('.dat') == -1:
        ASCII_seis_list.remove(ii)

strlist = []
#ASCII_seis_list.remove('UNPROCESSED')
for idx, ii in enumerate(ASCII_seis_list):
    chname = ii.split('_')[-1][0]
    if ii.split('_')[0] == 'STAT':
        statloc = ii.split('_')[1]
        string = ('axisem_2xh '+ii+' '+str(idx)+chname+'.bin '+'-h '+'100 '+'-l '+statloc
                 +' '+'-c '+chname)
        subprocess.call(string, shell=True)
    elif ii.split('_')[0] == 'STATneg':
        statloc = str(float(ii.split('_')[1])*-1.)
        string = ('axisem_2xh '+ii+' '+str(idx)+chname+'.bin '+'-h '+'100 '+'-l '+statloc
                 +' '+'-c '+chname)
        subprocess.call(string, shell=True)

#subprocess.call('cat *E.bin > E_final_trace', shell=True) 
#subprocess.call('cat *N.bin > N_final_trace', shell=True) 
subprocess.call('cat *Z.bin > Z_final_trace', shell=True) 
subprocess.call('rm *.bin', shell=True) 


#est = obspy.read('E_final_trace')
#nst = obspy.read('N_final_trace')
zst = obspy.read('Z_final_trace')

#for tr in nst:
#    tr.stats.distance = float(tr.stats.station[0:3])*111194.
for tr in zst:
    tr.stats.distance = float(tr.stats.station[0:3])*111194.
    

    
