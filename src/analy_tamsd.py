#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:56:21 2023

@author: huguesberry
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import signal_processing_module


class TA_MSD():
    def __init__(self,name_file,pix_size):
        self.px_size= pix_size# pixel size in micrometer (-)
        self.delta_t=0.01# acquisition period in second (10 ms)
        self.dnpnts_per_clust=100#expected # of points per cluster
        self.nmin=3# minimal number of points per trajectory
        self.sigma_noise=0.06# localization noise
        self.d = 2 #dimension of diffusion

        #initialisations
        self.four_sigma2=4*np.power(self.sigma_noise,2.0)
        #read the raw data - the format is shared with spatial map
        self.col_names=['x', 'y', 'traj','f','Dtrue', 'dr2','cl','D','dt']
        self.points = pd.read_csv(name_file)#.iloc[:100000,:]
        self.points['traj'] = self.points['traj'].astype(int)
        self.points['f'] = self.points['f'].astype(int)
        #print(self.points.iloc[:10,:])
        self.results = pd.DataFrame(columns= ['traj','length','alphapred','Dpred','tamsd','jumps','noise2','Dtrue'])



    def returntrajframe2zero(self,traj):
        f0=self.points[self.points.traj==traj].f.min()
        if f0>0:
            new_frames=self.points[self.points.traj==traj].f-f0
            self.points.loc[(self.points.traj==traj),'f']=new_frames.values
        return self.points[self.points.traj==traj]

    def return_jumps(self,in_traj):
        #in_traj.f = in_traj.f-in_traj.f.min()
        x = np.ones(int(in_traj.f.max())+1)
        x[in_traj.f] = 0
        y = (x[:-1]-x[1:])
        if np.abs(y[y!=0]).sum()==2*x.sum():
            jumpsinf1=1
        else:
            jumpsinf1=0
        return jumpsinf1,x.sum()


    def computealphaD(self,len_pred_alpha):
        self.points.x*=self.px_size
        self.points.y*=self.px_size

        trajnums = np.unique(self.points.traj)

        for i_traj,traj in enumerate(trajnums):
            #traj = 206.0

            currtraj=self.returntrajframe2zero(traj)
            length = currtraj.f.max()
            jumpsinf1,jumpsnbr = self.return_jumps(currtraj)


            if (length>10)&(jumpsinf1==1):#and(currtraj.shape[0]==length+1):
                Dtrue =  currtraj['Dtrue'].iloc[0]
                #computes the TA_MSD of the whole trajectory
                currta_msd=self.ta_msd_m(currtraj,len_pred_alpha)#-self.four_sigma

                #Hugues Method
                Dpred ,loc_noise_squarred = self.tamsd2Dandnoise(currta_msd)
                loc_noise_squarred = 0 #(0.06)**2
                alphapred = self.tamsd2alpha(currta_msd,len_pred_alpha,loc_noise_squarred)

                #My method
                """alphapred = self.tamsd2alphan(currta_msd,len_pred_alpha)
                Dpred  = self.tamsd2D(currta_msd,alphapred)
                loc_noise_squarred =0"""

                new_row = {'traj':traj,'length':length,'alphapred':alphapred,'alphatrue':0,'Dpred':Dpred,'tamsd':currta_msd,'jumps':jumpsnbr,'noise2':loc_noise_squarred,'Dtrue':Dtrue}
                self.results.loc[len(self.results)] = new_row

