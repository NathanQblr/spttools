#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:18:21 2024

@author: nathanquiblier
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fbm import fbm



class TA_MSD_METHOD_TESTER():
    def __init__(self):
        self.delta_t=0.01# acquisition period in second (10 ms)
        self.nmin=10# minimal number of points per trajectory
        self.sigma_noise= None #0.06# localization noise
        self.len_msd = 7
        self.len_traj = 99
        self.lim_size_traj = 14
        #self.d = 2 #dimension of diffusion

        columns = ['traj','D','alpha','method','dynamic','noise','lightsheet','alphapred','Dpred','sigma2pred','size_traj']
        self.results = pd.DataFrame(columns= columns)

    def ta_msd_m(self,initx,inity,initz,f,len_msd):
        """ the function that computes the time averaged MSD of a trajectory traj
        in_delta_t is the time between two consecutive frames """
        #attention aux traj avec des frames manquantes
        #in_traj.loc[:,'f'] = in_traj.f.view()-in_traj.f.min()
        msd_array = np.zeros(len_msd+1)#int(in_traj.f.max())+1)
        x = np.ones(int(f.max())+1)*np.inf
        y = np.ones(int(f.max())+1)*np.inf

        x[f] = initx
        y[f] = inity

        for tau in range(len_msd):#int(in_traj.f.max())):
            dx = x[tau+1:]-x[:-tau-1]
            dx = dx[(dx!=np.inf)&(dx!=-np.inf)]
            dx = np.nan_to_num(dx, copy=False)
            dy = y[tau+1:]-y[:-tau-1]
            dy = dy[(dy!=np.inf)&(dy!=-np.inf)]
            dy = np.nan_to_num(dy, copy=False)
            msd_array[tau+1] = np.square(dx).mean()+np.square(dy).mean()
        return msd_array



    def find_traj_lightsheet(self,idx,f):
        max_rep = 0
        rep = 0
        index_start_max_rep =0
        index_start_rep = 0
        frame = 0
        while frame <=idx.size-3:
            if (idx[frame]==1)&(idx[frame+2]==1):
                rep+=3
                frame +=3
            elif (idx[frame]==1)&(idx[frame+1]==1)&(idx[frame+2]==0):
                rep+=1
                frame +=1
            elif (idx[frame]==1)&(idx[frame+1]==0)&(idx[frame+2]==0):
                rep+=1
                frame+=1
            elif (idx[frame]==0)&(idx[frame+1]==0)&(idx[frame+2]==1):
                if rep>=self.lim_size_traj:
                    max_rep = rep
                    index_start_max_rep = index_start_rep
                    frame = f.max()
                rep=0
                index_start_rep = frame+2
                frame += 2
            elif (idx[frame]==0)&(idx[frame+1]==1)&(idx[frame+2]==1):
                rep+=3
                frame+=3
            elif (idx[frame]==0)&(idx[frame+1]==1)&(idx[frame+2]==0):
                rep+=2
                frame+=2
            elif (idx[frame]==0)&(idx[frame+1]==0)&(idx[frame+2]==0):
                frame+=3
        return index_start_max_rep,max_rep


    def apply_fit(self,currta_msd,f, method,d):
        alpha = -1
        D = -1
        sigma2 = -1
        if method == 0: #estim D60 and estimate loc noise => estim alpha
            D ,sigma2 = self.tamsd2Dandnoise(currta_msd,d)
            alpha = self.tamsd2alpha(currta_msd,self.len_msd,sigma2,d)

        if method == 1: #estim D60 and take a priori loc noise => estim alpha
            D ,sigma2 = self.tamsd2Dandnoise(currta_msd,d)
            sigma2 = (0.06)**2
            alpha = self.tamsd2alpha(currta_msd,self.len_msd,sigma2,d)

        if method == 2: #estim alpha without loc => estimate D60 without loc
            alpha = self.tamsd2alphan(currta_msd,self.len_msd)
            D  = self.tamsd2D(currta_msd,alpha,d)
            sigma2 = -1
        if method == 3: #estim alpha without loc and alpha = 1 => estimate D60 without loc
            alpha = self.tamsd2alphan(currta_msd,self.len_msd)
            D  = self.tamsd2D(currta_msd,1,d)
            sigma2 = -1
        return alpha,D,sigma2


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

    def tamsd2Dandnoise(self,currta_msd,d):
        #get the ta_msd up to point 60 ms
        #warining the time of the first point of the trajectory has frame=1
        #i.e. currtraj.f[0]=1, which corresponds to time t=0
        #so take the range from f=2 (t=dt) to f=7 (t=6*dt)
        # take the a priori that msd(t) = loc + 2Dt
        frames = 7
        tamsd60=currta_msd[:frames]
        x_reg=np.linspace(1,frames-1,frames-1,dtype=int)*self.delta_t
        y_reg=tamsd60[1:]
        XX = np.vstack([x_reg, np.ones_like(x_reg)]).T
        # force the lienar fit zo zero intercept
        slope=np.linalg.lstsq(XX,y_reg,rcond=None)[0]
        # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
        Dpred = slope[0]/(2*d)
        loc_noise_squarred = slope[1]/(d*2)

        return Dpred,loc_noise_squarred

    def tamsd2alpha(self,currta_msd,len_pred_alpha,loc_noise_squarred,d):
        tamsdalpha = currta_msd[:len_pred_alpha]
        frames = tamsdalpha.size
        x_reg=np.log(np.linspace(1,frames-1,frames-1,dtype=int)*self.delta_t)
        y_reg = tamsdalpha[1:]-d*2*loc_noise_squarred
        x_reg = x_reg[y_reg>0]
        y_reg=np.log(y_reg[y_reg>0])

        LXLX = np.vstack([x_reg, np.ones_like(x_reg)]).T
        #print(LXLX)
        # force the lienar fit zo zero intercept
        slope_alpha=np.linalg.lstsq(LXLX, y_reg,rcond=None)[0]
       # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
        alphapred = slope_alpha[0]
        return alphapred

    def tamsd2alphan(self,currta_msd,len_pred_alpha):
        tamsdalpha = currta_msd[:len_pred_alpha]
        frames = tamsdalpha.size
        x_reg=np.log(np.linspace(1,frames-1,frames-1,dtype=int)*self.delta_t/((frames-1)*self.delta_t))
        y_reg = tamsdalpha[1:]/tamsdalpha[frames-1]
        x_reg = x_reg[y_reg>0]
        y_reg=np.log(y_reg[y_reg>0])

        LXLX = np.vstack([x_reg, np.ones_like(x_reg)]).T
        #print(LXLX)
        # force the lienar fit zo zero intercept
        slope_alpha=np.linalg.lstsq(LXLX, y_reg,rcond=None)[0]
       # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
       # TA_MSD(t)/TA_MSD(i*delta_t)=~(i*delta_t/t)^alpha
        # À faire TA_MSD(t)/(mean in i TA_MSD(i*delta_t))=~(i*delta_t/t)^alpha à calc
        alphapred = slope_alpha[0]

        return alphapred

    def tamsd2D(self,currta_msd,alpha,d):
        #get the ta_msd up to point 60 ms
        #warining the time of the first point of the trajectory has frame=1
        #i.e. currtraj.f[0]=1, which corresponds to time t=0
        #so take the range from f=2 (t=dt) to f=7 (t=6*dt)
        frames = 7
        tamsd60=currta_msd[:frames]
        x_reg=np.power(np.linspace(1,frames-1,frames-1,dtype=int)*self.delta_t,alpha)-np.power((frames-1)*self.delta_t,alpha)
        y_reg=tamsd60[1:]-tamsd60[frames-1]
        XX = np.vstack([x_reg, np.ones_like(x_reg)]).T
        # force the lienar fit zo zero intercept
        slope=np.linalg.lstsq(XX,y_reg,rcond=None)[0]
        # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
         # TA_MSD(t)-TA_MSD(i*delta_t)=2dD(t^alpha-(i*delta_t)^alpha)
        Dpred = slope[0]/(2*d)
        #loc_noise_squarred = slope[1]/(self.d*2)
        return Dpred

    def computealphaDallmethalldata(self,traj_number):
        new_row_init = {'traj':traj_number}
        D = np.random.uniform(0.01,10)
        alpha = np.random.uniform(0.01,1)
        d = 2
        new_row_init['D'] = D
        new_row_init['alpha'] = alpha
        #Brownian Motion
        num_method = 4
        x,y,z,f = self.get_Bm_3d(D,self.len_traj)
        currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
        for k in range(num_method):
            new_row = new_row_init
            new_row['method'] = k
            new_row['dynamic'] = 'Bm'
            new_row['noise'] = 0
            new_row['lightsheet'] = 0
            new_row['size_traj'] = f.size
            new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
            self.results.loc[len(self.results)] = new_row
        lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
        if lightf.size>=self.lim_size_traj:
            currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
        for k in range(num_method):
            if lightf.size>=self.lim_size_traj:
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'Bm'
                new_row['noise'] = 0
                new_row['lightsheet'] = 1
                new_row['size_traj'] = lightf.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row

        #Add noise
        x,y,z = self.add_noise(x,y,z,self.sigma_noise)
        currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
        for k in range(num_method):
            new_row = new_row_init
            new_row['method'] = k
            new_row['dynamic'] = 'Bm'
            new_row['noise'] = 1
            new_row['lightsheet'] = 0
            new_row['size_traj'] = f.size
            new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
            self.results.loc[len(self.results)] = new_row
        lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
        if lightf.size>=self.lim_size_traj:
            currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
        for k in range(num_method):
            if lightf.size>=self.lim_size_traj:
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'Bm'
                new_row['noise'] = 1
                new_row['lightsheet'] = 1
                new_row['size_traj'] = lightf.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row

        #Fractionnal Brownian Motion
        num_method = 3
        x,y,z,f = self.get_fBm_3d(alpha,D,self.len_traj)
        currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
        for k in range(num_method):
            new_row = new_row_init
            new_row['method'] = k
            new_row['dynamic'] = 'fBm'
            new_row['noise'] = 0
            new_row['lightsheet'] = 0
            new_row['size_traj'] = f.size
            new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
            self.results.loc[len(self.results)] = new_row
        lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
        if lightf.size>=self.lim_size_traj:
            currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
        for k in range(num_method):
            if lightf.size>=self.lim_size_traj:
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'fBm'
                new_row['noise'] = 0
                new_row['lightsheet'] = 1
                new_row['size_traj'] = lightf.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row
        #Add noise
        x,y,z = self.add_noise(x,y,z,self.sigma_noise)
        currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
        for k in range(num_method):
            new_row = new_row_init
            new_row['method'] = k
            new_row['dynamic'] = 'fBm'
            new_row['noise'] = 1
            new_row['lightsheet'] = 0
            new_row['size_traj'] = f.size
            new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
            self.results.loc[len(self.results)] = new_row
        lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
        if lightf.size>=self.lim_size_traj:
            currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
        for k in range(num_method):
            new_row = new_row_init
            new_row['method'] = k
            new_row['dynamic'] = 'fBm'
            new_row['noise'] = 1
            new_row['lightsheet'] = 1
            new_row['size_traj'] = lightf.size
            new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
            self.results.loc[len(self.results)] = new_row

    def computealphaD_2popBM(self,traj_number):
            new_row_init = {'traj':traj_number}
            is_fast = (np.random.uniform()<0.2)
            if is_fast:
                D = np.random.uniform(1,10)
            else:
                D = 0.01
            alpha = 1
            d = 2
            new_row_init['D'] = D
            new_row_init['alpha'] = alpha
            #Brownian Motion
            num_method = 4
            x,y,z,f = self.get_Bm_3d(D,self.len_traj)
            currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
            for k in range(num_method):
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'Bm'
                new_row['noise'] = 0
                new_row['lightsheet'] = 0
                new_row['size_traj'] = f.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row
            lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
            if lightf.size>=self.lim_size_traj:
                currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
            for k in range(num_method):
                if lightf.size>=self.lim_size_traj:
                    new_row = new_row_init
                    new_row['method'] = k
                    new_row['dynamic'] = 'Bm'
                    new_row['noise'] = 0
                    new_row['lightsheet'] = 1
                    new_row['size_traj'] = lightf.size
                    new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                    self.results.loc[len(self.results)] = new_row

            #Add noise
            x,y,z = self.add_noise(x,y,z,self.sigma_noise)
            currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
            for k in range(num_method):
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'Bm'
                new_row['noise'] = 1
                new_row['lightsheet'] = 0
                new_row['size_traj'] = f.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row
            lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
            if lightf.size>=self.lim_size_traj:
                currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
            for k in range(num_method):
                if lightf.size>=self.lim_size_traj:
                    new_row = new_row_init
                    new_row['method'] = k
                    new_row['dynamic'] = 'Bm'
                    new_row['noise'] = 1
                    new_row['lightsheet'] = 1
                    new_row['size_traj'] = lightf.size
                    new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                    self.results.loc[len(self.results)] = new_row

    def computealphaD_2popFBM(self,traj_number):
            new_row_init = {'traj':traj_number}
            is_fast = (np.random.uniform()<0.2)
            if is_fast:
                D = np.random.uniform(1,10)
                alpha = 1
            else:
                D = 0.01
                alpha = np.random.uniform(0.1,1)
            d = 2
            new_row_init['D'] = D
            new_row_init['alpha'] = alpha
            #Fractionnal Brownian Motion
            num_method = 3
            x,y,z,f = self.get_fBm_3d(alpha,D,self.len_traj)
            currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
            for k in range(num_method):
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'fBm'
                new_row['noise'] = 0
                new_row['lightsheet'] = 0
                new_row['size_traj'] = f.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row
            lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
            if lightf.size>=self.lim_size_traj:
                currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
            for k in range(num_method):
                if lightf.size>=self.lim_size_traj:
                    new_row = new_row_init
                    new_row['method'] = k
                    new_row['dynamic'] = 'fBm'
                    new_row['noise'] = 0
                    new_row['lightsheet'] = 1
                    new_row['size_traj'] = lightf.size
                    new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                    self.results.loc[len(self.results)] = new_row
            #Add noise
            x,y,z = self.add_noise(x,y,z,self.sigma_noise)
            currta_msd=self.ta_msd_m(x,y,z,f,self.len_msd)
            for k in range(num_method):
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'fBm'
                new_row['noise'] = 1
                new_row['lightsheet'] = 0
                new_row['size_traj'] = f.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row
            lightx,lighty,lightz,lightf = self.from_3D2lightsheet(x,y,z,f)
            if lightf.size>=self.lim_size_traj:
                currta_msd=self.ta_msd_m(lightx,lighty,lightz,lightf,self.len_msd)
            for k in range(num_method):
                new_row = new_row_init
                new_row['method'] = k
                new_row['dynamic'] = 'fBm'
                new_row['noise'] = 1
                new_row['lightsheet'] = 1
                new_row['size_traj'] = lightf.size
                new_row['alphapred'],new_row['Dpred'],new_row['sigma2pred'] = self.apply_fit(currta_msd,f, k,d)
                self.results.loc[len(self.results)] = new_row