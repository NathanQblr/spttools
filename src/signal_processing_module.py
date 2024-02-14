import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ta_msd_m(currtraj,len_msd):
    """ the function that computes the time averaged MSD of a trajectory traj
    in_delta_t is the time between two consecutive frames """
    #attention aux traj avec des frames manquantes
    #in_traj.loc[:,'f'] = in_traj.f.view()-in_traj.f.min()
    msd_array = np.zeros(len_msd+1)#int(in_traj.f.max())+1)
    x = np.ones(int(currtraj.f.max())+1)*np.inf
    y = np.ones(int(currtraj.f.max())+1)*np.inf

    x[currtraj.f] = currtraj.x
    y[currtraj.f] = currtraj.y

    for tau in range(len_msd):#int(in_traj.f.max())):
        dx = x[tau+1:]-x[:-tau-1]
        dx = dx[(dx!=np.inf)&(dx!=-np.inf)]
        dx = np.nan_to_num(dx, copy=False)
        dy = y[tau+1:]-y[:-tau-1]
        dy = dy[(dy!=np.inf)&(dy!=-np.inf)]
        dy = np.nan_to_num(dy, copy=False)
        msd_array[tau+1] = np.square(dx).mean()+np.square(dy).mean()
    return msd_array

def tamsd2Dandnoise(currta_msd,d,delta_t):
    #get the ta_msd up to point 60 ms
    #warining the time of the first point of the trajectory has frame=1
    #i.e. currtraj.f[0]=1, which corresponds to time t=0
    #so take the range from f=2 (t=dt) to f=7 (t=6*dt)
    # take the a priori that msd(t) = loc + 2Dt
    frames = 7
    tamsd60=currta_msd[:frames]
    x_reg=np.linspace(1,frames-1,frames-1,dtype=int)*delta_t
    y_reg=tamsd60[1:]
    XX = np.vstack([x_reg, np.ones_like(x_reg)]).T
    # force the lienar fit zo zero intercept
    slope=np.linalg.lstsq(XX,y_reg,rcond=None)[0]
    # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
    Dpred = slope[0]/(2*d)
    loc_noise_squarred = slope[1]/(d*2)

    return Dpred,loc_noise_squarred

def tamsd2alpha(currta_msd,len_pred_alpha,loc_noise_squarred,d,delta_t):
    tamsdalpha = currta_msd[:len_pred_alpha]
    frames = tamsdalpha.size
    x_reg=np.log(np.linspace(1,frames-1,frames-1,dtype=int)*delta_t)
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

def tamsd2alphan(currta_msd,len_pred_alpha,delta_t):
    tamsdalpha = currta_msd[:len_pred_alpha]
    frames = tamsdalpha.size
    x_reg=np.log(np.linspace(1,frames-1,frames-1,dtype=int)*delta_t/((frames-1)*delta_t))
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

def tamsd2D(currta_msd,alpha,d,delta_t):
    #get the ta_msd up to point 60 ms
    #warining the time of the first point of the trajectory has frame=1
    #i.e. currtraj.f[0]=1, which corresponds to time t=0
    #so take the range from f=2 (t=dt) to f=7 (t=6*dt)
    frames = 7
    tamsd60=currta_msd[:frames]
    x_reg=np.power(np.linspace(1,frames-1,frames-1,dtype=int)*delta_t,alpha)-np.power((frames-1)*delta_t,alpha)
    y_reg=tamsd60[1:]-tamsd60[frames-1]
    XX = np.vstack([x_reg, np.ones_like(x_reg)]).T
    # force the lienar fit zo zero intercept
    slope=np.linalg.lstsq(XX,y_reg,rcond=None)[0]
    # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
      # TA_MSD(t)-TA_MSD(i*delta_t)=2dD(t^alpha-(i*delta_t)^alpha)
    Dpred = slope[0]/(2*d)
    #loc_noise_squarred = slope[1]/(self.d*2)
    return Dpred


def apply_fit(currta_msd, method,d,delta_t,len_msd,sigma=None):

    #  enum
    alpha = -1
    D = -1
    if sigma != None:
        sigma2 = sigma**2
    if method == 0: #estim D60 and estimate loc noise => estim alpha
        D ,sigma2 = tamsd2Dandnoise(currta_msd,d,delta_t)
        alpha = tamsd2alpha(currta_msd,len_msd,sigma2,d,delta_t)

    if method == 1: #estim D60 and take a priori loc noise => estim alpha
        D ,sigma2 = tamsd2Dandnoise(currta_msd,d,delta_t)
        alpha = tamsd2alpha(currta_msd,len_msd,sigma2,d,delta_t)

    if method == 2: #estim alpha without loc => estimate D60 without loc
        alpha = tamsd2alphan(currta_msd,len_msd,delta_t)
        D = tamsd2D(currta_msd,alpha,d,delta_t)
        sigma2 = -1
    if method == 3: #estim alpha without loc and alpha = 1 => estimate D60 without loc
        alpha = tamsd2alphan(currta_msd,len_msd,delta_t)
        D = tamsd2D(currta_msd,1,d,delta_t)
        sigma2 = -1
    return alpha,D,sigma2

def returntrajframe2zero(points,traj):
    """Put the first frame of a trajectory as frame 0

    Args:
        points (DataFrame): the dataframe containing all the trajectories
        traj (int): the trajectory to normlalize

    Returns:
        DataFrame : the modified trajectory
    """
    f0=points[points.traj==traj].f.min()
    if f0>0:
        new_frames=points[points.traj==traj].f-f0
        points.loc[(points.traj==traj),'f']=new_frames.values
    return points[points.traj==traj]

def return_jumps(in_traj):
    """ Analyse if a trajectory as jumps and count that jumps

    Args:
        in_traj (DataFrame): An spst trajectory from a dataframe

    Returns:
        int : 0 if jumps are greater tstrictly than 1 frame, 1 if else
        int : number of 1 frame jumps
    """
    x = np.ones(int(in_traj.f.max())+1)
    x[in_traj.f] = 0
    y = (x[:-1]-x[1:])
    if np.abs(y[y!=0]).sum()==2*x.sum():
        jumpsinf1=1
    else:
        jumpsinf1=0
    return jumpsinf1,x.sum()