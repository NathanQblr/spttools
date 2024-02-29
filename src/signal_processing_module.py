"""Module providing functions for classical spt analysis."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

def tamsd2diffcoefandnoise(currta_msd,d,delta_t):
    """get the ta_msd up to point 60 ms
    warining the time of the first point of the trajectory has frame=1
    i.e. currtraj.f[0]=1, which corresponds to time t=0
    so take the range from f=2 (t=dt) to f=7 (t=6*dt)
    take the a priori that msd(t) = loc_noise + 2Dt
    Args:
        currta_msd (np.array): numpy array containing a TAMSD estimate
        d (int): space dimension of the diffusion
        delta_t (float): time beetween 2 tamsd estimates

    Returns:
        _type_: _description_
    """
    frames = 7
    tamsd60 = currta_msd[:frames]
    x_reg = np.linspace(1,frames-1,frames-1,dtype=int)*delta_t
    y_reg = tamsd60[1:]
    xx = np.vstack([x_reg, np.ones_like(x_reg)]).T
    # force the lienar fit zo zero intercept
    slope=np.linalg.lstsq(xx,y_reg,rcond=None)[0]
    # TA_MSD(t)=2*d*D*t + 2*d*loc_noise_squarred so slope = 2*d*D
    coefdiff_pred = slope[0]/(2*d)
    loc_noise_squarred = slope[1]/(d*2)

    return coefdiff_pred,loc_noise_squarred

def tamsd2alpha(currta_msd,len_pred_alpha,loc_noise_squarred,d,delta_t):
    tamsdalpha = currta_msd[:len_pred_alpha]
    frames = tamsdalpha.size
    x_reg=np.log(np.linspace(1,frames-1,frames-1,dtype=int)*delta_t)
    y_reg = tamsdalpha[1:]-d*2*loc_noise_squarred
    x_reg = x_reg[y_reg>0]
    y_reg=np.log(y_reg[y_reg>0])

    lxlx = np.vstack([x_reg, np.ones_like(x_reg)]).T

    # force the lienar fit zo zero intercept
    slope_alpha=np.linalg.lstsq(lxlx, y_reg,rcond=None)[0]
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

    lxlx = np.vstack([x_reg, np.ones_like(x_reg)]).T
    #print(LXLX)
    # force the lienar fit zo zero intercept
    slope_alpha=np.linalg.lstsq(lxlx, y_reg,rcond=None)[0]
    # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
    # TA_MSD(t)/TA_MSD(i*delta_t)=~(i*delta_t/t)^alpha
    # À faire TA_MSD(t)/(mean in i TA_MSD(i*delta_t))=~(i*delta_t/t)^alpha à calc
    alphapred = slope_alpha[0]

    return alphapred

def tamsd2coeffdiff(currta_msd,alpha,d,delta_t):
    #get the ta_msd up to point 60 ms
    #warining the time of the first point of the trajectory has frame=1
    #i.e. currtraj.f[0]=1, which corresponds to time t=0
    #so take the range from f=2 (t=dt) to f=7 (t=6*dt)
    frames = 7
    tamsd60=currta_msd[:frames]
    x_reg=np.power(np.linspace(1,frames-1,frames-1,dtype=int)*delta_t,alpha)-np.power((frames-1)*delta_t,alpha)
    y_reg=tamsd60[1:]-tamsd60[frames-1]
    xx = np.vstack([x_reg, np.ones_like(x_reg)]).T
    # force the lienar fit zo zero intercept
    slope=np.linalg.lstsq(xx,y_reg,rcond=None)[0]
    # TA_MSD(t)=2*d*D*t so slope = 2*d*D + 2*d*loc_noise_squarred
      # TA_MSD(t)-TA_MSD(i*delta_t)=2dD(t^alpha-(i*delta_t)^alpha)
    coefdiff_pred = slope[0]/(2*d)
    #loc_noise_squarred = slope[1]/(self.d*2)
    return coefdiff_pred


def apply_fit(currta_msd, method,delta_t,len_tamsd,d = 2,sigma=None):
    """Apply the method selected of fit on the currta_msd which is a tamsd estimate

    Args:
        currta_msd (np.array): numpy array containing a TAMSD estimate
        method (int): selector of the method
        d (int): space dimension of the diffusion. Defaults to 2.
        delta_t (float): time beetween 2 tamsd estimates
        len_tamsd (int): choosen len where we fit the msd
        sigma (float): localisation noise. Defaults to None.

    Returns:
        alpha,coef_diff,sigma2: estimation of physical parameters
    """

    #  enum
    alpha = -1
    coef_diff = -1
    if sigma is not None:
        sigma2 = sigma**2
    if method == 0: #estim D60 and estimate loc noise => estim alpha
        coef_diff ,sigma2 = tamsd2diffcoefandnoise(currta_msd,d,delta_t)
        alpha = tamsd2alpha(currta_msd,len_tamsd,sigma2,d,delta_t)

    if method == 1: #estim D60 and take a priori loc noise => estim alpha
        coef_diff ,sigma2bin = tamsd2diffcoefandnoise(currta_msd,d,delta_t)
        alpha = tamsd2alpha(currta_msd,len_tamsd,sigma2,d,delta_t)

    if method == 2: #estim alpha without loc => estimate D60 without loc
        alpha = tamsd2alphan(currta_msd,len_tamsd,delta_t)
        coef_diff = tamsd2coeffdiff(currta_msd,alpha,d,delta_t)
        sigma2 = -1
    if method == 3: #estim alpha without loc and alpha = 1 => estimate D60 without loc
        alpha = tamsd2alphan(currta_msd,len_tamsd,delta_t)
        coef_diff = tamsd2coeffdiff(currta_msd,1,d,delta_t)
        sigma2 = -1
    return alpha,coef_diff,sigma2

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
    y = x[:-1]-x[1:]
    if np.abs(y[y!=0]).sum()==2*x.sum():
        jumpsinf1=1
    else:
        jumpsinf1=0
    return jumpsinf1,x.sum()

def plot_tamsd(res,display,delta_t,len_tamsd,methods):
    """Plot results of tamsd analysis

    Args:
        res (pd.DataFrame): dataframe containing results of tamsd analysis
        display (0 1): bool if we display or no the plots
        method (int): selector of the method
        delta_t (float): time beetween 2 tamsd estimates
        len_tamsd (int): choosen len where we fit the msd

    Returns:
        _type_: _description_
    """
    res['logDpred'] = np.log10(res['Dpred'])
    res['sigma2'] = np.sqrt(res.sigma2pred)
    lines = res.method.isin(methods)
    jointfig = sns.jointplot(data=res.loc[lines,:] , x='logDpred',y='alphapred',hue='method')
    jointfig.set_axis_labels(xlabel='D predicted in log scale',ylabel=r'$\alpha$ predicted')
    jointfig.figure.suptitle(r'Distribution of $\alpha$ and D as function of the estimation method')


    fig,ax=plt.subplots(2,1)
    for line in res.loc[lines,:].index:
        ax[0].loglog(np.linspace(delta_t,len_tamsd*delta_t,len_tamsd+1),res['tamsd'][line])
        ax[0].set_ylabel(r'MSD($\tau$)')
        ax[0].set_title('Time-av. MSD')
        #2. plot the distribution of D60 from TA_MSD
        d60_all = res.loc[lines,'Dpred'].to_numpy()
        #counts, bins = np.histogram(d60_all,bins=50)
        #ax[1].stairs(counts, bins,fill=True)
        ax[1] = sns.histplot(data=res.loc[lines,:],x='Dpred',hue='method' )
        #ax[1].set_xticks(np.arange(0,55,5))
        ax[1].grid(True,linestyle='--', linewidth=0.3)
        ax[1].set_xlabel(r'Est. D60 from the TA_MSD $(\mu \mathrm{m}^2/\mathrm{s})$')
        ax[1].set_ylabel('# of trajectories')
        ax[1].set_title(f'est. D60 = {np.mean(d60_all):3.3}+/-{np.std(d60_all):3.3}')
        fig.tight_layout(pad=2.0)

    if display:
        plt.show()
    return fig,jointfig
