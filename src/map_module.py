#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:56:21 2023

@author: huguesberry
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd



def inside_ellipse(xypos, xycenter, xyaxis, angle):
    '''returns true if the points (x,y) are inside the ellipse centered at (xc,yc)
    with xa and ya as x and y axes
    and rotated by angle radians'''
    xypos[:,0] -= xycenter[0]
    xypos[:,1] -= xycenter[1]
    dist = ((xypos[:,0]*np.cos(angle) + xypos[:,1]*np.sin(angle)) / xyaxis[0])**2
    dist += ((xypos[:,0]*np.sin(angle) - xypos[:,1]*np.cos(angle)) / xyaxis[1])**2
    return dist <= 1

def inside_ellipse_by_traj(currtraj, ellipse_x_center, ellipse_y_center, ellipse_x_axis, ellipse_y_axis, ellipse_alpha):
    '''returns boolean vector corresponding to the presence or not of the positions in currtraj inside the ellipse centered at (xc,yc)
    with xa and ya as x and y axes
    and rotated by angle radians'''
    x_posit = np.array([currtraj.x, currtraj.y]).T
    xy_center = np.array([ellipse_x_center, ellipse_y_center])
    xy_axis = np.array([ellipse_x_axis, ellipse_y_axis])
    isinside = inside_ellipse(x_posit, xy_center, xy_axis, ellipse_alpha)
    return isinside

def drop_traj_outside_ellipse_in_df(points, ellipse_x_center, ellipse_y_center, ellipse_x_axis, ellipse_y_axis, ellipse_alpha):
    '''drop the trajectories which has almost 1 point outside the ellipse centered at (xc,yc)
    with xa and ya as x and y axes
    and rotated by angle radians'''
    trajnums = np.unique(points.traj)
    for traj in trajnums:
        currtraj = points[points.traj == traj]
        isinside = inside_ellipse_by_traj(currtraj, ellipse_x_center, ellipse_y_center, ellipse_x_axis, ellipse_y_axis, ellipse_alpha)
        if ~isinside.all():
            # if at least one localization is out of the ellipse, then delete the trajectory
            points = points.drop(points[points.traj == traj].index).reset_index(drop=True)
    return points

def drop_traj_short_in_df(points, min_len):
    '''drop trajectories shorter than min_len in the DataFrame points'''
    trajnums = np.unique(points.traj)
    mask = np.ones(points.shape[0], dtype=bool)
    for traj in trajnums:
        currtraj = points[points.traj == traj]
        if currtraj.shape[0] <= min_len:
            mask[currtraj.index] = False
    return points[mask].reset_index(drop=True)

def compute_difft_diffr2_by_traj(currtraj,delta_t):
    '''compute the time differences and the squared distances in a trajectory currtraj'''
    difft=np.diff(currtraj.f)*delta_t
    diffr2=np.square(np.diff(currtraj.x))+np.square(np.diff(currtraj.y))
    return difft, diffr2

def compute_difft_diffr2_in_df(points,delta_t):
    '''compute the time differences and the squared distances in a DataFrame points'''
    trajnums = np.unique(points.traj)
    for traj in trajnums:
        currtraj = points[points.traj == traj]
        # delete the last data point in the trajectory cause it has no successor
        points = points.drop([points[points.traj == traj].index[-1]]).reset_index(drop=True)
        # inject in the correct columns
        difft, diffr2 = compute_difft_diffr2_by_traj(currtraj,delta_t)
        points.loc[points[points.traj == traj].index, 'dt'] = difft
        points.loc[points[points.traj == traj].index, 'dr2'] = diffr2
    return points

def create_Voronoi(points, number_points_in_cluster):
    '''treat each position as an indepedent point and kmeans them to get their Voronoi diagram
    of cluster in total'''
    tot_number_points = points.shape[0]
    nclust = (np.rint(tot_number_points / number_points_in_cluster)).astype(int)
    # compute the kmeans of the points
    km = KMeans(
        n_clusters=nclust, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit_predict(points.loc[:, ['x', 'y']])
    # tag each point with its cluster
    points.cl = km.labels_
    # add the voronoi meshing
    centers = km.cluster_centers_
    vor = Voronoi(centers)
    return points, vor

def compute_D_by_cl(points, sigma_noise):
    '''compute an estimation of the diffusion coefficient D in each cluster by averaging the distance beetween 2 steps'''
    nclust = max(points.cl)
    clnums = np.unique(points.cl)
    # container for the estimated D in each cluster
    D_by_clust = np.zeros((nclust + 1))
    Jumps_by_clust = np.zeros((nclust + 1))
    # ResD[i,0]= #points of cluster i
    # ResD[i,1]= #DMAP of cluster i
    # cluster by cluster
    for clus in clnums:
        # retrieve the points of cluster clus
        currpoints = points[points.cl == clus]
        # computes the maximum a posteriori of cluster clus
        D_by_clus = np.sum(np.divide(currpoints.dr2, currpoints.dt)) / (4 * currpoints.shape[0])
        D_by_clus -= sigma_noise**2 / currpoints.shape[0] * np.sum(np.divide(1, currpoints.dt))
        points.loc[points[points.cl == clus].index, 'D'] = D_by_clus
        # save for inspection of dependence of cluster size
        # Jumps_by_clust=currpoints.shape[0]
    return points

def call_map_D(points,min_number_per_traj,npoints_per_clus,sigma_noise,delta_t):
    trajnums = np.unique(points.traj)
    print("Nombre traj =",trajnums.size)
    print('computing the Voronoi meshes')
    #============= Compute square displacements =================#
    print("Compute square displacements")
    points = drop_traj_short_in_df(points,min_number_per_traj)
    points = compute_difft_diffr2_in_df(points,delta_t)
    #update the # of trajectories & # of points
    trajnums = np.unique(points.traj)
    #============= Voronoi meshing of the positions =================#
    print("Voronoi meshing of the positions")
    points, vor = create_Voronoi(points,npoints_per_clus)
    print('Infering the diffusion coeficient of each cluster')
    #============= compute the maximum likelyhood in each cluster =================#
    points = compute_D_by_cl(points,sigma_noise)
    return vor

def plot_mapD(points,voronoi,display):
    fig,ax=plt.subplots(3,1)
    # 1. plots the trajectory points colorcoded by their cluster id
    # and their voronoi clustering
    ax[0].set_aspect(1)
    ax[0].set_xlim(0.6,27.9)
    ax[0].set_ylim(0.6,27.9)
    ax[0].set_xticks(np.arange(5,30,5))
    ax[0].set_yticks(np.arange(5,30,5))
    voronoi_plot_2d(voronoi, ax[0],show_vertices=False, line_colors='orange',line_width=1, line_alpha=0.6, point_size=0)
    #plot indiviudal positions on top the mesh
    ax[0].scatter(points.x,points.y, c=points.cl,s=0.3)
    #ax = plt.gca()
    #ax.axis('equal')
    ax[0].set_xlabel(r'x $(\mu \mathrm{m})$')
    ax[0].set_ylabel(r'y $(\mu \mathrm{m})$')
    ax[0].set_title(r'map of the clusters and Voronoid meshing')

    #2. plots the trajectory points colorcoded by their D valur
    mycmap=cm.plasma#cool.reversed()
    #definzes the colorscale
    # find min/max values for normalization
    MINIMA = 0
    MAXIMA = 50
    norm = mpl.colors.Normalize(vmin=MINIMA, vmax=MAXIMA, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=mycmap)

    ax[1].set_aspect(1)
    ax[1].set_xlim(0.6,27.9)
    ax[1].set_ylim(0.6,27.9)
    ax[1].set_xticks(np.arange(5,30,5))
    ax[1].set_yticks(np.arange(5,30,5))
    im=ax[1].scatter(points.x,points.y, c=points.D,s=0.3,cmap=mycmap)
    cbar=plt.colorbar(im)
    ax[1].set_xlabel(r'x $(\mu \mathrm{m})$')
    ax[1].set_ylabel(r'y $(\mu \mathrm{m})$')
    ax[1].set_title(r'map of estimated D')
    cbar.set_label(r'Estimated D $(\mu \mathrm{m}^2/\mathrm{s})$')#, rotation=270)

    #5. plot the distribution of D from the map
    unique_Ds=np.unique(points.D)
    counts, bins = np.histogram(unique_Ds[:-1],bins=50)
    ax[2].stairs(counts, bins,fill=True)
    ax[2].set_xticks(np.arange(0,55,5))
    ax[2].grid(True,linestyle='--', linewidth=0.3)
    ax[2].set_xlabel(r'Est. D from the spatial maps $(\mu \mathrm{m}^2/\mathrm{s})$')
    ax[2].set_ylabel('# of clusters')
    ax[2].set_title(f'est. D = {np.mean(unique_Ds):3.3}+/-{np.std(unique_Ds):3.3}')
    ax[2].set_ylabel('# of clusters')

    if display:
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.show()
    return fig

#parameters
"""
PX_SIZE=0.065# pixel size in micrometer (65 nm)
DELTA_T=0.01# acquisition period in second (10 ms)
NPNTS_PER_CLUST=100#expected # of points per cluster
NMIN=3# minimal number of points per trajectory
SIGMA_NOISE=0.06# localization noise


#boundary of the nucleus in micrometers
#= an ellipse centered on (X_CENTER,Y_CENTER)
# with axis lenght = X_AXIS and Y_AXIS
# and tilted by ALPHA radians from the horizontal
X_CENTER=207.4*PX_SIZE
Y_CENTER=213.1*PX_SIZE
X_AXIS=220*PX_SIZE#336.8/2.*PX_SIZE
Y_AXIS=140*PX_SIZE#254.3/2.*PX_SIZE
ALPHA=-1.032


#plot and cmputation options
PLOT_FIG=True#if true, produces all the plots
#if false use the data saved in file COMPUTED_D_FNAME below
COMPUTED_D_FNAME='clustered_points_wD.csv'

#initialisations

NPNTS_TOT=0# # of points in total in the experiment
FOUR_SIGMA2=4*np.power(SIGMA_NOISE,2.0)

#read the raw data
NAME_F='../dat/SPT:RPB1_ABC4M_format.txt'#'../dat/SPT:MB_0_format.txt'#'../data/0002.rpt_tracked.txt'
COL_NAMES=['x', 'y', 'f','traj', 'dr2', 'cl', 'D', 'dt']#['x', 'y', 'traj','f', 'dr2','cl','D','dt']
points = pd.read_csv(NAME_F, header=0)#.iloc[:100000,:]
points.x *= PX_SIZE
points.y *= PX_SIZE


vor = call_map_D(points,NMIN,NPNTS_PER_CLUST,SIGMA_NOISE)
figure = plot_mapD(points,vor,True)"""
