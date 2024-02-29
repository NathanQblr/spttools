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



def inside_ellipse(xypos, xycenter, xyaxis, angle):
    '''returns true if the points (x,y) are inside the ellipse centered at (xc,yc)
    with xa and ya as x and y axes
    and rotated by angle radians'''
    xypos[:,0] -= xycenter[0]
    xypos[:,1] -= xycenter[1]
    dist = ((xypos[:,0]*np.cos(angle) + xypos[:,1]*np.sin(angle)) / xyaxis[0])**2
    dist += ((xypos[:,0]*np.sin(angle) - xypos[:,1]*np.cos(angle)) / xyaxis[1])**2
    return dist <= 1

def inside_rectangle(xypos, xycenter, xyaxis, angle):
    '''returns true if the points (x,y) are inside the rotated rectangle centered at (xc,yc)
    with given width, height and rotated by angle radians'''
    xypos[:,0] -= xycenter[0]
    xypos[:,1] -= xycenter[1]

    # Rotate the points back by the negative angle
    rot_x = xypos[:,0]*np.cos(-angle) - xypos[:,1]*np.sin(-angle)
    rot_y = xypos[:,0]*np.sin(-angle) + xypos[:,1]*np.cos(-angle)

    # Check if the points are inside the rectangle
    return (np.abs(rot_x) < xyaxis[0]/2) & (np.abs(rot_y) < xyaxis[1]/2)

def inside_form_by_traj(currtraj, form_x_center, form_y_center, form_x_axis, form_y_axis, form_alpha,form):
    '''returns boolean vector corresponding to the presence or not of the positions in currtraj inside the form parammetrized by form == 1'''
    x_posit = np.array([currtraj.x, currtraj.y]).T
    xy_center = np.array([form_x_center, form_y_center])
    xy_axis = np.array([form_x_axis, form_y_axis])
    isinside = form(x_posit, xy_center, xy_axis, form_alpha)
    return isinside

def drop_traj_outside_form_in_df(points, form_x_center, form_y_center, form_x_axis, form_y_axis, form_alpha,form):
    '''drop the trajectories which has almost 1 point outside the ellipse centered at (xc,yc)
    with xa and ya as x and y axes
    and rotated by angle radians'''
    trajnums = np.unique(points.traj)
    for traj in trajnums:
        currtraj = points[points.traj == traj]
        isinside = inside_form_by_traj(currtraj, form_x_center, form_y_center, form_x_axis, form_y_axis, form_alpha,form)
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
    currtraj = currtraj.sort_values(by = 'f')
    difft=np.diff(currtraj.f)*delta_t
    diffr2=np.square(np.diff(currtraj.x))+np.square(np.diff(currtraj.y))
    return difft, diffr2

def compute_difft_diffr2_in_df(points,delta_t):
    '''compute the time differences and the squared distances in a DataFrame points'''
    trajnums = np.unique(points.traj)
    mask = np.ones(points.shape[0], dtype=bool)
    for traj in trajnums:
        currtraj = points[points.traj == traj]
        # inject in the correct columns
        difft, diffr2 = compute_difft_diffr2_by_traj(currtraj,delta_t)
        # delete the last data point in the trajectory cause it has no successor
        mask[points[points.traj == traj].index[-1]] = False
        if (difft.max()>2*delta_t) or (difft.min()==0):
            #delete points with more than 2 frames missing
            #and trajectories with 2 frame numbers identical
            mask[currtraj.index] = False
        points.loc[points[points.traj == traj].index[:-1], 'dt'] = difft
        points.loc[points[points.traj == traj].index[:-1], 'dr2'] = diffr2
    return points[mask].reset_index(drop=True)

def create_voronoi(points, number_points_in_cluster):
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

def compute_coeffdiff_by_cl(points, sigma_noise):
    '''compute an estimation of the diffusion coefficient D in each cluster by averaging the distance beetween 2 steps'''
    nclust = max(points.cl)
    #print("Nclust =",nclust)
    clnums = np.unique(points.cl)
    # container for the estimated D in each cluster
    coefdiff_by_clust = np.zeros((nclust + 1))
    jumps_by_clust = np.zeros((nclust + 1))
    # ResD[i,0]= #points of cluster i
    # ResD[i,1]= #DMAP of cluster i
    # cluster by cluster
    for clus in clnums:
        # retrieve the points of cluster clus
        currpoints = points[points.cl == clus]
        # computes the maximum a posteriori of cluster clus
        coefdiff_by_clust[clus] = np.sum(np.divide(currpoints.dr2, currpoints.dt)) / (4 * currpoints.shape[0])
        coefdiff_by_clust[clus] -= sigma_noise**2 / currpoints.shape[0] * np.sum(np.divide(1, currpoints.dt))
        points.loc[points.cl == clus, 'D'] = coefdiff_by_clust[clus]
        # save for inspection of dependence of cluster size
        jumps_by_clust=currpoints.shape[0]
    return points

def call_map_coefdiff(points,min_number_per_traj,npoints_per_clus,sigma_noise,delta_t):
    """Call every function to create a voronoi map of diffusion coefficient on points data

    Args:
        points (pd.DataFrame): data for analysis
        min_number_per_traj (int): minimimal number necessary to keep a trajectory for analysis
        npoints_per_clus (int): number of points in each voronoi cell
        sigma_noise (float): localisation noise
        delta_t (float): time beetween 2 frames

    Returns:
        points,vor: results opf mapd D and voronoi diagram
    """
    trajnums = np.unique(points.traj)
    print("Nombre traj data=",trajnums.size)
    print('computing the Voronoi meshes')
    #============= Compute square displacements =================#
    print("Compute square displacements")
    points = drop_traj_short_in_df(points,min_number_per_traj)
    points = compute_difft_diffr2_in_df(points,delta_t)
    #update the # of trajectories & # of points
    trajnums = np.unique(points.traj)
    print("Nombre traj after tri=",trajnums.size)
    #============= Voronoi meshing of the positions =================#
    print("Voronoi meshing of the positions")
    points, vor = create_voronoi(points,npoints_per_clus)
    print('Infering the diffusion coeficient of each cluster')
    #============= compute the maximum likelyhood in each cluster =================#
    points = compute_coeffdiff_by_cl(points,sigma_noise)
    return points,vor

def plot_mapcoefdiff(points,voronoi,display):
    """Plot results of methods of estimation of diffusion coefficient map

    Args:
        points (pd.DataFrame): data for analysis
        voronoi (_type_): voronoi result of the map
        display (0 1): boolean is we display plot or not

    Returns:
        matplotlib.figure: plot
    """
    fig,ax=plt.subplots(3,1)
    # 1. plots the trajectory points colorcoded by their cluster id
    # and their voronoi clustering
    ax[0].set_aspect(1)
    #ax[0].set_xlim(0.6,27.9)
    #ax[0].set_ylim(0.6,27.9)
    #ax[0].set_xticks(np.arange(5,30,5))
    #ax[0].set_yticks(np.arange(5,30,5))
    voronoi_plot_2d(voronoi, ax[0],show_vertices=False, line_colors='orange',line_width=1, line_alpha=0.6, point_size=0)
    #plot indiviudal positions on top the mesh
    ax[0].scatter(points.x,points.y, c=points.cl,s=0.3)
    #ax = plt.gca()
    #ax.axis('equal')
    ax[0].set_xlabel(r'x $(\mu \mathrm{m})$')
    ax[0].set_ylabel(r'y $(\mu \mathrm{m})$')
    ax[0].set_title(r'map of the clusters and Voronoid meshing')

    #2. plots the trajectory points colorcoded by their D valur
    mycmap = cm.plasma#cool.reversed()
    #definzes the colorscale
    # find min/max values for normalization
    min_coeffdiff = 0
    max_coeffdiff = 50
    norm = mpl.colors.Normalize(vmin=min_coeffdiff, vmax=max_coeffdiff, clip=False)
    mapper = cm.ScalarMappable(norm=norm, cmap=mycmap)

    ax[1].set_aspect(1)
    #ax[1].set_xlim(0.6,27.9)
    #ax[1].set_ylim(0.6,27.9)
    #ax[1].set_xticks(np.arange(5,30,5))
    #ax[1].set_yticks(np.arange(5,30,5))
    im=ax[1].scatter(points.x,points.y, c=points.D,s=0.3,cmap=mycmap)
    cbar=plt.colorbar(im)
    ax[1].set_xlabel(r'x $(\mu \mathrm{m})$')
    ax[1].set_ylabel(r'y $(\mu \mathrm{m})$')
    ax[1].set_title(r'map of estimated D')
    cbar.set_label(r'Estimated D $(\mu \mathrm{m}^2/\mathrm{s})$')#, rotation=270)

    #5. plot the distribution of D from the map
    unique_coefdiff=np.unique(points.D)
    counts, bins = np.histogram(unique_coefdiff[:-1],bins=50)
    ax[2].stairs(counts, bins,fill=True)
    #ax[2].set_xticks(np.arange(0,55,5))
    ax[2].grid(True,linestyle='--', linewidth=0.3)
    ax[2].set_xlabel(r'Est. D from the spatial maps $(\mu \mathrm{m}^2/\mathrm{s})$')
    ax[2].set_ylabel('# of clusters')
    ax[2].set_title(f'est. D = {np.mean(unique_coefdiff):3.3}+/-{np.std(unique_coefdiff):3.3}')
    ax[2].set_ylabel('# of clusters')

    if display:
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.show()
    return fig
