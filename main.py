import yaml
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import matplotlib.widgets as mwidgets
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns

import src.signal_processing_module as sp
import src.map_module as mm

class Runner():
  """Analysis runner
  """
  def __init__(self, config_path: None) -> None:
    '''
    Args:
        config_path:
            The path to the configuration file.
    '''
    self.config_path = config_path
    #self._path_manager = PathManager(config_path=config_path)
    # The Hydra configuration object, which will be set when this is run.
    with open(config_path, 'r') as file:
      self.config = yaml.safe_load(file)
    self.delta_t = self.config['data']['delta_t']
    self.sigma_noise = self.config['data']['sigma_noise']
    self.dimension = self.config['data']['d']
    self.pix_size = self.config['data']['pxsize']

  def load_data(self):
    self.path_data = self.config['data']['path_data']
    columns = self.config['data']['columns']
    if columns == 0 :
      self.data = pd.read_csv(self.path_data, header = 0)#.iloc[:10000,:]
    else:
      self.data = pd.read_csv(self.path_data, columns)#.iloc[:10000,:]
    self.data.x *= self.pix_size
    self.data.y *= self.pix_size
    self.data = self.data.astype({"f": int, "traj": int})

  def run_tamsd(self):
    results = pd.DataFrame(columns = ['traj','length','alphapred','Dpred','sigma2pred','tamsd','jumps','method'])
    config =  self.config['tamsd']
    min_len_traj = config['nmin']
    len_tamsd = config['len_tamsd']
    methods = config['methods']
    trajnums = np.unique(self.points.traj)
    for traj in trajnums:
      currtraj = sp.returntrajframe2zero(self.points,traj)
      length = currtraj.f.max()
      jumpsinfeq1,jumpsnbr = sp.return_jumps(currtraj)
      if (length>min_len_traj)&(jumpsinfeq1==1):
        #computes the TA_MSD of the whole trajectory
        currta_msd=sp.ta_msd_m(currtraj,len_tamsd)
        for method in methods:
          alpha,D,sigma2 = sp.apply_fit(currta_msd,method,self.dimension,self.delta_t,len_tamsd,self.sigma_noise)
          new_row = {'traj':traj,'length':length,'alphapred':alpha,'sigma2pred':sigma2,'Dpred':D,'tamsd':currta_msd,'jumps':jumpsnbr,'method':method}
          results.loc[len(results)] = new_row
    return results

  def run_mapd(self):
    #Peut être mieux vaut enregistrer voronoi et D
    config =  self.config['mapD']
    npoints_per_clus = config['npnts_per_cluster']
    min_number_per_traj = config['nmin']
    if self.config['mask']['is']==1:
      points = pd.DataFrame()
      while points.size<self.config['mask']['ntraj']:
        self.select_mask_on_data()
        if self.config['mask']['type']=='ellipse':
          print("Get rid of trajectories that are not inside the ellipse")
          form = mm.inside_ellipse
        if self.config['mask']['type']=='rectangle':
          print("Get rid of trajectories that are not inside the rectangle")
          form = mm.inside_rectangle
        points = mm.drop_traj_outside_form_in_df(self.data,self.x_center,self.y_center,self.x_axis,self.y_axis,self.alpha,form)
        print("Nuber of trajectories : ",np.unique(points.traj).size)
    else:
      points = self.data[['x', 'y','traj','f']]
    points = points.assign(dr2=0,cl=0,D=0,dt=0) #squared norm between positions, cluster label of point,diff coefficience of point,time difference between positions
    points, voronoi = mm.call_map_D(points,min_number_per_traj,npoints_per_clus,self.sigma_noise,self.delta_t)
    return points, voronoi

  def plot_mapD(self):
    display = self.config['mapD']['display']
    self.figure_mapd = mm.plot_mapD(self.result_mapd,self.voronoi,display)

  def plot_data(self):
    self.load_data()
    print("Size before : ",np.unique(self.data.traj).size)
    resolution = 100
    H, xedges, yedges = np.histogram2d(self.data.x.to_numpy(), self.data.y.to_numpy(),bins=resolution)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(title='Data smoothed')
    plt.imshow(H, interpolation='bilinear', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    def onselect(eclick, erelease):
        print(eclick.xdata, eclick.ydata)
        print(erelease.xdata, erelease.ydata)
    props = dict(facecolor='blue', alpha=0.5)
    shape = mwidgets.RectangleSelector(ax, onselect, interactive=True,props=props)
    shape.add_state('rotate')
    plt.show()
    corners = np.vstack(shape.corners)
    width_shape = np.linalg.norm(corners[:,1] - corners[:,0])
    height_shape = np.linalg.norm(corners[:,1] - corners[:,2])
    x_center = shape.center[0]#*self.pix_size
    y_center = shape.center[0]#*self.pix_size
    x_axis = width_shape#*self.pix_size
    y_axis = height_shape #*self.pix_size
    alpha = shape.rotation
    sommet = shape.center
    sommet -= np.array([width_shape,height_shape])/2
    print("angle = ",alpha)
    print("Hauteur :",height_shape,' largeur : ',width_shape)
    print("Center ",shape.center)
    form = mm.inside_rectangle
    points = mm.drop_traj_outside_form_in_df(self.data,x_center,y_center,x_axis,y_axis,alpha,form)
    print("Size after : ",np.unique(points.traj).size)
    #form = Ellipse(xy = shape.center, width = width_shape, height = height_shape, angle=alpha, alpha=0.1, linewidth=1, edgecolor='r', facecolor='white')
    form = Rectangle(xy = sommet, width=width_shape, height=height_shape, angle=alpha, rotation_point='center')

    fig2 , ax2 = plt.subplots()
    ax2.imshow(H, interpolation='bilinear', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax2.add_patch(form)
    plt.show()

    H, xedges, yedges = np.histogram2d(points.x.to_numpy(), points.y.to_numpy(),bins=100)
    fig2 , ax2 = plt.subplots()
    ax2.imshow(H, interpolation='bilinear', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.show()

  def select_mask_on_data(self):
    H, xedges, yedges = np.histogram2d(self.data.x.to_numpy(), self.data.y.to_numpy(),bins=100)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(title='Data smoothed')
    plt.imshow(H, interpolation='none', origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    def onselect(eclick, erelease):
        print(eclick.xdata, eclick.ydata)
        print(erelease.xdata, erelease.ydata)
    props = dict(facecolor='blue', alpha=0.5)
    if self.config['mask']['type'] == 'ellipse':
      shape = mwidgets.EllipseSelector(ax, onselect, interactive=True,props=props)
    if self.config['mask']['type'] == 'rectangle':
      shape = mwidgets.RectangleSelector(ax, onselect, interactive=True,props=props)
    shape.add_state('rotate')
    plt.show()
    corners = np.vstack(shape.corners)
    self.x_axis = np.linalg.norm(corners[:,1] - corners[:,0])
    self.y_axis = np.linalg.norm(corners[:,1] - corners[:,2])
    self.x_center = shape.center[0]
    self.y_center = shape.center[0]
    self.alpha = shape.rotation


  def apply_mask_on_data(self):
    if self.config['mask']['is']==1:
      self.points = pd.DataFrame()
      while self.points.size<self.config['mask']['ntraj']:
        self.select_mask_on_data()
        if self.config['mask']['type']=='ellipse':
          print("Get rid of trajectories that are not inside the ellipse")
          form = mm.inside_ellipse
        if self.config['mask']['type']=='rectangle':
          print("Get rid of trajectories that are not inside the rectangle")
          form = mm.inside_rectangle
        self.points = mm.drop_traj_outside_form_in_df(self.data,self.x_center,self.y_center,self.x_axis,self.y_axis,self.alpha,form)
        print("Nuber of trajectories : ",np.unique(self.points.traj).size)
    else:
      self.points = self.data[['x', 'y','traj','f']]
    self.points = self.points.assign(dr2=0,cl=0,D=0,dt=0) #squared norm between positions, cluster label of point,diff coefficience of point,time difference between positions


  def run(self):
    self.load_data()
    self.apply_mask_on_data()

    if self.config['tamsd']['do']:
      self.results_tamsd = self.run_tamsd()
    if self.config['mapD']['do']:
      self. result_mapd, self.voronoi = self.run_mapd()
      if self.config['mapD']['plot']:
        self.plot_mapD()
    #Attention à la réutilisatation de data et points ==> effacage de données

  def save_results(self):
    path = self.config['data']['path_res_folder']
    self.results_tamsd.to_csv(path+'results_tamsd.csv')
    self.results_mapd.to_csv(path+'results_mapd.csv')



A = Runner('config_def.yaml')
#A.plot_data()
A.run()
A.results_tamsd.to_csv('res.csv')
plt.show()