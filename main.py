import yaml
import pandas as pd
import numpy as np

import signal_processing_module as sp
import map_module as mm

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
    if columns ==None :
      self.data = pd.read_csv(self.path_data, header = 0)
    else:
      self.data = pd.read_csv(self.path_data, columns)
    self.data.x *= self.pix_size
    self.data.y *= self.pix_size

  def run_tamsd(self):
    results = pd.DataFrame(columns = ['traj','length','alphapred','Dpred','sigma2pred','tamsd','jumps','method'])
    config =  self.config['tamsd']
    min_len_traj = config['nmin']
    len_tamsd = config['len_tamsd']
    methods = config['methods']
    trajnums = np.unique(self.data.traj)
    for traj in trajnums:
      currtraj = sp.returntrajframe2zero(traj)
      length = currtraj.f.max()
      jumpsinfeq1,jumpsnbr = sp.return_jumps(currtraj)
      if (length>min_len_traj)&(jumpsinfeq1==1):
        #computes the TA_MSD of the whole trajectory
        currta_msd=sp.ta_msd_m(currtraj,len_tamsd)
        for method in methods:
          alpha,D,sigma2 = sp.apply_fit(currtraj,method,self.dimension,self.delta_t,len_tamsd,self.sigma_noise**2)
          new_row = {'traj':traj,'length':length,'alphapred':alpha,'sigma2pred':sigma2,'Dpred':D,'tamsd':currta_msd,'jumps':jumpsnbr,'method':method}
          results.loc[len(results)] = new_row
    return results

  def run_mapd(self):
    #Peut être mieux vaut enregistrer voronoi et D
    config =  self.config['mapD']
    npoints_per_clus = config['npnts_per_cluster']
    min_number_per_traj = config['nmin']
    if config['mask']['is']==1:
      x_center = config['mask']['x_center']
      y_center = config['mask']['y_center']
      x_axis = config['mask']['x_axis']
      y_axis = config['mask']['y_axis']
      alpha = config['mask']['alpha']
      if config['mask']['type']=='ellipse':
        print("Get rid of trajectories that are not inside the ellipse")
        points = mm.drop_traj_outside_ellipse_in_df(self.data,x_center,y_center,x_axis,y_axis,alpha)
      if config['mask']['type']=='rectangle':
        print("Get rid of trajectories that are not inside the rectangle")
        print("create this fonction")
        #points = mm.drop_traj_outside_rectangle_in_df(self.data,x_center,y_center,x_axis,y_axis,alpha)
      points.dr2 = np.zeros(np.size(points)) #squared norm between positions
      points.cl = np.zeros(np.size(points))#cluster label of point
      points.D = np.zeros(np.size(points))#diff coefficience of point
      points.dt = np.zeros(np.size(points))#time difference between positions
      voronoi = mm.call_map_D(points,min_number_per_traj,npoints_per_clus,self.sigma_noise)
      return points, voronoi





  def run(self):
    self.load_data()

    if self.config['tamsd']['do'] == 1:
      self.results_tamsd = self.run_tamsd()
    if self.config['mapD']['do'] == 1:
      self.voronoi,self. result_mapd = self.run_mapd()

    #plots