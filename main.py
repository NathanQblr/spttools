"""Main of my SPT data analyser module"""
import os, argparse,sys
import yaml
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets


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

        with open(config_path, 'r', encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.delta_t = self.config['data']['delta_t']
        self.sigma_noise = self.config['data']['sigma_noise']
        self.dimension = self.config['data']['d']
        self.pix_size = self.config['data']['pxsize']

        self.path_data = None
        self.data = None
        self.points = None

        self.x_axis = None
        self.y_axis = None
        self.x_center = None
        self.y_center = None
        self.alpha = None

        self.results_tamsd = None
        self.results_mapd = None
        self.voronoi = None
        self.figure_mapd = None
        self.figure_tamsd = None
        self.joint_figure_tamsd = None

    def load_data(self):
        """Load SPT data from a csv and save it into data as DataFrame
        """
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
        """Manage the classical analysis of tamsd on data
        Returns:
            results : DataFrame containing each number of trajectories with its estimated D
        """
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
                    alpha,diffusion_coef,sigma2 = sp.apply_fit(currta_msd,method,self.delta_t,len_tamsd,self.dimension,self.sigma_noise)
                    new_row = {'traj':traj,'length':length,'alphapred':alpha,'sigma2pred':sigma2,'Dpred':diffusion_coef,'tamsd':currta_msd,'jumps':jumpsnbr,'method':method}
                    results.loc[len(results)] = new_row
        return results

    def run_mapd(self):
        """Manage the spatial estimation of D on data

        Returns:
            points : copy of data with the information estimated by mapD on the cluster the frame is and the estimated D
            vornoi : voronoi web
        """
        #Peut être mieux vaut enregistrer voronoi et D
        config =  self.config['mapD']
        npoints_per_clus = config['npnts_per_cluster']
        min_number_per_traj = config['nmin']
        points, voronoi = mm.call_map_coefdiff(self.points,min_number_per_traj,npoints_per_clus,self.sigma_noise,self.delta_t)
        return points, voronoi

    def plot_map_diffusion(self):
        """Plot the results of mapD
        """
        display = self.config['mapD']['display']
        self.figure_mapd = mm.plot_mapcoefdiff(self.results_mapd,self.voronoi,display)

    def plot_tamsd(self):
        """Plot the results of classical SPT analysis
        """
        display = self.config['tamsd']['display']
        methods_plotted = self.config['tamsd']['plot']
        self.figure_tamsd,self.joint_figure_tamsd = sp.plot_tamsd(self.results_tamsd,display,self.delta_t,self.config['tamsd']['len_tamsd'],methods_plotted)

    def select_mask_on_data(self):
        """Permit to open a selection window to keep only zone that we want to analyse on the data
        """
        image, xedges, yedges = np.histogram2d(self.data.x.to_numpy(), self.data.y.to_numpy(),bins=100)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(title='Select your data')
        plt.imshow(image, interpolation='bilinear', origin='lower',norm='symlog',cmap='PiYG',
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
        """Applys the mask selected in select_mask_on_data to data and save it into points
        """
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

    def save_results(self):
        """Save results in the directory chosen in the confiuration file
        """
        path = self.config['data']['path_res_folder']
        self.results_tamsd.to_csv(path+'/results_tamsd.csv')
        self.results_mapd.to_csv(path+'/results_mapd.csv')
        self.figure_mapd.savefig(path+'/fig_mapd.pdf')
        self.figure_tamsd.savefig(path+'/fig_tamsd.pdf')
        self.joint_figure_tamsd.savefig(path+'/joint_fig_tamsd.pdf')

    def run(self):
        """Run all analysis defined in config file
        """
        self.load_data()
        self.apply_mask_on_data()

        if self.config['tamsd']['do']:
            self.results_tamsd = self.run_tamsd()
        if self.config['tamsd']['plot']:
            self.plot_tamsd() #Pb de plot
        if self.config['mapD']['do']:
            self. results_mapd, self.voronoi = self.run_mapd()
        if self.config['mapD']['plot']:
            self.plot_map_diffusion()
        if self.config['data']['path_res_folder'] is not None:
            self.save_results()



def main() :
    '''
    Main function to run an experiment
    Args:
        *args, **kwargs:
            Positional and keyword arguments
    Returns:
        The value returned by calling the runner.
    '''
    parser=argparse.ArgumentParser(description="sample argument parser", conflict_handler='resolve')
    parser.add_argument("config_path")

    if len(sys.argv) == 1:
        print("Config by default")
        path = 'config_def.yaml'
    else :
        arg=parser.parse_args()
        path = arg.config_path
    return Runner(path).run()


if __name__ == '__main__':
    main()
