"Format data from csv col = [x,y,trajectory number, frame number] to newcol = [x,y, frame number,trajectory number,'dr2','cl','D','dt']"
import os
import pandas as pd
import numpy as np


NAME_FOLDER = '/Users/nathanquiblier/Downloads/new analyse rpb1 Dmax 2µm2s OK nov 2020'
NAME_F = '../dat/SPT:RPB1_ABC4M_format.txt'

dirlist = os.listdir(NAME_FOLDER)


COL_NAMES=['x', 'y', 'f','traj', 'dr2', 'cl', 'D', 'dt']
datas = []
maxi = 0
for file in dirlist:
    if file[-4:]=='.txt':
      points = pd.read_csv(NAME_FOLDER+'/'+file,sep='\t',header = None,names=COL_NAMES)
      if np.unique(points.traj).size>400:
        #print("oh")
        #maxi = np.unique(points.traj).size
        break
#print("Maxi : ", maxi)

points['traj'] = points['traj']
points['traj'] = points['traj'].astype(int)
points['f'] = points['f'].astype(int)
#points['Dtrue'] = np.zeros(points.shape[0])
#print(points)


points.to_csv(NAME_F,index=False)