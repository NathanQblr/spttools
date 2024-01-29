"Format data from csv col = [x,y,trajectory number, frame number] to newcol = [x,y, frame number,trajectory number,'dr2','cl','D','dt']"
import os
import pandas as pd
import numpy as np


NAME_FOLDER = '/Users/nathanquiblier/Downloads/new analyse rpb1 Dmax 2µm2s OK nov 2020'
NAME_F = '../data/SPT:RPB1_ABC4M_format.txt'

dirlist = os.listdir(NAME_FOLDER)


COL_NAMES=['x', 'y', 'f','traj', 'dr2', 'cl', 'D', 'dt']
datas = []
dataset_n = 1
for file in dirlist:
    if file[-4:]=='.txt':
      datas.append(pd.read_csv(NAME_FOLDER+'/'+file,sep='\t',header = None,names=COL_NAMES))
      datas[-1]['dataset'] = dataset_n*np.ones_like(datas[-1].x)
      dataset_n+=1

points = pd.concat(datas,axis=0,ignore_index=True)
points['traj'] = points['traj']*100+points['dataset']
points['Dtrue'] = np.zeros(points.shape[0])
#print(points)


points.to_csv(NAME_F,index=False)