"Format data from csv col = [x,y,trajectory number, frame number] to newcol = [x,y, frame number,trajectory number,'dr2','cl','D','dt']"
import os
import pandas as pd
import numpy as np


NAME_FOLDER = '/Users/nathanquiblier/Downloads/Noyaux complet pour Nathan'
NAME_F = '../dat/SPT:RPB1_ABC4M_nucleus'

dirlist = os.listdir(NAME_FOLDER)


COL_NAMES=['x', 'y', 'f','traj', 'dr2', 'cl', 'D', 'dt']
datas = []
num = 0
for file in dirlist:
    if file[-4:]=='.txt':
      points = pd.read_csv(NAME_FOLDER+'/'+file,sep='\t',header = None,names=COL_NAMES)
      if np.unique(points.traj).size>200:
        points['traj'] = points['traj']
        points['traj'] = points['traj'].astype(int)
        points['f'] = points['f'].astype(int)
        points.to_csv(NAME_F+str(num)+'.txt',index=False)
        num+=1
        if num > 10 :
          break
