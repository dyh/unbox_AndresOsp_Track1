import pandas as pd
import numpy as np

def ccw2(A,B,C):
    return (C[:,1]-A[:,1])*(B[:,0]-A[:,0]) > (B[:,1]-A[:,1])*(C[:,0]-A[:,0])

def intersect2(A,B):
    o1 = ccw2(A[:,0:2],B[:,0:2],B[:,2:4]) 
    o2 = ccw2(A[:,2:4],B[:,0:2],B[:,2:4]) 
    o3 = ccw2(A[:,0:2],A[:,2:4],B[:,0:2]) 
    o4 = ccw2(A[:,0:2],A[:,2:4],B[:,2:4])
    return np.logical_and((o1 != o2),(o3 != o4)), o1

class counter_classification:
    def __init__(self, df_mov):

        # get info numpy to optimize running time
        self.calib_lines = df_mov.values[:,1:5].astype(int)
        self.bool_guide  = df_mov.values[:,5].astype(bool)       
        self.ids_movements = np.unique(df_mov.values[:,0]).astype(int)
        self.direction_ref = df_mov.values[:,6].astype(bool)
        self.bool_nan_dir = df_mov.values[:,6]!=df_mov.values[:,6]
        # create a checklist to see if the conditions were met
        self.check_list = np.zeros((self.ids_movements.shape[0],self.bool_guide .shape[0]), dtype=bool)
        for i,id in enumerate(self.ids_movements):
            self.check_list[i] = df_mov.values[:,0]==id

    def classify(self,vector):

        out = np.zeros((self.ids_movements.shape[0]),dtype=bool)
        # create matrix with the input vector
        vector2intersect = np.zeros_like(self.calib_lines,dtype=int)
        vector2intersect[:] = vector
        # intersect the vector with every calib line
        res_inter, direction = intersect2(self.calib_lines,vector2intersect)
        # check if the lines should intersec or not
        bool_res = np.equal(res_inter, self.bool_guide)
        # check if the detected cross is in the good direction
        bool_direction = np.equal(self.direction_ref, direction)
        # if we dont care put the dirction to true it to true
        bool_direction[self.bool_nan_dir]=True
        # check all the crossing conditions
        for i,check in enumerate(self.check_list):
            out[i]= bool_res[check].all() and bool_direction[check].all()
        # create the output
        if self.ids_movements[out].size == 0:
            output = np.array([-1])
        elif self.ids_movements[out].size > 1:
            output = np.array([0])
        else:
            output = self.ids_movements[out]

        return output
