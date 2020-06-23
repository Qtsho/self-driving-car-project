# FH Dortmund.
# Code written by Tien Tran, 2020.
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
class LidarProcessing:
    def __init__(self):
        super().__init__()
   
    def get_point (self, data_path):
        lidar_pcls =  LidarPointCloud.from_file(data_path)
        return lidar_pcls
    
    @classmethod
    def proccess_point (cls,lidar_pcls,min_height, max_height,max_measure_long,max_measure_width,pre_process):
        if pre_process:
            z = lidar_pcls.points[2]
            x = lidar_pcls.points[0]
            y = lidar_pcls.points[1]
            intensity_array = lidar_pcls.points[3]
            index_result = np.where((z <= min_height) | (z>= max_height) 
                            |(abs(x) >max_measure_width) |(abs(y) >max_measure_long))
        
            x = np.delete (x,index_result)
            y = np.delete (y,index_result)
            z = np.delete (z,index_result)
            intensity_array = np.delete (intensity_array,index_result)

            lidar_pcls_mod =  np.vstack (((x,y),(z,intensity_array)))
        else:
            z = lidar_pcls.points[2]
            x = lidar_pcls.points[0]
            y = lidar_pcls.points[1]
            intensity_array = lidar_pcls.points[3]
            lidar_pcls_mod =  np.vstack (((x,y),(z,intensity_array)))
        return lidar_pcls_mod