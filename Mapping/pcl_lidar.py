# FH Dortmund.
# Code written by Tien Tran, 2020.
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
file_path = '/home/tientran/data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin'

with open(file_path, "rb") as f:
    number = f.read(4)
    while number != b"":
        print(np.frombuffer(number, dtype=np.float32))
        number = f.read(4)
        
data_path, boxes, camera_intrinsic = nusc_.get_sample_data(lidar_token)
    
view=  np.eye(4)
lidar_pcls =  LidarPointCloud.from_file(data_path)
# p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
# seg = p.make_segmenter()
# seg.set_model_type(pcl.SACMODEL_PLANE)
# seg.set_method_type(pcl.SAC_RANSAC)
# indices, model = seg.segment()
    

class ground_plane:
    def __init__(self):
        
