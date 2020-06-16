# FH Dortmund.
# Code written by Tien Tran, 2020.
import nuscenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import argparse
import os
from tqdm import tqdm
import json
import export_2d_annotations_as_json as exp
from nuscenes import NuScenes
import matplotlib.pyplot as plt
import matplotlib 
from nuscenes.map_expansion.map_api import NuScenesMap
import scipy.io
from typing import List, Tuple, Union
from collections import OrderedDict
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from nuscenes.utils.geometry_utils import view_points

import numpy as np 
def inverse_sensor_model():
    print()

    
if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Test that the installed dataset is complete.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/home/tientran/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-mini.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    
    parser.add_argument('--visibilities', type=str, default=['', '1', '2', '3', '4'],
                        help='Visibility bins, the higher the number the higher the visibility.', nargs='+')
    parser.add_argument('--image_limit', type=int, default=-1, help='Number of images to process or -1 to process all.')
    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose = bool(args.verbose)
    nusc_map = NuScenesMap(dataroot='/home/tientran/data/sets/nuscenes', map_name='boston-seaport')
    # Init.
    nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)
   
    my_scene = nusc_.scene[0]

    
    #print(my_scene)
    """
    List sample, attributes, categories, scenes
    """
    #nusc_.list_sample()
    #nusc_.list_attributes()
    #nusc_.list_categories()
    nusc_.list_scenes()

    """ 
    Test API
    """
    # loading scene token
    my_scene_token = nusc_.field2token('scene', 'name', 'scene-0061')
    print ("Scene token:" , my_scene_token)
    #nusc_.render_scene(my_scene_token[0],out_path= '/home/tientran/nuscenes-devkit/Render/scene-0103.avi')
# Query loading all sample tokens in one scene (2hz)
    sample_tokens = nusc_.field2token('sample','scene_token', my_scene_token[0])
    
#  Query  annotations in [] sample    
    sample_annotation_tokens = nusc_.field2token('sample_annotation','sample_token',sample_tokens[24] )
    
    
    # for loop through all sample records of this scene
    for sample_token in sample_tokens:
                sample_record = nusc_.get('sample', sample_token)
                sample_data_record = nusc_.get('sample_data', sample_record['data']['LIDAR_TOP'])
                pose_record = nusc_.get('ego_pose', sample_data_record['ego_pose_token'])
                #print (sample_data_record)

    #for loop through all annotation in one sample
    annotation_position = []
    for sample_annotation_token in sample_annotation_tokens:
            #print (sample_annotation_token)
            sample_annotation = nusc_.get('sample_annotation', sample_annotation_token)
            annotation_position.append(sample_annotation.get("translation"))
    
   
    fig, axes = plt.subplots(figsize=(18, 9))
    view=  np.eye(4)

    sample_record = nusc_.get('sample', sample_tokens[0])
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc_.get_sample_data(lidar)
    """
    Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
    """
    LidarPointCloud.from_file(data_path).render_height(axes, view=view)

    for box in boxes:
            box.render(axes, view=view)
            corners = view_points(boxes[0].corners(), view, False)[:2, :]
            axes.set_xlim([np.min(corners[0, :]) - 100, np.max(corners[0, :]) + 100])
            axes.set_ylim([np.min(corners[1, :]) - 100, np.max(corners[1, :]) + 100])
            axes.axis('off')
            axes.set_aspect('equal')
    nusc_.render_pointcloud_in_image(sample_record['token'], pointsensor_channel='LIDAR_TOP',render_intensity= True,
    camera_channel = 'CAM_FRONT')
    plt.show()
