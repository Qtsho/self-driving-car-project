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

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    #print(my_scene)
    """
    List sample, attributes, categories, scenes
    """
    #nusc_.list_sample()
    #nusc_.list_attributes()
    #nusc_.list_categories()
    #nusc_.list_scenes()

    """ 
    Test API
    """
    # loading scene token
    my_scene_token = nusc_.field2token('scene', 'name', 'scene-0103')
    print ("Scene token:" , my_scene_token)
    #nusc_.render_scene(my_scene_token[0],out_path= '/home/tientran/nuscenes-devkit/Render/scene-0103.avi')
# Query loading all sample tokens in one scene (2hz)
    sample_tokens = nusc_.field2token('sample','scene_token', my_scene_token[0])
    #print (sample_tokens)
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
    
    # get 2D projection
    print("Generating 2D reprojections of the nuScenes dataset")
    sample_data_camera_tokens = [s['token'] for s in nusc_.sample_data if (s['sensor_modality'] == 'camera') and
                                 s['is_key_frame']]

    # For debugging purposes: Only produce the first n images.
    if args.image_limit != -1:
         sample_data_camera_tokens = sample_data_camera_tokens[:args.image_limit]

    # read the reprojection table from json file. 
    # The reprojection is frome the whole scene
    reprojections = []
    rects     =   []
    with open('/home/tientran/data/sets/nuscenes/v1.0-mini/image_annotations.json') as json_file:
        reprojections = json.load(json_file)
    print("Plotting all reprojection")
    for  reprojection in tqdm(reprojections):
        #rects.append(matplotlib.patches.Rectangle(, ), 
        #                             1225.8893058022243, 513.6450176828284,  
        #                            color ='green')) 
        if reprojection["category_name"]=='movable_object.barrier':
            color = 'black'
        elif reprojection["category_name"]=='vehicle.bicycle':
            color = 'red'
        elif reprojection["category_name"]=='vehicle.car':
            color = 'green'
        elif reprojection["category_name"]=='human.pedestrian.adult':
            color = 'salmon' 
        else:
            color = 'orange' 
        if reprojection["sample_data_token"] == 'e3d495d4ac534d54b321f50006683844':
            if reprojection["category_name"] == 'movable_object.barrier':  
                #nusc_.get()
                sample_data = nusc_.get("sample_data", 'e3d495d4ac534d54b321f50006683844')
                sample= nusc_.get("sample", sample_data["sample_token"])
                cam_front_data = nusc_.get('sample_data', sample['data']['CAM_FRONT'])
                #Add position
                pose_record = nusc_.get('ego_pose', reprojection["sample_data_token"])
                ax.add_patch(matplotlib.patches.Circle((pose_record["translation"][0],pose_record["translation"][1]),
                                                        radius=50,color ='blue'))

                ax.add_patch(matplotlib.patches.Rectangle((reprojection["bbox_corners"][0],reprojection["bbox_corners"][1]), 
                                          reprojection["bbox_corners"][3],reprojection["bbox_corners"][2],  
                                         color =color,fill = False)) 
                              
       #if reprojection["instance_token"] == 'f4b2632a2f9947da9f7959a3bd0e322c':
          #  ax.add_patch(matplotlib.patches.Rectangle((reprojection["bbox_corners"][0],reprojection["bbox_corners"][1]), 
           #                           reprojection["bbox_corners"][3],reprojection["bbox_corners"][2],  
             #                        color ='blue',fill = False))                              
            
    nusc_.render_sample_data(cam_front_data['token']) 
    #plt.xlim([-4000, 4000]) 
    #plt.ylim([-4000, 4000]) 


    
    sample_record = nusc_.get('sample', sample_tokens[3])               
    #nusc_.render_sample_data(sample_record['data']['LIDAR_TOP'], nsweeps=5, underlay_map=True)
    #nusc_.render_annotation(sample_annotation_tokens[7])
    # nusc_.render_egoposes_on_map(log_location='boston-seaportsadfsafasf',scene_tokens= my_scene_token)
    

## MAP tutorial
    # layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    # camera_channel = 'CAM_FRONT'
    # nusc_map.render_map_in_image(nusc_, sample_tokens[1], layer_names=layer_names,
    # camera_channel=camera_channel)
    # nusc_map.render_egoposes_on_fancy_map(nusc_,scene_tokens= my_scene_token, verbose=False)

    plt.show()
