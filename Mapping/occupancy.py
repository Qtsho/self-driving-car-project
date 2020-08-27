# FH Dortmund.
# Code written by Tien Tran, 2020.
import nuscenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import argparse
import os
from tqdm import tqdm
import json
from nuscenes import NuScenes
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from nuscenes.map_expansion.map_api import NuScenesMap
import scipy.io
from typing import List, Tuple, Union
from collections import OrderedDict
from pyquaternion.quaternion import Quaternion
from shapely.geometry import MultiPoint, box
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from PIL import Image
from matplotlib import cm
import datetime
import occupancy_grid_utils as occ
import inverse_sensor_model as inv
import lidar_processing_utils 
import bresenham
import ransac
import pcl_lidar
    
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
    #fig, axes = plt.subplots(figsize=(18, 9))
    # Init.
    nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)
   
    my_scene = nusc_.scene[0]
    prior = np.log(0.5 / (1 - 0.5))
    #lidar_pcls.render_height(axes, view=view)

    ##PARAMS for mapping
    # Renault Zoe max height is 1.56m
    min_height = -1.5
    max_height = 0.1
    #how far from the car to get measured point
    width = 10
    length = 10

    #PARAMS for RANSAC
    # Renault Zoe max height is 1.56m
    min_height_ran = -10
    max_height_ran= 2
    #how far from the car to get measured point
    max_width_ran = 8
    max_long_ran = 20

    r = 1
    local_map = []
    pose_different_x = []
    pose_different_y = []
    map_scale = 2
    
    # Create a map
    map = occ.Map( lenght=length*map_scale,width=width*map_scale,resolution = r)
    
    
    #Ploting option

    print_local_map = False
    render_sample_data = False
    detect_ground = True
    test_ransac= False
    """
    List sample, attributes, categories, scenes
    """
    #nusc_.list_sample()
    #nusc_.list_attributes()
    #nusc_.list_categories()
    #nusc_.list_scenes()
    #SELECT SCENE
    my_scene_token = nusc_.field2token('scene', 'name', 'scene-0103')
    print ("Scene token:" , my_scene_token)
    # Query loading all sample tokens in one scene (2hz)
    sample_tokens = nusc_.field2token('sample','scene_token', my_scene_token[0])
    # get the first pose record to calculate pose different
    
    sample_record = nusc_.get('sample', sample_tokens[0])
    sample_data_record = nusc_.get('sample_data', sample_record['data']['LIDAR_TOP'])
    first_pose_record = nusc_.get('ego_pose', sample_data_record['ego_pose_token'])

    #Loop through all samples


    for l in tqdm(range(len(sample_tokens))):
        #get first sample of the scene
        sample_record = nusc_.get('sample', sample_tokens[l])
        lidar_token = sample_record['data']['LIDAR_TOP']
        sample_data_record = nusc_.get('sample_data', lidar_token)
         
        if render_sample_data == True:
            nusc_.render_sample_data(lidar_token)
        #position of that sensor
        pose_record = nusc_.get('ego_pose', sample_data_record['ego_pose_token'])
        time_stamp = pose_record ['timestamp']
        print(datetime.datetime.fromtimestamp(time_stamp/ 1000000.0))
        #convert to first pose is the center of the map
       
        pose_different_x.append(pose_record['translation'][0] - first_pose_record['translation'][0])  
        pose_different_y.append(pose_record['translation'][1] - first_pose_record['translation'][1])
        
        data_path, boxes, camera_intrinsic = nusc_.get_sample_data(lidar_token)
    
        view=  np.eye(4)
        lidar_pcls =  LidarPointCloud.from_file(data_path)

        #BEGIN RANSAC
        if detect_ground == True:
            ransac_point = lidar_processing_utils.LidarProcessing.proccess_point(lidar_pcls=lidar_pcls,
                                            min_height = min_height_ran,max_height = max_height_ran,
                                            max_measure_long = max_long_ran, max_measure_width = max_width_ran, pre_process= True)
            ##TEST DATA
           
            ##Run RANSAC to l sample data
            #Running 1 features as z to fit the height
            print("Detecting ground plane with RANSAC")
            ransac_fit = ransac.Ransac()
            ransac_point = ransac_point[:3,:]
            ransac_point_t = ransac_point.T
            
            if test_ransac == True:         
             
                x_array =  [1,2,3,2,5,6,47,8,9,10]
                y_array =  [1,2,3,4,15,7,6,2,1,10]
                z_array =  [0,1,2,6,-0,-4,-0,-0,-0,-0.2]

                point_test = np.array ([x_array,y_array,z_array],np.int32)
                ransac_point=point_test
                ransac_point_t = point_test.T
            plane, inlier_list, d, outlier_list= ransac_fit.fit_plane_ransac(ransac_point_t,iters = 10000, 
                                                        inlier_thresh = 0.01,return_outlier_list = True,plot= True) #[x y z]T
           
            print(inlier_list)
            ## calculate the plane equation ax+by+cz = d
            z = lambda x,y: (- d - plane[0]*x -  plane[1]*y) /  plane[2]
            
            tmp_x = np.linspace(np.amin(ransac_point[0 ,inlier_list]),np.amax(ransac_point[0,inlier_list]),10)
            tmp_y = np.linspace(np.amin(ransac_point[1 ,inlier_list]),np.amax(ransac_point[1,inlier_list]),10)
            x,y = np.meshgrid(tmp_x,tmp_y)

            fig = plt.figure()
            
            ax = Axes3D(fig)
            #plcs = lidar_pcls 
            #plcs.points = ransac_point
            #lidar_pcls.render_height(ax,view = view)

            #ax.scatter(ransac_point[0],ransac_point[1],ransac_point[2],c = 'r', marker= ',')
            
            ax.scatter(ransac_point[0,inlier_list],ransac_point[1,inlier_list],ransac_point[2,inlier_list],c = ransac_point[2,inlier_list], marker= ',')
            ax.scatter(ransac_point[0,outlier_list],ransac_point[1,outlier_list],ransac_point[2,outlier_list],c = 'g', marker= ',')
           
            ax.plot_surface(x, y, z(x,y))
            #ax.plot_surface(x, y,0,cmap='viridis')
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            #ax.scatter(ransac_point[0,inlier_list],ransac_point[1,inlier_list],ransac_point[2,inlier_list],c = 'b', marker= ',')
            
            

            plt.show()
           

            ##END RANSAC

        lidar_pcls_mod = lidar_processing_utils.LidarProcessing.proccess_point(lidar_pcls=lidar_pcls,
                                            min_height = min_height,max_height = max_height,
                                            max_measure_long = length, max_measure_width = width, pre_process= True)
        
        print (lidar_pcls_mod)
        
        
        
        #create an map grid 100*100 with resolution r (m)
        
        lidar_pcls_mod= lidar_pcls_mod/r # convert to fit the map
        #using Bresenham algorithm to calculate line between endpoint
        #
        #loop through all measurements
        xe = lidar_pcls_mod[0] + pose_different_y[l]
        min_x = np.amin(xe)
        ye = lidar_pcls_mod[1] + pose_different_x[l]
        min_y = np.amin(ye)

        xe += abs(min_x)
        ye += abs(min_y)
        sensor_position = [int(abs(min_x))+int(pose_different_y[l]),int(abs(min_y))+int(pose_different_x[l])] 
        sensor_pos_local =[int(abs(np.amin(lidar_pcls_mod[0]))),
                        int(abs(np.amin(lidar_pcls_mod[1])))] 
        #intergrate measurement 5 times
        #local
        local_map.append(occ.Map( lenght=length*map_scale,
                                width=width*map_scale,resolution = r))
        local_map[l].mapUpdate(sensor_pos_local,lidar_pcls_mod,0,0)
        
        #global
        map.mapUpdate(sensor_position,lidar_pcls_mod,pose_different_x[l],pose_different_y[l])
      
    #print local map
    if print_local_map == True:
        fig, axs = plt.subplots(8,5)
        index = 0
        fig.figsize=(map.lenght/2, map.width/2)
        fig.suptitle('Local maps', fontsize=16)
        for i in range (5):
            for j in range (8):
                axs[j,i].contourf(local_map[index].grid[:,:], cmap=cm.Greys)
                axs[j,i].set_title('n = %d' % index)
                axs[j,i].get_xaxis().set_visible(False)
                axs[j,i].get_yaxis().set_visible(False)
                index += 1
                axs[j,i].axis('tight')
    
    #print global map
    else:
        plt.contourf(local_map[l].grid[:,:], cmap=cm.Greys)        
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.show()

