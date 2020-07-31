#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Justin
FH Dortmund
"""
import numpy as np

import radar_processing_utils
import ego_vehicle_utils

class Classify:
    
    def __init__(self):
        super().__init__()
        
        
    @classmethod
    def classify_detections(cls, detecs_distance, detecs_phi, detecs_radial_velocity, ego_velocity):
        
        v_ego_classi = ego_velocity * np.cos( np.deg2rad( detecs_phi )) # prepare ego_velocity for classification
        
        
        determine_absolute = abs(v_ego_classi + detecs_radial_velocity) # | v_ego * cos(phi) + v_radial |
        
        moving_index = np.where(determine_absolute >= 1) # returns all index where threshold is bigger than 1
        
        
        # delete static detections
        moving_dist = detecs_distance[moving_index] # get distances of moving detections
        moving_phi = detecs_phi[moving_index] # get angle of moving detections
        
        # calculate x- and y-coordiantes of detections
        moving_x_coord = moving_dist * np.cos( np.deg2rad( moving_phi))
        moving_y_coord = moving_dist * np.sin( np.deg2rad( moving_phi))
        
        # put them into one array for return
        moving_coords = np.vstack((moving_x_coord, moving_y_coord))
        
        return moving_coords
