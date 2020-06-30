#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File with class VehicleSpeed to compute the speed/velocity of the NuScenes-
Ego-Vehicle between two frames.
The returned informations of the classmethods can be used to classify object
if they are moving (ambigious or unam)

@author: Justin
FH Dortmund
"""
import numpy as np
import pandas as pd


class VehicleSpeed:
    
    
    def __init__(self):
        super().__init__()


    @classmethod
    def calculate_velocity(cls, nusc_, sample_data_record_radar, use_sweep_for_calculation = True):
        """ 
        Hier wird die Geschwindigkeit des Autos bestimmt. Dazu werd zwei Translationen
        mit Zeitstempel berücksichtigt. |(Trans1 - Trans0)| / (Time1 - Time0)
        Dazu wird aus einem Sample_Data_Record (Tien) der aktuelle UND nächste Token für 
        die jeweiligen Ego_Pose verwendet. Die nächste (next) ist dabei immmer ein Sweep
        
        Estimate the velocity for ego vehicle.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        """
        current_pose = nusc_.get("ego_pose", sample_data_record_radar["ego_pose_token"])
        
        has_prev = sample_data_record_radar["prev"] != "" # Check if the current sample_data has a previous one
        has_next = sample_data_record_radar["next"] != "" # Chick if the current sample_data has a next one
        
        # Put information in arrays for calculation
        current_timestamp = np.array([current_pose["timestamp"]])
        current_translation = np.array([current_pose["translation"]])
        current_rotation = np.array([current_pose["rotation"]])
        
        # calculate yaw-rate of current pose
        w, x, y, z = current_rotation[:, 0], current_rotation[:, 1],current_rotation[:, 2], current_rotation[:, 3]
        current_yaw_rate = np.rad2deg(np.arctan((2*(x*y + w*z))/(w**2 + x**2 - y**2 - z**2)))
        
        # in the following I check, if there are previous and next frames
        if has_next:
            next_sample_data = nusc_.get("sample_data", sample_data_record_radar["next"]) # sweep-information
            next_pose = nusc_.get("ego_pose", next_sample_data["ego_pose_token"])
            
            # get information of next ego_pose
            next_timestamp = np.array([next_pose["timestamp"]])
            next_translation = np.array([next_pose["translation"]])
            
        else:
            next_timestamp = current_timestamp
            next_translation = current_translation
            
        if has_prev:
            prev_sample_data = nusc_.get("sample_data", sample_data_record_radar["prev"]) # sweep-information
            prev_pose = nusc_.get("ego_pose", prev_sample_data["ego_pose_token"])
            
            prev_timestamp = np.array([prev_pose["timestamp"]])
            prev_translation = np.array([prev_pose["translation"]])
            
        else:
            prev_timestamp = current_timestamp
            prev_translation = current_translation
        
        # calculate the driven distance/length between the frames/sweeps
        driven_length = np.linalg.norm(next_translation - prev_translation)
        
        # calculate the past time between the frames/sweeps
        time_diff = (next_timestamp - prev_timestamp) * 1e-6
        
        ego_velocity = driven_length / time_diff
        
        return ego_velocity, current_yaw_rate