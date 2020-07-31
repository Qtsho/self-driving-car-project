#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Justin
FH Dortmund
"""
from nuscenes.nuscenes import NuScenes
import numpy as np

import matplotlib.pyplot as plt # just for
import matplotlib.ticker as ticker # visualization

import radar_processing_utils
import ego_vehicle_utils
import classification_utils

""" Default Settings """
dataroot = "/home/nuscenes/data/sets/nuscenes/"


""" Tien Beginning """
#parser = arg

""" Radar processing """

nusc_ = NuScenes(version='v1.0-mini', dataroot = dataroot, verbose=True)

my_scene = nusc_.scene[0] # das passt nicht mit my_scene_token zusammen, mit Tien reden

my_scene_token = nusc_.field2token("scene", "name", "scene-0757") # Token von scene[1], urspr.= "scene-0103"
print("\nScene token:", my_scene_token)

# Query loading all sample token in once scene (2 Hz)
sample_tokens = nusc_.field2token("sample", "scene_token", my_scene_token[0])

sample_record = nusc_.get("sample", sample_tokens[30]) # den ersten Sample auslesen
sample_data_record_radar = nusc_.get("sample_data", sample_record["data"]["RADAR_FRONT"]) # Sample_data des ersten Sample auslesen (für Radar)
first_pose_record_radar = nusc_.get("ego_pose", sample_data_record_radar["ego_pose_token"])


""" FOR TESTING """
ego_velo_TEST, yaw_rate_TEST = ego_vehicle_utils.VehicleSpeed.calculate_velocity(nusc_, sample_data_record_radar)
pcd_path_TEST = dataroot + sample_data_record_radar["filename"]
radar_pcls_TEST = radar_processing_utils.Radar.get_radar_points(pcd_path_TEST)
detecs_distance_TEST, detecs_phi_TEST, detecs_radial_velocity_TEST = radar_processing_utils.Radar.calculation_of_radar_data(radar_pcls_TEST)
# ego_velo_TEST_comp = np.mean(np.sqrt((radar_pcls_TEST.points[6] - radar_pcls_TEST.points[8])**2 + (radar_pcls_TEST.points[7] - radar_pcls_TEST.points[9])**2))
moving_coords_TEST = classification_utils.Classify.classify_detections(detecs_distance_TEST, detecs_phi_TEST, detecs_radial_velocity_TEST, ego_velo_TEST)

# hier die Classi aufrufen

""" I NEED SPECIAL RADAR_SAMPLE_DATA_RECORD !!!!!"""

""" THIS NEEDS TO BE DELETED, ITS JUST FOR TESTING"""
for i in range(len(sample_tokens)):
    
    sample_record = nusc_.get("sample", sample_tokens[i])
    sample_data_record_radar = nusc_.get("sample_data", sample_record["data"]["RADAR_FRONT"])
    ego_velo, yaw_rate = ego_vehicle_utils.VehicleSpeed.calculate_velocity(nusc_, sample_data_record_radar)
    
    pcd_path = dataroot + sample_data_record_radar["filename"]
    radar_pcls = radar_processing_utils.Radar.get_radar_points(pcd_path)
    
    detecs_distance, detecs_phi, detecs_radial_velocity = radar_processing_utils.Radar.calculation_of_radar_data(radar_pcls)
    
    
    if i == 0:
        ego_v_array = ego_velo
        yaw_r_array = yaw_rate
        
    else:
        ego_v_array = np.vstack((ego_v_array, ego_velo))
        yaw_r_array = np.vstack((yaw_r_array, yaw_rate))
    

""" 
JUST FOR VISUALIZATION 
YOU CAN DELETE OR IGNORE IT
"""
# aktuellen Plot zuweisen
ax = plt.gca()
   
# obere und rechte Achse unsichtbar werden lassen
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")   
    
# untere Achse auf die y=0 Position bewegen
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
    
ax.xaxis.set_major_locator(ticker.MultipleLocator(10)) # kleine "Striche", x-Achse (untere)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
# linke Achse auf die Position x == 0 bewegen
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))
    
ax.set_title("Detections")
    
"""
for direction in ["xzero", "yzero"]:
# adds arrows at the ends of each axis
ax.axis[direction].set_axisline_style("-|>")
"""
        
#ax.xaxis.set_axisline_style("->", size=1.5)
    
# x und y Werte der Points plotten
# i wird hier als Index verwendet, damit beim Aufruf der Fkt. der zu plottende
# Sample ausgewählt werden kann
# Die y-Werte werden hier zuerst aufgerufen, da diese über die x-Achse verlaufen (Conti-Radar)
ax.plot(np.array(radar_pcls_TEST.points[1]),np.array(radar_pcls_TEST.points[0]), "r.")
   
# x-Achse (Bei Radar = y-Achse) limitieren, sodass Plots verglichen werden können
#ax.set_xlim(60.0, -60.0)
    
#plt.xlabel("Latitude (y, in m)")
ax.set_xlabel("Latitude (y, in m)")
#plt.ylabel("Longitude (x, in m)")
#ax.set_ylabel("Longitude (x, in metersen)", labelpad=120)
ax.set_ylabel("Longitude (x, in Meter)")
    
ax.yaxis.set_label_coords(0, 0.5)
plt.grid(True)
plt.show()