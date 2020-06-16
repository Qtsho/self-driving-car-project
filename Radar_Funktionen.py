#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:20:22 2020

@author: Justin
"""


""" Importierungen """

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.cluster import DBSCAN
#from math import atan, degrees, sqrt

import Plot_Funktionen



""" Wichtige Grundparameter """

dataroot = "/home/nuscenes/data/sets/nuscenes/"
#nusc = NuScenes(version='v1.0-mini', dataroot='/home/nuscenes/data/sets/nuscenes', verbose=True)





""" _____________________________ """
""" Selbstgeschriebene Funktionen """
""" _____________________________ """



def get_all_scene_samples(nusc, first_sample_token, last_sample_token):
    """
    
    Funktion um alle Samples aus einer Scene zu ziehen. Mit diesen
    Samples können dann die im weiteren Verlauf weitere Daten geholt
    werden. Erzeugt wurde diese Funktion um RADAR-Daten (Front_Radar,
    NuScenes) auslesen zu können.
    
    :nusc = Instanz der Klasse NuScenes (NuScenes.py)
    :first_sample_token = nusc.scene[x]["first_sample_token"]
    :last_sample_token = nusc.scene[x]["last_sample_token"]
    
    """
    
    first_sample = nusc.get("sample", first_sample_token)
    last_sample = nusc.get("sample", last_sample_token)
    
    scene_samples = np.empty(0) # Array erstellen, in welches alle Samples geschrieben werden
    current_sample = first_sample
    
    while current_sample != last_sample:
        scene_samples = np.append(scene_samples, current_sample["token"])
        next_sample = current_sample["next"]
        
        if next_sample == "":
            break
        
        # nusg.get gibt ein Dict aus
        current_sample = nusc.get("sample", next_sample) 

    scene_samples = np.append(scene_samples, last_sample["token"]) # letzte Sample anfügen
    nbr_samples = len(scene_samples)
    print("\nEs wurden " + str(len(scene_samples)) + " gefunden.\n")
    
    return scene_samples, nbr_samples



def get_data_from_sample(nusc, scene_samples, wanted_info_or_sensor: str = "RADAR_FRONT", wanted_sample_data_info: str = "filename"):
    """
    Mit dieser Funktion kann von einem Sample die gewünschte
    Information gezogen werden.
    Anschließend werden die Filenames noch ausgeben
    
    :nusc = Instanz der Klasse NuScenes (NuScenes.py)
    :scene_samples = aus get_all_scene_samples erhaltenes Array
    :wanted_info_or_sensor = z.B. 'RADAR_FRONT' (siehe 'data' von 'samples')
    :wanted_sample_data_info = z.B. 'filename' (siehe 'sample_data')
    Weitere Beispiele siehe Doku
    """

    length_samples = len(scene_samples)
    data_from_samples = [0] * length_samples  # hier Liste, da im weiteren Verlauf mit strings gearbeitet wird
    sample_datas = [0] * length_samples     # auch hier eine Liste
    data_info = [0] * length_samples        # Liste für Informationen aus sample_data 
    
    for i in range(length_samples):
        
        """ Aus samples.json die gewünschte data-Information lesen """
        # die ausgelesenen data-Informationen sind Strings
        data_from_samples[i] = nusc.get("sample", scene_samples[i])["data"][wanted_info_or_sensor]
        
        """ Samples_Data aus sample_data.json lesen """
        # sample_datas ist eine Liste, in der sich folglich dicts befinden
        sample_datas[i] = nusc.get("sample_data", data_from_samples[i])
        
        """ Gewünschte Information (z.B. filename) aus den sample_data ziehen """
        # Hier wird aus dem Dict (sample_data) die gewünschte Information gelesen
        data_info[i] = sample_datas[i][wanted_sample_data_info]
        
    return data_info, sample_datas, data_from_samples



def radar_points_from_cloud(filename, result_as_array=False):
    """
    Mit dieser Funktion kann bei angegebenen filename eine RADAR-PointCloud
    gelesen werden. Diese werden dann in ein ARRAY? geschrieben und für die
    Weiterverarbeitung returned
    
    :filename = Pfad der PCL-Datei, data_info aus get_data_from_sample(..wanted_sample_data_info = "filename")
    """
    
    number_of_samplefiles = len(filename)
    
    # Zunächst die Filter disablen, damit man alle Radardaten bekommt
    RadarPointCloud.disable_filters()
    
    #scene_points_array = np.empty(number_of_samplefiles)
    scene_points_list = [] # List für die Point-Arrays (je Array 18 x bis 125)
    #scene_points_as_array = np.zeros(number_of_samplefiles)
    
    for i in range(number_of_samplefiles):
        
        # print(filename[i]) wird zum debuggen benutzt
        path = dataroot + filename[i]
        
        # trotz des Aufrufs von disable_filters werden hier die Filter (nochmals) definiert
        point_of_current_file = RadarPointCloud.from_file(path,
                                                          invalid_states=list(range(17+1)),
                                                          dynprop_states=list(range(7+1)),
                                                          ambig_states=list(range(4+1)))
        
        if result_as_array:
        # Punkte in Array anfügen        
            scene_points_as_array = np.vstack()
            
        if not result_as_array:
            # Default wird Liste erstellt
            scene_points_list.append(point_of_current_file.points)
            
            
    if result_as_array:
        return scene_points_as_array
    
    else:
         return scene_points_list



def calculate_distance(scene_points, distance_unit="m", result_as_array=False):
    """
    Mit dieser Funktion kann aus PointCloudDaten (radar_points_from_cloud)
    die Distanz zum Objekt berechnet werden.
    
    Points haben 18 Zeilen, die wichtigsten:
        
    x y z . . . vx vy vx_comp vy_comp
    1 2 3       6  7     8       9
    
    Distanz = sqrt(x**2 + y**2)
    :scene_points = aus radar_points_from_cloud erhaltenes Array/List (scene_points_list)
    :distance_unit = Einheit in der Distanz berechnet und ausgegeben werden soll (m (Default), mi, km)
    :result_as_array = Wenn True, dann returnt die Funktion ein Array anstelle einer Liste (Default = False)
    
    : return =  [List mit floats] Liste die (AnzahlSamples)Zeilen und (AnzahlMessungenProSample) Reihen hat
                Es werden die einzelnen Distancen der Messungen ausgegeben 
    """
    
    # Je nach gewünschter Einheit wird der Faktor angepasst

    if (distance_unit == "m"):
        distance_factor = 1
        
    elif (distance_unit == "mi"):
        distance_factor = 1/1609
        
    elif (distance_unit == "km"):
        distance_factor = 1/1000
        
    
    distances_of_scene = []
    
    # es gibt pro Scene um die 40 Samples (len(scene_points), 2 Hz).
    # Diese werden hier nacheinander abgearbeitet
    for cur_nbr_sample in range(len(scene_points)):
        
        dist_per_sample = []
        
        # es können bis zu 125 Punkte/Messungen vorhanden sein
        for cur_nbr_measurement in range(len(scene_points[cur_nbr_sample][0])):
            
            # Liste hat 'Anzahl Scene_points' Zeilen und 'Anzahl Messungen' Spalten
            # Berechnung der Distanz (Pythagoras)
            dist_per_measure = (
                
                math.sqrt(
                            scene_points[cur_nbr_sample][0][cur_nbr_measurement]**2
                            +
                            scene_points[cur_nbr_sample][1][cur_nbr_measurement]**2
                        )
                * distance_factor
            )
            
            dist_per_sample.append(dist_per_measure) # pro Sample sollen (bis zu 125) Distanzen in eine Liste geschrieben werden
                
        distances_of_scene.append(dist_per_sample) # Liste, insgesamt gibt es ca. 40 Reihen/Samples, die je bis 125 Distanzen haben
        
        # wenn die Ausgabe ein Array sein soll (Format jedoch komisch, evtl. verbesserungswürdig)
        if result_as_array:
            distances_of_scene = np.asarray(distances_of_scene)


    return distances_of_scene


def calculate_velocity(scene_points, velocity_unit="m/s", result_as_array=False):
    """
    Mit dieser Funktion kann aus PointCloudDaten (radar_points_from_cloud)
    die Radialgeschwindigkeit zum Objekt berechnet werden.
    
    Da NuScenes bereits vom Radarsensor verarbeitete Daten ausgibt, muss bei der Radialgeschwindigkeit
    "getrickst" werden. vy ist meist 0 bzw. nan, weshalb hier unterschieden werden muss. Grund ist, dass
    Continental davon ausgeht, dass Ziele sich in gleicher Richtung bzw. entgegekommend zu uns bewegen.
    
    Points haben 18 Zeilen, die wichtigsten:
        
    x y z . . . vx vy vx_comp vy_comp
    1 2 3       7  8     9       10
    
    vx = ...[6]
    vy = ...[7]
    
    Radialvelocity = sqrt(vx**2 + vy**2)
    :scene_points = aus radar_points_from_cloud erhaltenes Array/List (scene_points_list)
    :velocity_unit = Einheit in der Distanz berechnet und ausgegeben werden soll (m/s (Default), mi/h, km/h)
    :result_as_array = Wenn True, dann returnt die Funktion ein Array anstelle einer Liste (Default = False)
    
    : return =  [List mit floats] Liste die (AnzahlSamples)Zeilen und (AnzahlMessungenProSample) Reihen hat
                Es werden die einzelnen Radialgeschwindigkeiten der Messungen ausgegeben 
    """
    
    # Je nach gewünschter Einheit wird der Faktor angepasst

    if (velocity_unit == "m/s"):
        velocity_factor = 1
        
    elif (velocity_unit == "mi/h"):
        velocity_factor = 2.237
        
    elif (velocity_unit == "km/h"):
        velocity_factor = 3.6
        
    
    velocities_of_scene = [] # empty List for all velocities of one Scene (~40 Samples)
    
    # es gibt pro Scene um die 40 Samples (len(scene_points), 2 Hz).
    # Diese werden hier nacheinander abgearbeitet
    for cur_nbr_sample in range(len(scene_points)):
        
        
        # NAN-CHECK: Hier wird gecheckt, ob in vy nan-Werte stehen.
        # Diese werden ggf. korrigiert bzw. für den weiteren Verlauf 0 gesetzt
        for i in range(len(scene_points[cur_nbr_sample][7])):
            
            if math.isnan(scene_points[cur_nbr_sample][7][i]):
                scene_points[cur_nbr_sample][7][i] = 0
         
        # NAN-CHECK ENDE
                
        velo_per_sample = []
        
        
        # Liste hat (Anzahl Scene_Samples/Points) Zeilen und Anzahl_Messungen Spalten
        for cur_nbr_measurement in range(len(scene_points[cur_nbr_sample][6])):
            
            # Berechnung der Geschwindigkeit (Pythagoras)
            velo_per_measure = (
                
                math.sqrt(
                            scene_points[cur_nbr_sample][6][cur_nbr_measurement]**2
                            +
                            scene_points[cur_nbr_sample][7][cur_nbr_measurement]**2
                        )
                * velocity_factor
            )
            
            velo_per_sample.append(velo_per_measure)
            
        velocities_of_scene.append(velo_per_sample)
        
        ###### HIER NOCH RETURN, WENN VELOS ALS ARRAY GEWÜNSCHT 
        
    return velocities_of_scene
        


def calculate_angle(scene_points, angle_unit="rad", result_as_array=False):
    """
    Mit dieser Funktion können aus PointCloudDaten (radar_points_from_cloud)
    die Winkel zu Objekten berechnet werden.
    
       
    Points haben 18 Zeilen, die wichtigsten:
        
    x y z . . . vx vy vx_comp vy_comp
    1 2 3       7  8     9       10
    
    x  = ...[0]
    y  = ...[1]
    vx = ...[6]
    vy = ...[7]
    
    Winkel = arctan(y / x) 180/pi

    :scene_points = aus radar_points_from_cloud erhaltenes Array/List (scene_points_list)
    :angle_unit = Format mit dem der berechnet und ausgegeben werden soll (degree/Grad (°, Default), rad/Bogenamß)
    :result_as_array = Wenn True, dann returnt die Funktion ein Array anstelle einer Liste (Default = False)
    
    : return =  [List mit floats] Liste die (AnzahlSamples)Zeilen und (AnzahlMessungenProSample) Reihen hat
                Es werden die einzelnen Winkel der Messungen ausgegeben 
    """
    
    # Je nach gewünschter Einheit wird der Faktor angepasst

    if (angle_unit == "degree"):
        degree_factor = True
        
    elif (angle_unit == "rad"):
        degree_factor = False

        
    
    angles_of_scene = [] # empty List for all Angle of one Scene (~40 Samples)
    
    
    for cur_nbr_sample in range(len(scene_points)):
        
        angle_per_sample = []
        
        for cur_nbr_measurement in range(len(scene_points[cur_nbr_sample][0])):
            
            # falls eine Winkelberechnung erfolgen soll (Default True)
            if degree_factor:
                angle_per_measure = (
                    
                    # von Rad in Deg
                    np.rad2deg( 
                    np.arctan(scene_points[cur_nbr_sample][1][cur_nbr_measurement]
                              /
                              scene_points[cur_nbr_sample][0][cur_nbr_measurement]
                              )
                        )
                    )
            
            else:
                angle_per_measure = (
                    
                    np.arctan(scene_points[cur_nbr_sample][1][cur_nbr_measurement]
                              /
                              scene_points[cur_nbr_sample][0][cur_nbr_measurement]
                              )              
                    )
            
            angle_per_sample.append(angle_per_measure)
            
        angles_of_scene.append(angle_per_sample)
            
    return angles_of_scene

def calculate_radial_velocity(scene_points, angles_of_scene, velocity_unit="m/s"):
    
    # Je nach gewünschter Einheit wird der Faktor angepasst

    if (velocity_unit == "m/s"):
        velocity_factor = 1
        
    elif (velocity_unit == "mi/h"):
        velocity_factor = 2.237
        
    elif (velocity_unit == "km/h"):
        velocity_factor = 3.6

    
    velocities_of_scene = []
    
    
    for cur_nbr_sample in range(len(scene_points)):
        
        
        # NAN-CHECK: Hier wird gecheckt, ob in vy nan-Werte stehen.
        # Diese werden ggf. korrigiert bzw. für den weiteren Verlauf 0 gesetzt
        for i in range(len(scene_points[cur_nbr_sample][7])):
            
            if math.isnan(scene_points[cur_nbr_sample][7][i]):
                scene_points[cur_nbr_sample][7][i] = 0
         
        # NAN-CHECK ENDE
                
        velo_per_sample = []
        
        
        # Liste hat (Anzahl Scene_Samples/Points) Zeilen und Anzahl_Messungen Spalten
        for cur_nbr_measurement in range(len(scene_points[cur_nbr_sample][6])):
            
            # Berechnung der Radialgeschwindigkeit mit der Annahme, dass
            # vy = 0 ist. ---> V_rad = cos(phi) * vx | phi als rad
            velo_per_measure = (
                
                (
                np.cos(angles_of_scene[cur_nbr_sample][cur_nbr_measurement])
                *
                scene_points[cur_nbr_sample][6][cur_nbr_measurement]
                )
                
                * velocity_factor
            )
            
            velo_per_sample.append(velo_per_measure)
            
        velocities_of_scene.append(velo_per_sample)
        
        ###### HIER NOCH RETURN, WENN VELOS ALS ARRAY GEWÜNSCHT 
        
    return velocities_of_scene
    
    

def calculate_ego_velocity(nusc, sample_datas, timestamplist, velocity_unit="m/s"):
    """
    Mit dieser Funktion wird aus der Ego_Pose (ego_pose.json) mit den passenden
    Sample-Zeitstempeln (timestamp, aus Sample_data) die Geschwindigkeit des
    Autos berechnet.
    
    :sample_datas = Aus 'get_data_from_sample'
    """
    
    timestamp_list = []
    translation_list = []
    
    
    for i in range(len(sample_datas)):
        
        # die einzelnen Zeitstempel der Messungen raussuchen
        current_timestemp = sample_datas[i]["timestamp"]
        timestamp_list.append(current_timestemp)
        
        current_ego_pose_token = sample_datas[i]["ego_pose_token"]
        current_translation = nusc.get("ego_pose", cur_ego_pose_token)["translation"]
        translation_list.append(current_translation)
        
    ####################### HIER NOCH WEITERMACHEN
        
        
def clustering(scene_points):
    
    # zunächst die x- und y-Werte richtig in ein Array schreiben    ( y0 | x0 )
    # siehe Matrixstruktur                                          ( y1 | x1 )
    
    
    # zunächst wird array erstellt mit: ( x1 x2 x3 ... )
    #                                   ( y1 y2 y3 ... )
    array = np.vstack([scene_points[Index_des_Samples][1], scene_points[Index_des_Samples][0]])
    
    # jetzt transponieren
    transpose_array = array.transpose()
    
    # mal plotten lassen
    plt.scatter(transpose_array[:, 0], transpose_array[:, 1])