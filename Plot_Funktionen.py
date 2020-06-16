#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:06:18 2020

@author: Justin

Library mit Funktionen zum Plotten
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




def plot_detections(scene_points_list, sample_index: int):
    """
    Funktion zum Plotten der (ungeclusterten) Detektionen. Es werden (zunächst)
    nur die x- und y-Werte geplottet um einen Überblick über die gesamten Detektionen
    zu erhalten.
    
    :scene_points_list = Liste der ausgelesenen Point Clouds
    :sample_index = Index des Samples, welcher geplottet werden soll (1 - ~41)
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
    
    """for direction in ["xzero", "yzero"]:
        # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")"""
        
    #ax.xaxis.set_axisline_style("->", size=1.5)
    
    # x und y Werte der Points plotten
    # i wird hier als Index verwendet, damit beim Aufruf der Fkt. der zu plottende
    # Sample ausgewählt werden kann
    # Die y-Werte werden hier zuerst aufgerufen, da diese über die x-Achse verlaufen (Conti-Radar)
    ax.plot(np.array(scene_points_list[sample_index][1]),np.array(scene_points_list[sample_index][0]), "r.")
    
    # x-Achse (Bei Radar = y-Achse) limitieren, sodass Plots verglichen werden können
    ax.set_xlim(60.0, -60.0)
    
    #plt.xlabel("Latitude (y, in m)")
    ax.set_xlabel("Latitude (y, in m)")
    #plt.ylabel("Longitude (x, in m)")
    #ax.set_ylabel("Longitude (x, in metersen)", labelpad=120)
    ax.set_ylabel("Longitude (x, in Meter)")
    
    ax.yaxis.set_label_coords(0, 0.5)
    plt.grid(True)
    plt.show()