# FH Dortmund.
# Code written by Tien Tran, 2020.
import numpy as np
import matplotlib.pyplot as plt
import time
import bresenham
import inverse_sensor_model as inv
from tqdm import tqdm
from typing import Tuple, List

class Map:

    def __init__(self, lenght: float= 100.0, 
                        width: float= 100.0, 
                         resolution: float= 0.1):
        #create empty 100*100 grid in log odd form

        # in LogOdds Notation from octomap
        loccupied = 0.85 #0.7p
        lfree = -0.4     #0.4 p  
        lmin = -2.0      #0.1p  
        lmax = 3.5       #
        self.lenght = lenght
        self.width = width
        self.resolution = resolution

    # p = np.arange(0.01, 1.0, 0.01)
    # lo = np.log(p/(1-p))

    # plt.plot(p, lo)
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('Probability $p$')
    # plt.ylabel(r'Log Odds, $\log(\frac{p}{1-p})$')

        print ("%ix%i Grid" % (self.lenght/resolution, self.width/resolution))
    
    # Log Odds Grid must be initialized with zeros! p = 0.5
        self.grid = np.zeros((int(self.lenght/resolution), int(self.width/resolution)), dtype=np.float32) 

        print ("Create a Occupancy grid of %.2fGB" % (self.grid.nbytes/1024.0**2))
    
    def mapUpdate(self,
                sensor_position,
                lidar_pcls_mod,
                pos_x,
                pos_y):
        #loop through all measurements
        xe = lidar_pcls_mod[0] + pos_y
        min_x = np.amin(xe)
        ye = lidar_pcls_mod[1] + pos_x
        min_y = np.amin(ye)

        xe += abs(min_x) 
        ye += abs(min_y)
        #This cause pronlem sensor_position = [int(abs(min_x)),int(abs(min_y))]        
        for i in tqdm(range(len(lidar_pcls_mod[0]))):
        #get end point coordinate in interger
            x = int(xe[i]) # switch coordinate to loop through the nparray
            y = int(ye[i])
            #Have condition first when pre-processing
            if y >= (self.lenght/self.resolution) or  x >= (self.width/self.resolution): 
                continue #measure is out of grid, skip
            self.grid[x,y] += inv.locc
            
        #calculate the line between sensor and z using bresenham
            l = bresenham.bresenham(sensor_position,[x,y])
            if self.grid[x,y]>inv.lmax: #clamping
                self.grid[x,y]=inv.lmax

            for (j,k) in l.path: 
                if j >= (self.lenght/self.resolution) or  k >= (self.width/self.resolution): 
                    continue #measure is out of grid, skip
                self.grid [j,k] += inv.lfree
                if  self.grid[j,k]<inv.lmin: #min
                     self.grid[j,k]=inv.lmin
    