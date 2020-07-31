"""
@author: Justin
FH Dortmund
"""
from nuscenes.utils.data_classes import RadarPointCloud
import numpy as np


class Radar:
    
    
    def __init__(self):
        super().__init__()
        
        
    @classmethod
    def get_radar_points(cls, data_path) -> 'RadarPointCloud':
        
        # To get all points/detections we need to disable settings (..._states)
        radar_pcls = RadarPointCloud.from_file(data_path,
                                               invalid_states = list(range(17+1)),
                                               dynprop_states = list(range(7+1)),
                                               ambig_states = list(range(4+1)))
        
        # radar_points_array = radar_pcls.points
        
        return radar_pcls
    
    
    
    @classmethod
    def calculation_of_radar_data(cls, radar_pcls) -> np.ndarray:
        """
        This method calculates the raw radardata (distance, phi, radial-velocity)
        of every point/detection out of the PointCloud.

        Parameters
        ----------
        radar_pcls : np.ndarray
            PointCloud (RadarPointCloud-Object) from 'get_radar_points'.

        Returns
        -------
        detections_distance : np.ndarray
            An n*1-array with the distances of the radar-detections to the radarsensor (n <= 125).
        detections_phi : np.ndarray
            An n*1-array with the angles of the radar-detections r.t. the radarsensor (n <= 125).
        detections_radial_velocity : np.ndarray
            An n*1-array with the radial velocities of the radar-detections (n <= 125).
            Note that these velocities not consider the velocity of the ego-vehicle.

        """
        x_array = radar_pcls.points[0]
        y_array = radar_pcls.points[1]
        
        vx_array = radar_pcls.points[6]
        
        
        # calculate the distances of the detections to the radar-sensor
        detections_distance = np.sqrt(x_array**2 + y_array**2)
   
        # calculate the angle (phi, degrees) of the detections w.r.t the radar-sensor-frame
        detections_phi = np.rad2deg( np.arctan( y_array / x_array ))
        
        # calculate the radial velocity of the detections to the radar-sensor
        # w.r.t detections_phi (!!!)
        detections_radial_velocity = vx_array * np.cos(np.deg2rad(detections_phi))
        
        
        return detections_distance, detections_phi, detections_radial_velocity
