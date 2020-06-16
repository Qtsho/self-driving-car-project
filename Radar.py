"""
- Grundeinstellungen vornehmen
- Instanz der Klasse NuScenes aus nuscenes.py erstellen
"""

#%matplotlib inline
import Radar_Funktionen # Import der eiegenen Library
import Plot_Funktionen

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import numpy as np

from math import atan, degrees
from sklearn.cluster import DBSCAN



nusc = NuScenes(version='v1.0-mini', dataroot='/home/nuscenes/data/sets/nuscenes', verbose=True)
# change the dataroot (home/nuscenes ergänzt)

# der vordere Radar soll für die Verarbeitung genutzt werden
sensor = "RADAR_FRONT"
dataroot = "/home/nuscenes/data/sets/nuscenes/"

"""
Aus dem Mini-Dataset die ausgewählte Scene (scene_0757) einlesen
"""

test_scene = nusc.scene[4]
test_scene # auflisten

"""
Folglich werden Samples, Sampledaten, Sensordaten und entsprechende
Filenames herausgesucht, damit die Scene vollständig abgebildet
bzw. weiterverarbeitet werden kann
"""

# den ersten und letzten Sample_Token aus der Scene auslesen
first_sample_token = test_scene["first_sample_token"]
last_sample_token = test_scene["last_sample_token"]

scene_samples, nbr_samples = Radar_Funktionen.get_all_scene_samples(nusc, first_sample_token, last_sample_token)

# Filenames der RADAR-FRONT Samples auslesen
data_info, sample_datas, data_from_samples = Radar_Funktionen.get_data_from_sample(nusc, scene_samples)


# Die PointClouds der Scene (Scene-Samples) auslesen
scene_points_list = Radar_Funktionen.radar_points_from_cloud(data_info)









"""
# aus den Samples werden die Daten [data] des entsprechenden
# CAM_FRONT wird auch ausgelesen um die passenden jpg zu haben 

vid_data_test_sample_first = test_sample_first["data"]["CAM_FRONT"]
vid_data_test_sample_last = test_sample_last["data"]["CAM_FRONT"]

vid_sample_data_test_sample_first = nusc.get("sample_data", vid_data_test_sample_first)
vid_sample_data_test_sample_last = nusc.get("sample_data", vid_data_test_sample_last)
"""




# EXTRA EINGEFÜGT (https://forum.nuscenes.org/t/differenes-between-radar-point-cloud-data-and-processed-data/178)
# hier werden irgendwie die bereits erkannten Objekte angezeigt

"""
# Video darstellen:
from PIL import Image

video_first = dataroot+vid_filename_first
video_last = dataroot+vid_filename_last

im_first = Image.open(video_first)
im_last = Image.open(video_last)

im_first.show()
im_last.show()

Es muss leider selbstständig geschlossen werden, dient daher nur
zu Testzwecken

im_first.close() # close gibt
im_last.close()  # nur Pointer frei
"""


