# FH Dortmund.
# Code written by Tien Tran, 2020.
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
lmin = -2.0
lmax = 3.5
locc =np.log(0.7 / (1 - 0.7))
lfree = np.log(0.4 / (1 - 0.4))
def log_inv_sensor_model(z: np.ndarray, c: np.ndarray):
    
    #c : cell in grid map to be considered
    #z: measurment from range sensor
    rz = np.sqrt(np.square(z[0])+np.square(z[1]))
    rc = np.sqrt(np.square(c[0])+np.square(c[1]))
    if rc > rz:
        # The sensor detects a wall for this cell
        return np.log(0.7 / (1 - 0.7))
    # The sensor detects free space for this cell
    return np.log(0.4 / (1 - 0.4))

