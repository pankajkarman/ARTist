import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


@xr.register_dataarray_accessor('viz')
class PlotAccessor(object):
    def __init__(self, da):
        self._obj = da
    
    def plot(self):
        print('Wait: Figures are on the way.')
        pass
    
    
    
def add_grid(gridfile, ltranslon=True):
    rad2deg = 45./np.arctan(1.)
    g = xr.open_dataset(gridfile) 
    vlon = g.clon_vertices
    vlat = g.clat_vertices
    vlon, vlat = vlon*rad2deg, vlat*rad2deg
    clon, clat = g.clon*rad2deg, g.clat*rad2deg
    # ncells, nv = vlon.shape[0], vlon.shape[1]
    if ltranslon:
        vlon = (vlon + 360) % 360
        clon = (clon + 360) % 360
    else:
        vlon = vlon
    return g, vlon, vlat, clon, clat