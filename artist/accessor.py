import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

@xr.register_dataset_accessor('icon')
class IconAccessor(object):
    def __init__(self, ds):
        self._obj = ds
        
    @property    
    def dz(self):
        dz1 = -1 * self._obj.z_ifc.diff('height_2')
        dz1 = dz1.rename({'height_2':'height'})
        dz1 = dz1.assign_coords(height=(dz1.height - 1))
        return dz1  
    
    def get_modes(self, mode_type='mixed'):
        modes = []
        for vname in list(self._obj.keys()):
            if mode_type in vname:
                modes.append(vname)
        return modes
    
    def add_grid(self, gridfile, ltranslon=True):
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
    
    
    
@xr.register_dataarray_accessor('geo')
class GeoAccessor(object):
    def __init__(self, da):
        self._obj = da
    
    def regrid(self, g, lon, lat, method='linear'):
        rad2deg = 45./np.arctan(1.) 
        clon, clat = g.clon*rad2deg, g.clat*rad2deg
        y, x = np.meshgrid(lat, lon)
        nda = griddata((clon, clat), self._obj.squeeze(), (x, y), method=method)
        nda = xr.DataArray(nda, dims=['Longitude', 'Latitude'], coords=[lon, lat])    
        return nda.T