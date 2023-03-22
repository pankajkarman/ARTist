import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from   matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

from .utils import add_grid

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
    
    
    
@xr.register_dataarray_accessor('icon')
class GridAccessor(object):
    def __init__(self, da):
        self._obj = da
        
    def regrid(self, gridfile, lon, lat, method='linear', ltranslon=True):
        g, _, _, _, _ = add_grid(gridfile, ltranslon=ltranslon)
        rad2deg = 45./np.arctan(1.) 
        clon, clat = g.clon*rad2deg, g.clat*rad2deg
        y, x = np.meshgrid(lat, lon)
        nda = griddata((clon, clat), self._obj.squeeze(), (x, y), method=method)
        nda = xr.DataArray(nda, dims=['Longitude', 'Latitude'], coords=[lon, lat])   
        return nda.T

    def tri_data(self, gridfile, cmap=cm.viridis, vrange=[], ltranslon=True): 
        if len(vrange)>0:
            norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[-1])
        else:
            norm = mpl.colors.Normalize(vmin=self._obj.min(), vmax=self._obj.max())
            
        _, self.vlon, self.vlat, _, _ = add_grid(gridfile, ltranslon=ltranslon)
        cmp = cm.ScalarMappable(norm=norm, cmap=cmap)
        triangles = np.stack((self.vlon, self.vlat), axis=2)
        colors = cmp.to_rgba(self._obj)    
        return triangles, colors, cmp
    
    def tri_plot(self, gridfile, ax, cmap=cm.viridis, vrange=[], ltranslon=False, add_colorbar=True, map_extent=None): 
        triangles, colors, cmp = self.tri_data(gridfile, cmap=cmap, vrange=vrange, ltranslon=ltranslon) 
        coll = PolyCollection(triangles, facecolor=colors, closed=True, edgecolor='face')
        ax.add_collection(coll, autolim=True)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        if add_colorbar:
            plt.colorbar(cmp)
            
        if map_extent:
            ax.set_xlim([map_extent[0], map_extent[1]])
            ax.set_ylim([map_extent[2], map_extent[3]])           
        else:
            ax.set_xlim([self.vlon.min(), self.vlon.max()])
            ax.set_ylim([self.vlat.min(), self.vlat.max()])            
        return cmp