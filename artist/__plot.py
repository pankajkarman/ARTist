import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cartopy.crs as ccrs

@xr.register_dataarray_accessor('viz')
class PlotAccessor(object):
    def __init__(self, da):
        self._obj = da
    
    def tricontourf(self, ax, cmap='coolwarm', levels=10, add_colorbar=True, map_extent=None, projection=None):
        if projection:
            if projection == ccrs.PlateCarree():
                tcf = ax.tricontourf(self._obj.clon, self._obj.clat, self._obj, cmap=cmap, levels=levels)
            else:
                try:
                    mproj = projection.transform_points(
                        ccrs.PlateCarree(),
                        self._obj.clon,
                        self._obj.clat
                    )
                    x, y =  mproj[:, 0],  mproj[:, 1]
                    tcf = ax.tricontourf(x, y, self._obj, cmap=cmap, levels=levels)
                except:
                    ax.set_global()
                    tcf = ax.tricontourf(self._obj.clon,  self._obj.clat, self._obj,
                            levels=levels,
                            cmap=cmap,
                            transform=ccrs.PlateCarree())

        else:
            tcf = ax.tricontourf(self._obj.clon, self._obj.clat, self._obj, cmap=cmap, levels=levels)

        if add_colorbar:
            cbar = plt.colorbar(tcf, orientation='vertical', pad=0.05)
            try:
                cbar.set_label(self._obj.attrs['standard_name'])
            except:
                pass
        return ax
