import xarray as xr
import numpy as np

def add_grid(gridfile, ltranslon=True):
    rad2deg = 180.0 / np.pi
    g = xr.open_dataset(gridfile)

    vlon = g.clon_vertices * rad2deg
    vlat = g.clat_vertices * rad2deg
    clon = g.clon * rad2deg
    clat = g.clat * rad2deg

    if ltranslon:
        vlon = (vlon + 360) % 360
        clon = (clon + 360) % 360

    return g, vlon, vlat, clon, clat
