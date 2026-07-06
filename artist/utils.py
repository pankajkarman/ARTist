import xarray as xr
import numpy as np

def add_grid(gridfile, ltranslon=True):
    """
    Open an ICON grid file and return grid geometry in degrees.

    Parameters
    ----------
    gridfile : str or path-like
        Path to an ICON grid NetCDF file containing `clon`, `clat`,
        `clon_vertices`, and `clat_vertices` in radians.
    ltranslon : bool, default True
        If True, transform longitudes from the [-180, 180] convention to
        [0, 360].

    Returns
    -------
    tuple
        `(grid, vlon, vlat, clon, clat)`, where `grid` is the opened xarray
        dataset and all coordinate arrays are converted to degrees.

    Examples
    --------
    >>> from artist.utils import add_grid
    >>> grid, vlon, vlat, clon, clat = add_grid("icon_grid.nc")
    """
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
