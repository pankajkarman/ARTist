import xarray as xr
import numpy as np


def distance(origin, destination, radius=6371.0):
    """
    Compute great-circle distance between lon/lat coordinates.

    Parameters
    ----------
    origin, destination : array-like
        Coordinate pairs in degrees as ``[lon, lat]``. Inputs may also be
        arrays whose last dimension is ``(lon, lat)``.
    radius : float, default 6371.0
        Spherical Earth radius in kilometers.

    Returns
    -------
    float or numpy.ndarray
        Distance in kilometers.

    Examples
    --------
    >>> from artist.utils import distance
    >>> distance([8.4, 49.0], [13.4, 52.5])
    """
    origin = np.asarray(origin, dtype=float)
    destination = np.asarray(destination, dtype=float)

    lon1, lat1 = np.moveaxis(origin, -1, 0)
    lon2, lat2 = np.moveaxis(destination, -1, 0)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


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
