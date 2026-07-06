import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from .__plot import (
    plot_hov as _plot_hov,
    plot_slice as _plot_slice,
    quick_plot as _quick_plot,
    show_slice_line as _show_slice_line,
    tri_data as _tri_data,
    tri_plot as _tri_plot,
)
from .utils import add_grid

@xr.register_dataset_accessor('icon')
class IconAccessor(object):
    def __init__(self, ds):
        self._obj = ds
        self._tree = None
        self._grid_points = None

    def add_grid(self, gridfile, ltranslon=False):
        rad2deg = 180.0 / np.pi
        g = xr.open_dataset(gridfile)

        vlon = g.clon_vertices * rad2deg
        vlat = g.clat_vertices * rad2deg
        clon = g.clon * rad2deg
        clat = g.clat * rad2deg

        if ltranslon:
            vlon = (vlon + 360) % 360
            clon = (clon + 360) % 360

        self.vlon, self.vlat = vlon, vlat
        self.clon, self.clat = clon, clat

        cell_dim = "ncells" if "ncells" in self._obj.dims else clon.dims[0]
        clon_values = np.asarray(clon.values)
        clat_values = np.asarray(clat.values)

        if cell_dim in self._obj.dims and self._obj.sizes[cell_dim] == clon_values.size:
            self._obj.coords["clon"] = (cell_dim, clon_values)
            self._obj.coords["clat"] = (cell_dim, clat_values)

        self._grid_points = np.column_stack((clon_values, clat_values))
        self._tree = None
        return g

    def get_dz(self):
        z_ifc = self._obj.z_ifc
        height_dim = "height_2" if "height_2" in z_ifc.dims else "height_3"
        dz = -z_ifc.diff(height_dim).rename({height_dim: "height"})
        return dz.assign_coords(height=(dz.height - 1))

    def look_up(self, key="mixed"):
        return [vname for vname in self._obj.data_vars if key in vname]

    def nearest_gridpoints(self, coordinate):
        """
        Vectorized nearest-neighbor mapping between lon/lat points and ICON cells.
        """
        if self._grid_points is None:
            self._grid_points = np.column_stack((np.asarray(self.clon.values), np.asarray(self.clat.values)))

        coords = np.asarray(coordinate, dtype=float)
        coords = np.atleast_2d(coords)

        try:
            from scipy.spatial import cKDTree

            if self._tree is None:
                self._tree = cKDTree(self._grid_points)
            _, gridpoints = self._tree.query(coords)
            return gridpoints.astype(int)
        except Exception:
            # Fallback without scipy: vectorized distance calculation in chunks.
            chunksize = 256
            out = np.empty(coords.shape[0], dtype=int)
            for start in range(0, coords.shape[0], chunksize):
                stop = start + chunksize
                diff = self._grid_points[None, :, :] - coords[start:stop, None, :]
                out[start:stop] = np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
            return out

    show_slice_line = _show_slice_line

        
@xr.register_dataarray_accessor('icon')
class GridAccessor(object):
    def __init__(self, da):
        self._obj = da
        
    def regrid(self, gridfile, lon, lat, method='linear', ltranslon=True):
        _, _, _, clon, clat = add_grid(gridfile, ltranslon=ltranslon)
        y, x = np.meshgrid(lat, lon)
        nda = griddata((clon, clat), self._obj.squeeze(), (x, y), method=method)
        nda = xr.DataArray(nda, dims=['Longitude', 'Latitude'], coords=[lon, lat])   
        return nda.T

    tri_data = _tri_data
    tri_plot = _tri_plot
    
@xr.register_dataarray_accessor('art')
class ArtAccessor(object):
    _grid_cache = {}

    def __init__(self, da):
        self._obj = da

    @staticmethod
    def _as_index_array(values):
        return np.asarray(values, dtype=int).ravel()

    def tracer_load(self, rho, dz):
        """
        Column tracer load.

        This keeps the operation lazy for dask-backed xarray objects and avoids
        creating extra intermediate variables in Python.
        """
        return (self._obj * rho * dz).sum("height")

    def select_plume(self, thresh, fill_value=0):
        """
        Select plume cells where tracer concentration is above a threshold.
        """
        return self._obj.where(self._obj > thresh).fillna(fill_value)

    def plume_center(self, v_cell, z_mc, dim=None, skipna=True):
        """
        Height of the plume center of mass.
        """
        tracer_mass = self._obj * v_cell
        weighted_height = tracer_mass * z_mc
        return weighted_height.sum(dim=dim, skipna=skipna) / tracer_mass.sum(dim=dim, skipna=skipna)

    def plume_top(self, z_mc, height_dim="height", fill_value=0):
        """
        Height of the highest plume cell at each horizontal location.
        """
        z_plume = z_mc.where(self._obj > 0)
        return z_plume.max(dim=height_dim).fillna(fill_value)

    def plume_bottom(self, z_mc, height_dim="height", fill_value=0):
        """
        Height of the lowest plume cell at each horizontal location.
        """
        z_plume = z_mc.where(self._obj > 0)
        return z_plume.min(dim=height_dim).fillna(fill_value)

    def value_at_plume_top(self, z_mc, val, height_dim="height", fill_value=0):
        """
        Value of another field at the plume top.
        """
        z_plume = z_mc.where(self._obj > 0)
        top = z_plume.max(dim=height_dim)
        return val.where(z_plume == top).max(dim=height_dim).fillna(fill_value)

    def max_conc_height(self, z_mc, height_dim="height", fill_value=0):
        """
        Height where plume concentration is maximal.
        """
        plume = self._obj.where(self._obj > 0)
        z_max = z_mc.where(plume == plume.max(dim=height_dim, skipna=True))
        return z_max.mean(dim=height_dim, skipna=True).fillna(fill_value)

    def make_slice_time(self, height, gridpoint, start_t=0, end_t=1, height_level_max=30):
        """
        Build vectorized arrays for a time-height plot at one native gridpoint.
        """
        gridpoint = int(np.asarray(gridpoint).ravel()[0])
        time_sel = slice(start_t, end_t)
        height_sel = slice(-height_level_max - 1, -1)

        height_slice = height.isel(height=height_sel, ncells=gridpoint, time=time_sel)
        tracer_slice = self._obj.isel(height=height_sel, ncells=gridpoint, time=time_sel)

        if height_slice.sizes.get("time", 0) == 0:
            raise ValueError("No timesteps selected. Check start_t and end_t.")

        height_vals = height_slice.transpose("time", "height").values
        tracer_vals = tracer_slice.transpose("time", "height").values

        timesteps = np.arange(start_t, end_t)
        timesteps_expanded = np.repeat(timesteps, height_vals.shape[1])
        y_axis = height_vals.ravel()
        dust = tracer_vals.ravel()
        ground = height.isel(height=-1, ncells=gridpoint, time=start_t).values
        return timesteps_expanded, y_axis, dust, timesteps, ground

    plot_hov = _plot_hov

    def make_slice(self, height, gridpoints, t, height_level_max=30):
        """
        Build vectorized arrays for a vertical slice from selected native cells.
        """
        gridpoints = self._as_index_array(gridpoints)
        height_sel = slice(-height_level_max - 1, -1)

        height_slice = height.isel(height=height_sel, ncells=gridpoints, time=t)
        tracer_slice = self._obj.isel(height=height_sel, ncells=gridpoints, time=t)

        y_axis = height_slice.transpose("ncells", "height").values.ravel()
        dust = tracer_slice.transpose("ncells", "height").values.ravel()
        x_data = np.arange(gridpoints.size)
        x_axis = np.repeat(x_data, height_slice.sizes["height"])
        ground = height.isel(ncells=gridpoints, time=t, height=-1).values
        return x_axis, y_axis, dust, x_data, ground, t

    plot_slice = _plot_slice

    quick_plot = _quick_plot
