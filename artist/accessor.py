import xarray as xr
import numpy as np
import pandas as pd

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
    """
    Dataset-level helpers for ICON/ICON-ART native grid data.

    Accessed as `ds.icon` after importing `artist`.
    """

    def __init__(self, ds):
        self._obj = ds
        self._tree = None
        self._grid_points = None

    def add_grid(self, gridfile, ltranslon=False):
        """
        Attach ICON grid center coordinates to the dataset accessor.

        Parameters
        ----------
        gridfile : str or path-like
            ICON grid NetCDF file containing center and vertex coordinates in
            radians.
        ltranslon : bool, default False
            If True, transform longitudes to the [0, 360] convention.

        Returns
        -------
        xarray.Dataset
            The opened grid dataset. Converted coordinates are stored on the
            accessor as `clon`, `clat`, `vlon`, and `vlat`.

        Examples
        --------
        >>> import artist
        >>> grid = ds.icon.add_grid("icon_grid.nc")
        >>> ds.icon.clon
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
        """
        Compute model-layer thickness from interface heights.

        Returns
        -------
        xarray.DataArray
            Layer thickness on the `height` dimension.

        Examples
        --------
        >>> import artist
        >>> dz = ds.icon.get_dz()
        """
        z_ifc = self._obj.z_ifc
        height_dim = "height_2" if "height_2" in z_ifc.dims else "height_3"
        dz = -z_ifc.diff(height_dim).rename({height_dim: "height"})
        return dz.assign_coords(height=(dz.height - 1))

    def look_up(self, key="mixed"):
        """
        Return dataset variable names containing a search key.

        Parameters
        ----------
        key : str, default "mixed"
            Substring to search for in data variable names.

        Examples
        --------
        >>> import artist
        >>> ds.icon.look_up("ash")
        """
        return [vname for vname in self._obj.data_vars if key in vname]

    def nearest_gridpoints(self, coordinate):
        """
        Find nearest native ICON cells for lon/lat coordinates.

        Parameters
        ----------
        coordinate : array-like
            A single `[lon, lat]` pair or an array with shape `(n, 2)`.

        Returns
        -------
        numpy.ndarray
            Integer native-cell indices nearest to each input coordinate.

        Examples
        --------
        >>> import artist
        >>> ds.icon.add_grid("icon_grid.nc")
        >>> ds.icon.nearest_gridpoints([[13.0, 52.0], [14.0, 53.0]])
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

    def show_slice_line(self, points, gridpoints, grid_stride=5):
        """
        Plot a vertical-slice path and the selected native gridpoints.

        This wrapper keeps `ds.icon.show_slice_line` visible to tab completion;
        the plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import artist
        >>> ds.icon.add_grid("icon_grid.nc")
        >>> points = [[13.0, 52.0], [14.0, 53.0]]
        >>> gridpoints = ds.icon.nearest_gridpoints(points)
        >>> ax = ds.icon.show_slice_line(points, gridpoints)
        """
        return _show_slice_line(self, points, gridpoints, grid_stride=grid_stride)

        
@xr.register_dataarray_accessor('icon')
class GridAccessor(object):
    """
    DataArray-level helpers for ICON native-grid fields.

    Accessed as `da.icon` after importing `artist`.
    """

    def __init__(self, da):
        self._obj = da
        
    def regrid(self, gridfile, lon, lat, method='linear', ltranslon=True):
        """
        Interpolate a native ICON field to a regular lon/lat grid.

        Parameters
        ----------
        gridfile : str or path-like
            ICON grid file matching this DataArray's native cell order.
        lon, lat : array-like
            Target longitude and latitude coordinates.
        method : {"linear", "nearest", "cubic"}, default "linear"
            Interpolation method passed to `scipy.interpolate.griddata`.
        ltranslon : bool, default True
            If True, transform grid longitudes to [0, 360] before regridding.

        Returns
        -------
        xarray.DataArray
            Regridded field with dimensions `Latitude` and `Longitude`.

        Examples
        --------
        >>> import numpy as np
        >>> import artist
        >>> lon = np.linspace(0, 20, 101)
        >>> lat = np.linspace(40, 60, 81)
        >>> regular = da.icon.regrid("icon_grid.nc", lon, lat)
        """
        try:
            from scipy.interpolate import griddata
        except ImportError as exc:
            raise ImportError(
                "SciPy is required for regridding. Install scipy to use "
                "DataArray.icon.regrid()."
            ) from exc

        _, _, _, clon, clat = add_grid(gridfile, ltranslon=ltranslon)
        y, x = np.meshgrid(lat, lon)
        nda = griddata((clon, clat), self._obj.squeeze(), (x, y), method=method)
        nda = xr.DataArray(nda, dims=['Longitude', 'Latitude'], coords=[lon, lat])   
        return nda.T

    def tri_data(self, gridfile, cmap=None, vrange=[], ltranslon=True):
        """
        Build triangle vertices and colors for native-grid plotting.

        The plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import artist
        >>> triangles, colors, cmap = da.icon.tri_data("icon_grid.nc")
        """
        return _tri_data(self, gridfile, cmap=cmap, vrange=vrange, ltranslon=ltranslon)

    def tri_plot(self, gridfile, ax, cmap=None, vrange=[], ltranslon=False, add_colorbar=True, map_extent=None):
        """
        Plot this DataArray on ICON native triangular cells.

        The plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import artist
        >>> fig, ax = plt.subplots()
        >>> da.icon.tri_plot("icon_grid.nc", ax)
        """
        return _tri_plot(
            self,
            gridfile,
            ax,
            cmap=cmap,
            vrange=vrange,
            ltranslon=ltranslon,
            add_colorbar=add_colorbar,
            map_extent=map_extent,
        )
    
@xr.register_dataarray_accessor('art')
class ArtAccessor(object):
    """
    ICON-ART tracer diagnostics for xarray DataArrays.

    Accessed as `da.art` after importing `artist`.
    """

    _grid_cache = {}

    def __init__(self, da):
        self._obj = da

    @staticmethod
    def _as_index_array(values):
        return np.asarray(values, dtype=int).ravel()

    def tracer_load(self, rho, dz):
        """
        Compute column tracer load.

        Parameters
        ----------
        rho : xarray.DataArray
            Air density on the tracer grid.
        dz : xarray.DataArray
            Layer thickness on the `height` dimension.

        This keeps the operation lazy for dask-backed xarray objects and avoids
        creating extra intermediate variables in Python.

        Examples
        --------
        >>> import artist
        >>> dz = ds.icon.get_dz()
        >>> load = ds["ash_mixed_acc"].art.tracer_load(ds["rho"], dz)
        """
        return (self._obj * rho * dz).sum("height")

    def select_plume(self, thresh, fill_value=0):
        """
        Select plume cells where tracer concentration is above a threshold.

        Parameters
        ----------
        thresh : float
            Minimum tracer value considered part of the plume.
        fill_value : scalar, default 0
            Value used outside the plume mask.

        Examples
        --------
        >>> import artist
        >>> plume = ds["ash_mixed_acc"].art.select_plume(1e-9)
        """
        return self._obj.where(self._obj > thresh).fillna(fill_value)

    def plume_center(self, v_cell, z_mc, dim=None, skipna=True):
        """
        Height of the plume center of mass.

        Parameters
        ----------
        v_cell : xarray.DataArray
            Cell volume or other mass-weighting factor.
        z_mc : xarray.DataArray
            Height of model-cell centers.
        dim : str or sequence of str, optional
            Dimensions over which to collapse. Use `None` for all dimensions.
        skipna : bool, default True
            Ignore missing values during reductions.

        Examples
        --------
        >>> import artist
        >>> center = plume.art.plume_center(ds["cell_volume"], ds["z_mc"], dim=("height", "ncells"))
        """
        tracer_mass = self._obj * v_cell
        weighted_height = tracer_mass * z_mc
        return weighted_height.sum(dim=dim, skipna=skipna) / tracer_mass.sum(dim=dim, skipna=skipna)

    def plume_top(self, z_mc, height_dim="height", fill_value=0):
        """
        Height of the highest plume cell at each horizontal location.

        Parameters
        ----------
        z_mc : xarray.DataArray
            Height of model-cell centers.
        height_dim : str, default "height"
            Vertical dimension name.
        fill_value : scalar, default 0
            Value used where no plume is present.

        Examples
        --------
        >>> import artist
        >>> top = plume.art.plume_top(ds["z_mc"])
        """
        z_plume = z_mc.where(self._obj > 0)
        return z_plume.max(dim=height_dim).fillna(fill_value)

    def plume_bottom(self, z_mc, height_dim="height", fill_value=0):
        """
        Height of the lowest plume cell at each horizontal location.

        Parameters
        ----------
        z_mc : xarray.DataArray
            Height of model-cell centers.
        height_dim : str, default "height"
            Vertical dimension name.
        fill_value : scalar, default 0
            Value used where no plume is present.

        Examples
        --------
        >>> import artist
        >>> bottom = plume.art.plume_bottom(ds["z_mc"])
        """
        z_plume = z_mc.where(self._obj > 0)
        return z_plume.min(dim=height_dim).fillna(fill_value)

    def value_at_plume_top(self, z_mc, val, height_dim="height", fill_value=0):
        """
        Value of another field at the plume top.

        Parameters
        ----------
        z_mc : xarray.DataArray
            Height of model-cell centers.
        val : xarray.DataArray
            Field to sample at the plume top.
        height_dim : str, default "height"
            Vertical dimension name.
        fill_value : scalar, default 0
            Value used where no plume is present.

        Examples
        --------
        >>> import artist
        >>> temp_at_top = plume.art.value_at_plume_top(ds["z_mc"], ds["temp"])
        """
        z_plume = z_mc.where(self._obj > 0)
        top = z_plume.max(dim=height_dim)
        return val.where(z_plume == top).max(dim=height_dim).fillna(fill_value)

    def max_conc_height(self, z_mc, height_dim="height", fill_value=0):
        """
        Height where plume concentration is maximal.

        Parameters
        ----------
        z_mc : xarray.DataArray
            Height of model-cell centers.
        height_dim : str, default "height"
            Vertical dimension name.
        fill_value : scalar, default 0
            Value used where no plume is present.

        Examples
        --------
        >>> import artist
        >>> z_max = plume.art.max_conc_height(ds["z_mc"])
        """
        plume = self._obj.where(self._obj > 0)
        z_max = z_mc.where(plume == plume.max(dim=height_dim, skipna=True))
        return z_max.mean(dim=height_dim, skipna=True).fillna(fill_value)

    def make_slice_time(self, height, gridpoint, start_t=0, end_t=1, height_level_max=30):
        """
        Build vectorized arrays for a time-height plot at one native gridpoint.

        Parameters
        ----------
        height : xarray.DataArray
            Height field with `time`, `height`, and `ncells` dimensions.
        gridpoint : int or array-like
            Native cell index. If array-like, the first value is used.
        start_t, end_t : int
            Time-index range passed to `isel`.
        height_level_max : int, default 30
            Number of upper model levels to include.

        Examples
        --------
        >>> import artist
        >>> x, y, values, timesteps, ground = da.art.make_slice_time(ds["z_mc"], 42, 0, 24)
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

    def plot_hov(self, height, gridpoints, start_t=0, end_t=100, levels=80, point_size=90):
        """
        Plot a time-height Hovmoller-style view at one native gridpoint.

        The plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import artist
        >>> da.art.plot_hov(ds["z_mc"], 42, start_t=0, end_t=24)
        """
        return _plot_hov(
            self,
            height,
            gridpoints,
            start_t=start_t,
            end_t=end_t,
            levels=levels,
            point_size=point_size,
        )

    def make_slice(self, height, gridpoints, t, height_level_max=30):
        """
        Build vectorized arrays for a vertical slice from selected native cells.

        Parameters
        ----------
        height : xarray.DataArray
            Height field with `time`, `height`, and `ncells` dimensions.
        gridpoints : array-like
            Native cell indices along the slice.
        t : int
            Time index passed to `isel`.
        height_level_max : int, default 30
            Number of upper model levels to include.

        Examples
        --------
        >>> import artist
        >>> x, y, values, x_data, ground, t = da.art.make_slice(ds["z_mc"], gridpoints, t=0)
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

    def plot_slice(self, height, gridpoints, t, deg_E_start, deg_E_end, deg_N, n=4000, levels=80, point_size=35):
        """
        Plot a vertical tracer slice along native gridpoints.

        The plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import artist
        >>> da.art.plot_slice(ds["z_mc"], gridpoints, t=0, deg_E_start=13, deg_E_end=14, deg_N=52)
        """
        return _plot_slice(
            self,
            height,
            gridpoints,
            t,
            deg_E_start,
            deg_E_end,
            deg_N,
            n=n,
            levels=levels,
            point_size=point_size,
        )

    def quick_plot(self, gridfile=None, cmap="coolwarm", levels=10, projection=None):
        """
        Quickly plot this DataArray on a Cartopy map.

        The plotting implementation lives in `artist.__plot`.

        Examples
        --------
        >>> import artist
        >>> ax = da.art.quick_plot(gridfile="icon_grid.nc")
        """
        return _quick_plot(self, gridfile=gridfile, cmap=cmap, levels=levels, projection=projection)
