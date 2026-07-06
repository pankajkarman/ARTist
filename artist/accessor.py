import xarray as xr
import numpy as np
import pandas as pd

from .__plot import (
    noncyl_gridlines as _noncyl_gridlines,
    noncyl_xticks as _noncyl_xticks,
    noncyl_yticks as _noncyl_yticks,
    plot_hov as _plot_hov,
    plot_slice as _plot_slice,
    quick_plot as _quick_plot,
    show_slice_line as _show_slice_line,
    tri_data as _tri_data,
    tri_plot as _tri_plot,
)
from .utils import add_grid, distance as _distance

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
            vlon_values = np.asarray(vlon.values)
            vlat_values = np.asarray(vlat.values)
            for name, var in self._obj.data_vars.items():
                if cell_dim in var.dims and var.sizes[cell_dim] == clon_values.size:
                    self._obj[name].attrs["_artist_vlon"] = vlon_values
                    self._obj[name].attrs["_artist_vlat"] = vlat_values

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

    def distance(self, origin, destination, radius=6371.0):
        """
        Compute great-circle distance between lon/lat coordinates.

        Parameters
        ----------
        origin, destination : array-like
            Coordinate pairs in degrees as `[lon, lat]`. Inputs may also be
            arrays whose last dimension is `(lon, lat)`.
        radius : float, default 6371.0
            Spherical Earth radius in kilometers.

        Returns
        -------
        float or numpy.ndarray
            Distance in kilometers.

        Examples
        --------
        >>> import artist
        >>> ds.icon.distance([8.4, 49.0], [13.4, 52.5])
        """
        return _distance(origin, destination, radius=radius)

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

    def sellonlat(self, lonmin, lonmax, latmin, latmax, lon="clon", lat="clat"):
        """
        Select native ICON cells inside a longitude/latitude box.

        This is the accessor version of the ``sellonlat`` helper used in the
        analysis scripts. The dataset must already contain center coordinates,
        either from the original file or from ``ds.icon.add_grid(...)``.

        Parameters
        ----------
        lonmin, lonmax, latmin, latmax : float
            Selection bounds in degrees.
        lon, lat : str, default "clon", "clat"
            Coordinate or variable names containing cell-center longitude and
            latitude in degrees.

        Returns
        -------
        xarray.Dataset
            Dataset restricted to cells inside the bounding box.

        Examples
        --------
        >>> import artist
        >>> ds.icon.add_grid("icon_grid.nc")
        >>> regional = ds.icon.sellonlat(-100, 40, -20, 60)
        """
        missing = [name for name in (lon, lat) if name not in self._obj]
        if missing:
            raise KeyError(
                "Dataset is missing {}. Call ds.icon.add_grid(...) or provide lon/lat names.".format(
                    ", ".join(missing)
                )
            )

        mask = (
            (self._obj[lon] > lonmin)
            & (self._obj[lon] < lonmax)
            & (self._obj[lat] > latmin)
            & (self._obj[lat] < latmax)
        )
        return self._obj.where(mask, drop=True)

    def sel_lonlat(self, lonmin, lonmax, latmin, latmax, lon="clon", lat="clat"):
        """
        Select native ICON cells inside a longitude/latitude box.

        This is a more xarray-style alias for ``sellonlat``.

        Examples
        --------
        >>> import artist
        >>> regional = ds.icon.sel_lonlat(-100, 40, -20, 60)
        """
        return self.sellonlat(lonmin, lonmax, latmin, latmax, lon=lon, lat=lat)

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

    def noncyl_xticks(self, ax, ticks, fontsize=None):
        """
        Draw longitude ticks for a non-cylindrical Cartopy projection.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes
            Cartopy axes to modify.
        ticks : sequence of float
            Longitude tick labels in degrees.
        fontsize : float, optional
            Tick-label font size.

        Examples
        --------
        >>> import cartopy.crs as ccrs
        >>> import matplotlib.pyplot as plt
        >>> import artist
        >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        >>> ds.icon.noncyl_xticks(ax, range(-180, 181, 60))
        """
        return _noncyl_xticks(ax, ticks, fontsize=fontsize)

    def noncyl_yticks(self, ax, ticks, fontsize=None):
        """
        Draw latitude ticks for a non-cylindrical Cartopy projection.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes
            Cartopy axes to modify.
        ticks : sequence of float
            Latitude tick labels in degrees.
        fontsize : float, optional
            Tick-label font size.

        Examples
        --------
        >>> import cartopy.crs as ccrs
        >>> import matplotlib.pyplot as plt
        >>> import artist
        >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        >>> ds.icon.noncyl_yticks(ax, range(-90, 91, 30))
        """
        return _noncyl_yticks(ax, ticks, fontsize=fontsize)

    def noncyl_gridlines(self, ax, xticks=None, yticks=None, fontsize=None, **gridline_kwargs):
        """
        Draw gridlines and edge ticks for non-cylindrical Cartopy maps.

        Parameters
        ----------
        ax : cartopy.mpl.geoaxes.GeoAxes
            Cartopy axes to modify.
        xticks, yticks : sequence of float, optional
            Longitude and latitude tick labels in degrees.
        fontsize : float, optional
            Tick-label font size.
        **gridline_kwargs
            Extra keyword arguments passed to `ax.gridlines`.

        Examples
        --------
        >>> import cartopy.crs as ccrs
        >>> import matplotlib.pyplot as plt
        >>> import artist
        >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        >>> ds.icon.noncyl_gridlines(ax, xticks=range(-180, 181, 60), yticks=range(-90, 91, 30))
        """
        return _noncyl_gridlines(
            ax,
            xticks=xticks,
            yticks=yticks,
            fontsize=fontsize,
            **gridline_kwargs,
        )

        
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


@xr.register_dataset_accessor("art")
class ArtDatasetAccessor(object):
    """
    Dataset-level ICON-ART optical diagnostics.

    Accessed as `ds.art` after importing `artist`.
    """

    _wavelengths = {"355": 0, "532": 1, "1064": 2}

    _lidar_ext = {
        "insol_acc": np.array([0.89209, 0.97248, 0.9832]),
        "insol_coa": np.array([0.13508, 0.13881, 0.14692]),
        "mixed_acc": np.array([1.26710, 1.33489, 1.36351]),
        "mixed_coa": np.array([0.19576, 0.19838, 0.21014]),
        "sol_ait": np.array([4.65102, 2.48959, 0.58964]),
        "sol_acc": np.array([1.23598, 1.31627, 1.19881]),
    }
    _lidar_bsc = {
        "insol_acc": np.array([0.07977, 0.09244, 0.09244]),
        "insol_coa": np.array([0.00569, 0.00695, 0.0102]),
        "mixed_acc": np.array([0.11832, 0.12636, 0.12038]),
        "mixed_coa": np.array([0.00791, 0.01039, 0.01571]),
        "sol_ait": np.array([0.36542, 0.21806, 0.0836]),
        "sol_acc": np.array([0.10440, 0.10777, 0.08138]),
    }
    _aod_ext = {
        "insol_acc": np.array([1.49926, 1.60644, 1.79298]),
        "insol_coa": np.array([0.14943, 0.15129, 0.15459]),
        "mixed_acc": np.array([6.01615, 4.70122, 1.93800]),
        "mixed_coa": np.array([0.26428, 0.26733, 0.28390]),
        "sol_ait": np.array([0.60659, 0.24575, 0.02586]),
        "sol_acc": np.array([5.00644, 3.06857, 0.70334]),
    }
    _saod_wavelengths = {"355": 0, "532": 1, "1064": 2, "8547": 3}
    _saod_ext = {
        "sol_ait": np.array([0.60659, 0.24575, 0.02586, 0.39568]),
        "sol_acc": np.array([5.00644, 3.06857, 0.70334, 0.40169]),
    }
    _component_density = {
        "dust": 2.65e3,
        "na": 2.2e3,
        "cl": 2.2e3,
        "soot": 1.3e3,
        "ash": 2.65e3,
        "h2o": 1.0e3,
        "so4": 1.8e3,
        "nh4": 1.8e3,
        "no3": 1.8e3,
    }

    def __init__(self, ds):
        self._obj = ds

    @classmethod
    def _wavelength_index(cls, wavelength):
        key = str(wavelength)
        if key not in cls._wavelengths:
            raise ValueError("wavelength must be one of 355, 532, or 1064 nm.")
        return key, cls._wavelengths[key]

    def _require(self, names):
        missing = [name for name in names if name not in self._obj]
        if missing:
            raise KeyError("Dataset is missing required variables: {}".format(", ".join(missing)))

    def _get_dz(self):
        z_ifc = self._obj["z_ifc"]
        height_dim = "height_2" if "height_2" in z_ifc.dims else "height_3"
        dz = -z_ifc.diff(height_dim).rename({height_dim: "height"})
        return dz.assign_coords(height=(dz.height - 1))

    @staticmethod
    def _tau(cross_section, concentration, rho):
        return cross_section * concentration * rho * 1e-6

    def _has_aerodyn(self):
        return all(
            name in self._obj
            for name in (
                "ash_mixed_acc",
                "ash_mixed_coa",
                "so4_mixed_acc",
                "so4_mixed_coa",
                "so4_sol_ait",
                "so4_sol_acc",
            )
        )

    @classmethod
    def _saod_wavelength_index(cls, wavelength):
        key = str(wavelength)
        if key not in cls._saod_wavelengths:
            raise ValueError("wavelength must be one of 355, 532, 1064, or 8547 nm.")
        return cls._saod_wavelengths[key]

    def _dataarray_or_var(self, tracer):
        if isinstance(tracer, str):
            self._require([tracer])
            return self._obj[tracer]
        return tracer

    @staticmethod
    def _mode_moment(diameter, number, rho, sigma, moment):
        exponent = 0.5 * moment**2 * np.log(sigma) ** 2
        return (diameter / 2.0) ** moment * number * rho * np.exp(exponent)

    def vmr_to_du(self, tracer, pres=None, temp=None, dz=None):
        """
        Convert a volume mixing ratio profile to Dobson units.

        Parameters
        ----------
        tracer : str or xarray.DataArray
            VMR field to integrate over height.
        pres, temp, dz : xarray.DataArray, optional
            Pressure, temperature, and layer thickness. If omitted, ``pres``
            and ``temp`` are read from the dataset and ``dz`` is computed from
            ``z_ifc``.

        Examples
        --------
        >>> import artist
        >>> so2_du = ds.art.vmr_to_du("TRSO2_chemtr")
        """
        tr = self._dataarray_or_var(tracer)
        pres = self._obj["pres"] if pres is None else pres
        temp = self._obj["temp"] if temp is None else temp
        dz = self._get_dz() if dz is None else dz
        column = (tr * pres / (8.314472 * temp)) * dz
        column = column * 6.02214076e23 / 2.69e20
        return column.sum("height")

    def plume_mass(self, var, thres_var, thres, cell_area=None, dz=None, molar_frac=1.0):
        """
        Compute total tracer mass inside a threshold-defined plume.

        Parameters
        ----------
        var : str
            Tracer variable to integrate. ``"OH"`` follows the OH conversion
            used in the source script.
        thres_var : str
            Variable used to define plume pixels.
        thres : float
            Threshold; cells with ``ds[thres_var] >= thres`` are included.
        cell_area, dz : xarray.DataArray, optional
            Cell area and layer thickness. Defaults are ``ds.cell_area`` and
            ``ds.art``'s layer-thickness calculation.
        molar_frac : float, default 1.0
            Multiplicative molar fraction applied to non-OH tracers.

        Examples
        --------
        >>> import artist
        >>> mass = ds.art.plume_mass("ash_mixed_acc", "ash_mixed_acc", 0.1)
        """
        self._require([thres_var, "rho"])
        cell_area = self._obj["cell_area"] if cell_area is None else cell_area
        dz = self._get_dz() if dz is None else dz
        threshold = self._obj[thres_var] >= thres

        if var == "OH":
            self._require(["OH_Nconc", "pres", "temp"])
            vmr_to_number_conc = (6.02214086e23 * self._obj.pres) / (8.314409 * self._obj.temp) * 1e-6
            cut_plume = (self._obj.OH_Nconc / vmr_to_number_conc).where(threshold) * 17.008 / 28.97
        else:
            self._require([var])
            cut_plume = self._obj[var].where(threshold) * molar_frac

        plume_load = (cut_plume * self._obj.rho * dz).sum("height")
        return (plume_load * cell_area).sum("ncells")

    def density(self, components):
        """
        Compute particle density by mass-weighted component density.

        Parameters
        ----------
        components : sequence of str
            Component variable names such as ``["so4_mixed_acc", "h2o_mixed_acc"]``.

        Examples
        --------
        >>> import artist
        >>> shell_density = ds.art.density(["so4_mixed_acc", "no3_mixed_acc", "h2o_mixed_acc"])
        """
        components = list(components)
        self._require(components)
        total = self._obj[components].to_array(dim="component").sum("component")
        density = 0
        for name in components:
            particle = name.split("_")[0]
            if particle not in self._component_density:
                raise ValueError("No default density is available for component {!r}.".format(particle))
            density = density + self._component_density[particle] * self._obj[name]
        return density / total

    def coating_fraction(self):
        """
        Compute ash core diameter fractions for mixed accumulation and coarse modes.

        Returns
        -------
        tuple of xarray.DataArray
            Accumulation-mode fraction, coarse-mode fraction, and ash-mass
            weighted total fraction.

        Examples
        --------
        >>> import artist
        >>> dcdt_acc, dcdt_coa, dcdt = ds.art.coating_fraction()
        """
        self._require(
            [
                "ash_mixed_acc",
                "so4_mixed_acc",
                "no3_mixed_acc",
                "h2o_mixed_acc",
                "ash_mixed_coa",
                "so4_mixed_coa",
                "no3_mixed_coa",
                "h2o_mixed_coa",
            ]
        )
        ds = self._obj
        fw_acc = ds.ash_mixed_acc / (ds.ash_mixed_acc + ds.so4_mixed_acc + ds.no3_mixed_acc + ds.h2o_mixed_acc)
        fw_coa = ds.ash_mixed_coa / (ds.ash_mixed_coa + ds.so4_mixed_coa + ds.no3_mixed_coa + ds.h2o_mixed_coa)

        dens_core_acc = self.density(["ash_mixed_acc"])
        dens_shell_acc = self.density(["so4_mixed_acc", "no3_mixed_acc", "h2o_mixed_acc"])
        fv_acc = fw_acc / (fw_acc + (1.0 - fw_acc) * dens_core_acc / dens_shell_acc)
        dcdt_acc = np.cbrt(fv_acc)

        dens_core_coa = self.density(["ash_mixed_coa"])
        dens_shell_coa = self.density(["so4_mixed_coa", "no3_mixed_coa", "h2o_mixed_coa"])
        fv_coa = fw_coa / (fw_coa + (1.0 - fw_coa) * dens_core_coa / dens_shell_coa)
        dcdt_coa = np.cbrt(fv_coa)

        total_ash = ds.ash_mixed_acc + ds.ash_mixed_coa
        dcdt = dcdt_acc * ds.ash_mixed_acc / total_ash + dcdt_coa * ds.ash_mixed_coa / total_ash
        return dcdt_acc, dcdt_coa, dcdt

    def effective_radius(self, kind="all"):
        """
        Compute aerosol effective radius for sulfate, ash, or all modes.

        Parameters
        ----------
        kind : {"all", "sulfate", "ash"}, default "all"
            Aerosol modes included in the diagnostic.

        Examples
        --------
        >>> import artist
        >>> r_eff = ds.art.effective_radius("ash")
        """
        kind = kind.lower()
        if kind not in {"all", "sulfate", "ash"}:
            raise ValueError("kind must be one of 'all', 'sulfate', or 'ash'.")

        ds = self._obj
        mode_defs = []
        if kind in {"all", "sulfate"}:
            self._require(["diam_sol_ait", "nmb_sol_ait", "so4_sol_ait", "diam_sol_acc", "nmb_sol_acc", "so4_sol_acc", "rho"])
            mode_defs.extend(
                [
                    ("diam_sol_ait", "nmb_sol_ait", "so4_sol_ait", 1e-3, 1.7),
                    ("diam_sol_acc", "nmb_sol_acc", "so4_sol_acc", 1e-1, 2.0),
                ]
            )
        if kind in {"all", "ash"}:
            self._require(
                [
                    "diam_insol_acc",
                    "nmb_insol_acc",
                    "ash_insol_acc",
                    "diam_insol_coa",
                    "nmb_insol_coa",
                    "ash_insol_coa",
                    "diam_mixed_acc",
                    "nmb_mixed_acc",
                    "ash_mixed_acc",
                    "diam_mixed_coa",
                    "nmb_mixed_coa",
                    "ash_mixed_coa",
                    "diam_giant",
                    "nmb_giant",
                    "ash_giant",
                    "rho",
                ]
            )
            mode_defs.extend(
                [
                    ("diam_insol_acc", "nmb_insol_acc", "ash_insol_acc", 1e-1, 2.0),
                    ("diam_insol_coa", "nmb_insol_coa", "ash_insol_coa", 1e0, 2.0),
                    ("diam_mixed_acc", "nmb_mixed_acc", "ash_mixed_acc", 1e-1, 2.2),
                    ("diam_mixed_coa", "nmb_mixed_coa", "ash_mixed_coa", 1e0, 2.2),
                    ("diam_giant", "nmb_giant", "ash_giant", 10.0, 2.0),
                ]
            )

        total_volume = 0
        total_area = 0
        for diam_name, number_name, mass_name, threshold, sigma in mode_defs:
            diameter = ds[diam_name].where(ds[mass_name] > threshold)
            volume = self._mode_moment(diameter, ds[number_name], ds.rho, sigma, 3).fillna(0)
            area = self._mode_moment(diameter, ds[number_name], ds.rho, sigma, 2).fillna(0)
            total_volume = total_volume + volume
            total_area = total_area + area

        return total_volume / total_area

    def reff_sulfate(self):
        """
        Compute sulfate effective radius.

        Examples
        --------
        >>> import artist
        >>> r_eff_sulfate = ds.art.reff_sulfate()
        """
        return self.effective_radius("sulfate")

    def reff_ash(self):
        """
        Compute ash effective radius.

        Examples
        --------
        >>> import artist
        >>> r_eff_ash = ds.art.reff_ash()
        """
        return self.effective_radius("ash")

    def reff_all(self):
        """
        Compute effective radius over sulfate and ash modes.

        Examples
        --------
        >>> import artist
        >>> r_eff_all = ds.art.reff_all()
        """
        return self.effective_radius("all")

    def rayleigh_part(self, wavelength, height_ref=90, scale_height=1.0):
        """
        Compute Rayleigh extinction and backscatter coefficients.

        Parameters
        ----------
        wavelength : {355, 532, 1064}
            Wavelength in nanometers.
        height_ref : scalar, default 90
            Reference height coordinate used for pressure and temperature.
        scale_height : float, default 1.0
            Exponential scaling height in meters, matching the original
            ICON-ART helper script.

        Returns
        -------
        tuple of xarray.DataArray
            Rayleigh extinction coefficient and backscatter coefficient.

        Examples
        --------
        >>> import artist
        >>> alpha, beta = ds.art.rayleigh_part(532)
        """
        wavelength, _ = self._wavelength_index(wavelength)
        self._require(["pres", "temp", "z_mc"])

        props = {
            "355": (355e-9, 0.0301, 1.00028570),
            "532": (532e-9, 0.02, 1.00027819),
            "1064": (1064e-9, 0.0273, 1.00027397),
        }
        ll, dd, ns = props[wavelength]
        number_density_stp = 2.547e19 / 100**3
        p0 = 101325.0
        t0 = 288.15

        ds = self._obj
        nsr = (
            number_density_stp
            * (ds.pres.sel(height=height_ref) / ds.temp.sel(height=height_ref))
            * np.exp(-ds.z_mc / scale_height)
        )
        alpha = (
            8.0
            / 3.0
            * np.pi**3
            * (ns**2 - 1.0) ** 2
            / (ll**4 * number_density_stp**2)
            * (6.0 + 3.0 * dd)
            / (6.0 - 7.0 * dd)
            * nsr
            * t0
            / p0
            * ds.pres
            / ds.temp
        )
        beta = 3.0 / (8.0 * np.pi) * alpha
        return alpha, beta

    def _lidar_ext_bsc(self, idx, spherical_only=False):
        ds = self._obj
        self._require(["ash_insol_acc", "ash_insol_coa", "rho"])

        ext = 0
        bsc = 0
        if not spherical_only:
            ext = ext + self._tau(self._lidar_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            ext = ext + self._tau(self._lidar_ext["insol_coa"][idx], ds.ash_insol_coa, ds.rho)
            bsc = bsc + self._tau(self._lidar_bsc["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            bsc = bsc + self._tau(self._lidar_bsc["insol_coa"][idx], ds.ash_insol_coa, ds.rho)

        if self._has_aerodyn():
            ext = ext + self._tau(
                self._lidar_ext["mixed_acc"][idx],
                ds.ash_mixed_acc + ds.so4_mixed_acc,
                ds.rho,
            )
            ext = ext + self._tau(
                self._lidar_ext["mixed_coa"][idx],
                ds.ash_mixed_coa + ds.so4_mixed_coa,
                ds.rho,
            )
            ext = ext + self._tau(self._lidar_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            ext = ext + self._tau(self._lidar_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
            bsc = bsc + self._tau(
                self._lidar_bsc["mixed_acc"][idx],
                ds.ash_mixed_acc + ds.so4_mixed_acc,
                ds.rho,
            )
            bsc = bsc + self._tau(
                self._lidar_bsc["mixed_coa"][idx],
                ds.ash_mixed_coa + ds.so4_mixed_coa,
                ds.rho,
            )
            bsc = bsc + self._tau(self._lidar_bsc["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            bsc = bsc + self._tau(self._lidar_bsc["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
        elif spherical_only:
            ext = ext + self._tau(self._lidar_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            ext = ext + self._tau(self._lidar_ext["insol_coa"][idx], ds.ash_insol_coa, ds.rho)
            bsc = bsc + self._tau(self._lidar_bsc["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            bsc = bsc + self._tau(self._lidar_bsc["insol_coa"][idx], ds.ash_insol_coa, ds.rho)
        return ext, bsc

    def att_bsct(self, wavelength):
        """
        Compute attenuated backscatter for aerosol tracers.

        Parameters
        ----------
        wavelength : {355, 532, 1064}
            Wavelength in nanometers.

        Examples
        --------
        >>> import artist
        >>> attenuated = ds.art.att_bsct(532)
        """
        _, idx = self._wavelength_index(wavelength)
        ext, bsc = self._lidar_ext_bsc(idx, spherical_only=False)
        ext_sum = (ext * self._get_dz()).cumsum(dim="height")
        return bsc * np.exp(-2.0 * ext_sum)

    def att_bsct_sph(self, wavelength):
        """
        Compute attenuated backscatter using spherical aerosol fractions.

        Examples
        --------
        >>> import artist
        >>> attenuated = ds.art.att_bsct_sph(532)
        """
        _, idx = self._wavelength_index(wavelength)
        ext, bsc = self._lidar_ext_bsc(idx, spherical_only=True)
        ext_sum = (ext * self._get_dz()).cumsum(dim="height")
        return bsc * np.exp(-2.0 * ext_sum)

    def aod(self, wavelength):
        """
        Compute layer aerosol optical depth.

        Parameters
        ----------
        wavelength : {355, 532, 1064}
            Wavelength in nanometers.

        Examples
        --------
        >>> import artist
        >>> layer_aod = ds.art.aod(532)
        >>> column_aod = layer_aod.sum("height")
        """
        _, idx = self._wavelength_index(wavelength)
        ds = self._obj
        self._require(["ash_insol_acc", "ash_insol_coa", "rho"])

        ext = self._tau(self._aod_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
        ext = ext + self._tau(self._aod_ext["insol_coa"][idx], ds.ash_insol_coa, ds.rho)

        if self._has_aerodyn():
            ext = ext + self._tau(self._aod_ext["mixed_acc"][idx], ds.ash_mixed_acc + ds.so4_mixed_acc, ds.rho)
            ext = ext + self._tau(self._aod_ext["mixed_coa"][idx], ds.ash_mixed_coa + ds.so4_mixed_coa, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
        elif "so4_sol_ait" in ds and "so4_sol_acc" in ds:
            ext = ext + self._tau(self._aod_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)

        return ext * self._get_dz()

    def aod_misr(self, wavelength, frac="all"):
        """
        Compute MISR-style layer aerosol optical depth by fraction.

        Parameters
        ----------
        wavelength : {355, 532, 1064}
            Wavelength in nanometers.
        frac : {"all", "ait", "acc", "insol", "mixed", "sol"}, default "all"
            Aerosol fraction to include.

        Examples
        --------
        >>> import artist
        >>> acc_aod = ds.art.aod_misr(532, frac="acc")
        """
        _, idx = self._wavelength_index(wavelength)
        ds = self._obj
        self._require(["rho"])
        frac = frac.lower()

        if frac == "all":
            self._require(["ash_insol_acc", "ash_mixed_acc", "so4_mixed_acc", "so4_sol_ait", "so4_sol_acc"])
            ext = self._tau(self._aod_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            ext = ext + self._tau(self._aod_ext["mixed_acc"][idx], ds.ash_mixed_acc + ds.so4_mixed_acc, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
        elif frac == "ait":
            self._require(["so4_sol_ait"])
            ext = self._tau(self._aod_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
        elif frac == "acc":
            self._require(["ash_insol_acc", "ash_mixed_acc", "so4_mixed_acc", "so4_sol_acc"])
            ext = self._tau(self._aod_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
            ext = ext + self._tau(self._aod_ext["mixed_acc"][idx], ds.ash_mixed_acc + ds.so4_mixed_acc, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
        elif frac == "insol":
            self._require(["ash_insol_acc"])
            ext = self._tau(self._aod_ext["insol_acc"][idx], ds.ash_insol_acc, ds.rho)
        elif frac == "mixed":
            self._require(["ash_mixed_acc", "so4_mixed_acc"])
            ext = self._tau(self._aod_ext["mixed_acc"][idx], ds.ash_mixed_acc + ds.so4_mixed_acc, ds.rho)
        elif frac == "sol":
            self._require(["so4_sol_ait", "so4_sol_acc"])
            ext = self._tau(self._aod_ext["sol_ait"][idx], ds.so4_sol_ait, ds.rho)
            ext = ext + self._tau(self._aod_ext["sol_acc"][idx], ds.so4_sol_acc, ds.rho)
        else:
            raise ValueError("frac must be one of 'all', 'ait', 'acc', 'insol', 'mixed', or 'sol'.")

        return ext * self._get_dz()

    def sulfate_aod(self, wavelength):
        """
        Compute sulfate-only layer aerosol optical depth.

        Parameters
        ----------
        wavelength : {355, 532, 1064, 8547}
            Wavelength in nanometers.

        Examples
        --------
        >>> import artist
        >>> sulfate_layer_aod = ds.art.sulfate_aod(532)
        >>> sulfate_column_aod = sulfate_layer_aod.sum("height")
        """
        idx = self._saod_wavelength_index(wavelength)
        ds = self._obj
        self._require(["so4_sol_ait", "so4_sol_acc", "rho"])
        sol_ait = ds.so4_sol_ait.where(ds.so4_sol_ait > 1e-3).fillna(0)
        sol_acc = ds.so4_sol_acc.where(ds.so4_sol_acc > 1e-1).fillna(0)
        ext = self._tau(self._saod_ext["sol_ait"][idx], sol_ait, ds.rho)
        ext = ext + self._tau(self._saod_ext["sol_acc"][idx], sol_acc, ds.rho)
        return ext * self._get_dz()

    def saod(self, wavelength):
        """
        Compute sulfate-only layer aerosol optical depth.

        This script-name alias mirrors ``saod`` from ``for_Dorsa.py``.

        Examples
        --------
        >>> import artist
        >>> sulfate_layer_aod = ds.art.saod(532)
        """
        return self.sulfate_aod(wavelength)

    def ssa(self, wavelength):
        """
        Compute mass-weighted single-scattering albedo.

        Only 532 nm is implemented in the source optical-property table.

        Examples
        --------
        >>> import artist
        >>> single_scattering_albedo = ds.art.ssa(532)
        """
        wavelength, idx = self._wavelength_index(wavelength)
        if wavelength != "532":
            raise ValueError("SSA is currently implemented only for 532 nm.")

        ds = self._obj
        self._require(
            [
                "ash_insol_acc",
                "ash_insol_coa",
                "ash_mixed_acc",
                "ash_mixed_coa",
                "so4_mixed_acc",
                "so4_mixed_coa",
                "so4_sol_ait",
                "so4_sol_acc",
            ]
        )
        ssa = {
            "insol_acc": np.array([0.0, 0.98669, 0.0]),
            "insol_coa": np.array([0.0, 0.90148, 0.0]),
            "mixed_acc": np.array([0.0, 0.99619, 0.0]),
            "mixed_coa": np.array([0.0, 0.93797, 0.0]),
            "sol_ait": np.array([0.0, 1.000000, 0.0]),
            "sol_acc": np.array([0.0, 1.000000, 0.0]),
        }
        mass_total = (
            ds.ash_insol_acc
            + ds.ash_insol_coa
            + ds.ash_mixed_acc
            + ds.ash_mixed_coa
            + ds.so4_mixed_acc
            + ds.so4_mixed_coa
            + ds.so4_sol_ait
            + ds.so4_sol_acc
        )
        return (
            ssa["insol_acc"][idx] * ds.ash_insol_acc / mass_total
            + ssa["mixed_acc"][idx] * (ds.so4_mixed_acc + ds.ash_mixed_acc) / mass_total
            + ssa["sol_ait"][idx] * ds.so4_sol_ait / mass_total
            + ssa["sol_acc"][idx] * ds.so4_sol_acc / mass_total
            + ssa["insol_coa"][idx] * ds.ash_insol_coa / mass_total
            + ssa["mixed_coa"][idx] * (ds.so4_mixed_coa + ds.ash_mixed_coa) / mass_total
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
