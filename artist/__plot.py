import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from copy import copy
from matplotlib.collections import PolyCollection

from .utils import add_grid

"""
Plotting implementations used by the xarray accessors.

Functions in this module intentionally accept an accessor instance as their
first argument. Public methods in `accessor.py` forward to these functions so
that notebook tab completion still sees normal methods on `ds.icon`, `da.icon`,
and `da.art`.
"""


def _cartopy():
    """
    Import Cartopy only when a map plotting method needs it.

    Keeping Cartopy lazy lets users import `artist` and build the documentation
    in environments where the optional map plotting dependency is not installed.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "Cartopy is required for ARTist map plotting methods. "
            "Install cartopy to use this plotting function."
        ) from exc
    return ccrs, cfeature


def _shapely_geometry():
    try:
        import shapely.geometry as sgeom
    except ImportError as exc:
        raise ImportError(
            "Shapely is required for non-cylindrical gridline ticks. "
            "Install shapely to use this plotting function."
        ) from exc
    return sgeom


def _outline_line(ax):
    sgeom = _shapely_geometry()
    if hasattr(ax, "outline_patch"):
        vertices = ax.outline_patch.get_path().vertices.tolist()
    else:
        vertices = ax.spines["geo"].get_path().vertices.tolist()
    return sgeom.LineString(vertices)


def _find_side(line, side):
    """
    Return one side of a rectangular Cartopy outline as a LineString.
    """
    sgeom = _shapely_geometry()
    minx, miny, maxx, maxy = line.bounds
    points = {
        "left": [(minx, miny), (minx, maxy)],
        "right": [(maxx, miny), (maxx, maxy)],
        "bottom": [(minx, miny), (maxx, miny)],
        "top": [(minx, maxy), (maxx, maxy)],
    }
    return sgeom.LineString(points[side])


def _noncyl_ticks(ax, ticks, tick_location, line_constructor, tick_extractor, n_steps=30):
    """
    Compute visible tick locations for non-cylindrical Cartopy projections.
    """
    ccrs, _ = _cartopy()
    sgeom = _shapely_geometry()
    axis = _find_side(_outline_line(ax), tick_location)
    extent = ax.get_extent(ccrs.PlateCarree())
    locations = []

    for tick in ticks:
        xy = line_constructor(tick, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        line = sgeom.LineString(proj_xyz[..., :2].tolist())
        intersection = axis.intersection(line)
        if intersection.is_empty:
            locations.append(None)
        else:
            locations.append(tick_extractor(intersection.xy)[0])

    labels = copy(list(ticks))
    while True:
        try:
            index = locations.index(None)
        except ValueError:
            break
        locations.pop(index)
        labels.pop(index)
    return locations, labels


def noncyl_xticks(ax, ticks, fontsize=None):
    """
    Draw longitude ticks on the bottom axis for non-cylindrical projections.

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> import artist
    >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    >>> ds.icon.noncyl_xticks(ax, range(-180, 181, 60))
    """
    tick_extractor = lambda xy: xy[0]
    line_constructor = lambda tick, n, bounds: np.vstack(
        (np.zeros(n) + tick, np.linspace(bounds[2], bounds[3], n))
    ).T
    xticks, labels = _noncyl_ticks(ax, list(ticks), "bottom", line_constructor, tick_extractor)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    label_kwargs = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_xticklabels(
        [ax.xaxis.get_major_formatter()(label) for label in labels],
        **label_kwargs,
    )
    return ax


def noncyl_yticks(ax, ticks, fontsize=None):
    """
    Draw latitude ticks on the left axis for non-cylindrical projections.

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> import artist
    >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    >>> ds.icon.noncyl_yticks(ax, range(-90, 91, 30))
    """
    tick_extractor = lambda xy: xy[1]
    line_constructor = lambda tick, n, bounds: np.vstack(
        (np.linspace(bounds[0], bounds[1], n), np.zeros(n) + tick)
    ).T
    yticks, labels = _noncyl_ticks(ax, list(ticks), "left", line_constructor, tick_extractor)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    label_kwargs = {} if fontsize is None else {"fontsize": fontsize}
    ax.set_yticklabels(
        [ax.yaxis.get_major_formatter()(label) for label in labels],
        **label_kwargs,
    )
    return ax


def noncyl_gridlines(ax, xticks=None, yticks=None, fontsize=None, **gridline_kwargs):
    """
    Draw gridlines and visible edge ticks for non-cylindrical Cartopy maps.

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import matplotlib.pyplot as plt
    >>> import artist
    >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    >>> ds.icon.noncyl_gridlines(ax, xticks=range(-180, 181, 60), yticks=range(-90, 91, 30))
    """
    ccrs, _ = _cartopy()
    defaults = {"linestyle": "--", "linewidth": 0.6, "alpha": 0.5}
    defaults.update(gridline_kwargs)
    ax.gridlines(crs=ccrs.PlateCarree(), **defaults)
    if xticks is not None:
        noncyl_xticks(ax, xticks, fontsize=fontsize)
    if yticks is not None:
        noncyl_yticks(ax, yticks, fontsize=fontsize)
    return ax

@xr.register_dataarray_accessor('viz')
class PlotAccessor(object):
    """
    Lightweight plotting accessor for DataArrays with `clon`/`clat` coordinates.

    Accessed as `da.viz` after importing `artist`.
    """

    def __init__(self, da):
        self._obj = da
    
    def tricontourf(self, ax, cmap='coolwarm', levels=10, add_colorbar=True, map_extent=None, projection=None):
        """
        Draw a triangular contour plot from native lon/lat cell centers.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes. May be a Cartopy GeoAxes.
        cmap : str or Colormap, default "coolwarm"
            Matplotlib colormap.
        levels : int or sequence, default 10
            Contour levels passed to `Axes.tricontourf`.
        add_colorbar : bool, default True
            If True, add a colorbar to the current figure.
        map_extent : sequence, optional
            Reserved for future extent handling.
        projection : cartopy.crs.Projection, optional
            Projection used to transform `clon`/`clat`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import cartopy.crs as ccrs
        >>> import artist
        >>> fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
        >>> da.viz.tricontourf(ax, projection=ccrs.Robinson())
        """
        ccrs, _ = _cartopy()
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


def show_slice_line(self, points, gridpoints, grid_stride=5):
    """
    Show a vertical-slice path and the matching native gridpoints.

    Parameters
    ----------
    self : IconAccessor
        Dataset accessor carrying `clon` and `clat` attributes.
    points : array-like
        Lon/lat points defining the requested slice line.
    gridpoints : array-like
        Native ICON cell indices selected for the slice.
    grid_stride : int, default 5
        Plot every nth native gridpoint in the background.

    Examples
    --------
    >>> import artist
    >>> ds.icon.add_grid("icon_grid.nc")
    >>> points = [[13.0, 52.0], [14.0, 53.0]]
    >>> gridpoints = ds.icon.nearest_gridpoints(points)
    >>> ds.icon.show_slice_line(points, gridpoints)
    """
    points = np.asarray(points)
    gridpoints = np.asarray(gridpoints, dtype=int)
    clon = np.asarray(self.clon.values)
    clat = np.asarray(self.clat.values)

    plt.figure(figsize=[16, 9], facecolor="white")

    ccrs, cfeature = _cartopy()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=2,
        color="black",
        draw_labels=True,
        alpha=0.5,
        linestyle="--",
    )
    plt.title("position of vertical slice(red) and corresponding gridpoints(blue)")

    plt.plot(clon[::grid_stride], clat[::grid_stride], "k,")
    plt.plot(points[:, 0], points[:, 1], "r+-")
    plt.plot(clon[gridpoints], clat[gridpoints], "bx")
    return ax


def tri_data(self, gridfile, cmap=None, vrange=[], ltranslon=True):
    """
    Build native-cell triangle vertices and RGBA colors for plotting.

    Parameters
    ----------
    self : GridAccessor
        DataArray accessor for the field to color.
    gridfile : str or path-like
        ICON grid file matching the DataArray cell order.
    cmap : str or Colormap, optional
        Colormap. Defaults to `matplotlib.cm.viridis`.
    vrange : sequence, optional
        Two-value color normalization range.
    ltranslon : bool, default True
        If True, transform longitudes to [0, 360].

    Examples
    --------
    >>> import artist
    >>> triangles, colors, cmap = da.icon.tri_data("icon_grid.nc")
    """
    cmap = cmap or cm.viridis
    if len(vrange) > 0:
        norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[-1])
    else:
        norm = mpl.colors.Normalize(vmin=self._obj.min(), vmax=self._obj.max())

    _, self.vlon, self.vlat, _, _ = add_grid(gridfile, ltranslon=ltranslon)
    cmp = cm.ScalarMappable(norm=norm, cmap=cmap)
    triangles = np.stack((self.vlon, self.vlat), axis=2)
    colors = cmp.to_rgba(self._obj)
    return triangles, colors, cmp


def tri_plot(self, gridfile, ax, cmap=None, vrange=[], ltranslon=False, add_colorbar=True, map_extent=None):
    """
    Plot a DataArray on ICON native triangular cells.

    Parameters
    ----------
    self : GridAccessor
        DataArray accessor for the field to plot.
    gridfile : str or path-like
        ICON grid file matching the DataArray cell order.
    ax : matplotlib.axes.Axes
        Target axes.
    cmap : str or Colormap, optional
        Colormap. Defaults to `matplotlib.cm.viridis`.
    vrange : sequence, optional
        Two-value color normalization range.
    ltranslon : bool, default False
        If True, transform longitudes to [0, 360].
    add_colorbar : bool, default True
        If True, add a colorbar.
    map_extent : sequence, optional
        `[lon_min, lon_max, lat_min, lat_max]` plot extent.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import artist
    >>> fig, ax = plt.subplots()
    >>> da.icon.tri_plot("icon_grid.nc", ax)
    """
    triangles, colors, cmp = self.tri_data(gridfile, cmap=cmap, vrange=vrange, ltranslon=ltranslon)
    coll = PolyCollection(triangles, facecolor=colors, closed=True, edgecolor="face")
    ax.add_collection(coll, autolim=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if add_colorbar:
        plt.colorbar(cmp)

    if map_extent:
        ax.set_xlim([map_extent[0], map_extent[1]])
        ax.set_ylim([map_extent[2], map_extent[3]])
    else:
        ax.set_xlim([self.vlon.min(), self.vlon.max()])
        ax.set_ylim([self.vlat.min(), self.vlat.max()])
    return cmp


def plot_hov(self, height, gridpoints, start_t=0, end_t=100, levels=80, point_size=90):
    """
    Plot a time-height view for one native gridpoint.

    Parameters
    ----------
    self : ArtAccessor
        DataArray accessor for the tracer field.
    height : xarray.DataArray
        Height field with `time`, `height`, and `ncells` dimensions.
    gridpoints : int or array-like
        Native cell index. If array-like, the first value is used.
    start_t, end_t : int
        Time-index range passed to `isel`.
    levels : int or sequence, default 80
        Contour levels passed to `tricontourf`.
    point_size : float, default 90
        Scatter marker size.

    Examples
    --------
    >>> import artist
    >>> da.art.plot_hov(ds["z_mc"], 42, start_t=0, end_t=24)
    """
    timesteps_expanded, y_axis, pollbetu, timesteps, ground = self.make_slice_time(
        height, gridpoints, start_t=start_t, end_t=end_t
    )

    if timesteps_expanded.size < 3 or y_axis.size < 3:
        raise ValueError("At least three points are required for tricontourf.")

    print(timesteps_expanded.shape, y_axis.shape, pollbetu.shape)

    plt.figure(figsize=(16, 9), facecolor="white")
    plt.grid()
    plt.tricontourf(timesteps_expanded, y_axis, pollbetu, cmap="Reds", levels=levels)
    cmap1 = plt.scatter(
        timesteps_expanded, y_axis, marker="o", s=point_size, c=pollbetu, cmap="Reds"
    )
    plt.colorbar(cmap1, label=getattr(self._obj, "name", "tracer"))
    plt.ylabel("Height (m)")
    plt.xlabel("Timestep (h)")
    plt.xticks(timesteps[::3])
    plt.yticks()
    plt.title("lat=xxx   lon=xxx")
    plt.tight_layout()
    return

def plot_slice(self, height, gridpoints, t, deg_E_start, deg_E_end, deg_N, n=4000, levels=80, point_size=35):
    """
    Plot a vertical tracer slice along selected native gridpoints.

    Parameters
    ----------
    self : ArtAccessor
        DataArray accessor for the tracer field.
    height : xarray.DataArray
        Height field with `time`, `height`, and `ncells` dimensions.
    gridpoints : array-like
        Native ICON cell indices along the slice.
    t : int
        Time index passed to `isel`.
    deg_E_start, deg_E_end : float
        Longitude labels used for the x-axis.
    deg_N : float
        Latitude label used in the title.
    n : int, default 4000
        Horizontal spacing scale used for plotting.
    levels : int or sequence, default 80
        Contour levels passed to `tricontourf`.
    point_size : float, default 35
        Scatter marker size.

    Examples
    --------
    >>> import artist
    >>> da.art.plot_slice(ds["z_mc"], gridpoints, t=0, deg_E_start=13, deg_E_end=14, deg_N=52)
    """
    x_axis, y_axis, dust, x_data, ground, t = self.make_slice(height, gridpoints, t)
    x_scaled = x_axis * n
    vmax = np.nanmax(dust)

    plt.figure(figsize=(26, 9), facecolor="white")
    plt.title(
        "vertical slice from {} to {}°E, {}°N at timestep {}".format(
            deg_E_start, deg_E_end, deg_N, t
        )
    )

    cmap1 = plt.tricontourf(x_scaled, y_axis, dust, cmap="Reds", vmin=0, vmax=vmax, levels=levels)
    plt.scatter(x_scaled, y_axis, c=dust, cmap="Reds", vmin=0, vmax=vmax, marker="o", s=point_size)

    plt.colorbar(cmap1, label="dust optical depth")
    plt.xticks(x_data * n, np.linspace(deg_E_start, deg_E_end, len(x_data)))
    plt.fill_between(x_data * n, ground[::-1], color="grey")
    plt.ylim(0)
    plt.xlabel("Longitude in °E")
    plt.ylabel("Height in m")
    return


def quick_plot(self, gridfile=None, cmap="coolwarm", levels=10, projection=None):
    """
    Quickly plot a DataArray on a Cartopy map.

    Parameters
    ----------
    self : ArtAccessor
        DataArray accessor for the field to plot.
    gridfile : str or path-like, optional
        ICON grid file. Required when the DataArray has no `clon`/`clat`
        coordinates.
    cmap : str or Colormap, default "coolwarm"
        Matplotlib colormap.
    levels : int or sequence, default 10
        Contour levels passed to `tricontourf`.
    projection : cartopy.crs.Projection, optional
        Map projection. Defaults to Robinson.

    Examples
    --------
    >>> import artist
    >>> ax = da.art.quick_plot(gridfile="icon_grid.nc")
    """
    ccrs, _ = _cartopy()
    projection = projection or ccrs.Robinson()
    rad2deg = 180.0 / np.pi

    if "clon" in self._obj.coords and "clat" in self._obj.coords:
        clon = self._obj.coords["clon"].values
        clat = self._obj.coords["clat"].values
    else:
        if gridfile is None:
            raise ValueError("Pass gridfile or attach clon/clat coordinates before quick_plot().")
        g = xr.open_dataset(gridfile)
        clon = g["clon"].values * rad2deg
        clat = g["clat"].values * rad2deg

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), subplot_kw={"projection": projection})
    mproj = projection.transform_points(ccrs.PlateCarree(), clon, clat)
    x, y = mproj[:, 0], mproj[:, 1]
    tcf = ax.tricontourf(x, y, self._obj, cmap=cmap, levels=levels)

    plt.colorbar(tcf, orientation="vertical", pad=0.05)
    ax.coastlines()
    plt.show()
    return ax
