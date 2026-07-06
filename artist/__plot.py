import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import PolyCollection

from .utils import add_grid

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


def show_slice_line(self, points, gridpoints, grid_stride=5):
    """
    Show where the vertical slice is and which gridpoints are used.
    """
    points = np.asarray(points)
    gridpoints = np.asarray(gridpoints, dtype=int)
    clon = np.asarray(self.clon.values)
    clat = np.asarray(self.clat.values)

    plt.figure(figsize=[16, 9], facecolor="white")

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
    Original quick map, with optional gridfile.

    If the DataArray already has `clon` and `clat` coordinates, `gridfile`
    is not needed. Otherwise pass the ICON grid file as before.
    """
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
