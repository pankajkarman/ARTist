"""
Utilities for preparing EDGAR emissions for ICON OEM input.

The functions in this module wrap the reusable workflow from the
``edgar-temporal.ipynb`` notebook. ``emiproc`` is imported lazily so ARTist can
still be used for ordinary post-processing without installing the EDGAR/OEM
toolchain.
"""

from pathlib import Path

import pandas as pd
import xarray as xr
import numpy as np

from .utils import add_grid


DEFAULT_ICON_GRID_URL = "http://icon-downloads.mpimet.mpg.de/grids/public/edzw/icon_grid_0030_R02B05_G.nc"


def _open_oem_dataset(path):
    try:
        return xr.open_dataset(path)
    except Exception:
        return xr.open_dataset(path, engine="scipy")


def _require_emiproc():
    try:
        import emiproc  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "EDGAR/OEM helpers require emiproc. Install it with "
            "`pip install emiproc` before using artist.oem or ds.oem.map_edgar()."
        ) from exc


def _enum_value(enum_cls, value):
    if not isinstance(value, str):
        return value
    try:
        return getattr(enum_cls, value)
    except AttributeError as exc:
        choices = [name for name in dir(enum_cls) if name.isupper()]
        raise ValueError("{} must be one of {}.".format(value, ", ".join(choices))) from exc


def download_edgar(local_dir, year, substances):
    """
    Download EDGAR inventory files with emiproc.

    Parameters
    ----------
    local_dir : str or path-like
        Directory where EDGAR files are stored.
    year : int
        EDGAR inventory year.
    substances : sequence of str
        Substance names to download, for example ``["CH4", "CO2"]``.

    Examples
    --------
    >>> from artist.oem import download_edgar
    >>> download_edgar("./edgar", year=2022, substances=["CH4", "CO2"])
    """
    _require_emiproc()
    from emiproc.inventories.edgar import download_edgar_files

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    return download_edgar_files(local_dir, year=year, substances=list(substances))


def load_edgar_inventory(local_dir, pattern="*.nc", year=None, substances=None, use_short_category_names=True):
    """
    Load an EDGAR v8 inventory from local NetCDF files.

    Parameters
    ----------
    local_dir : str or path-like
        Directory containing EDGAR NetCDF files.
    pattern : str, default "*.nc"
        Glob pattern passed to ``EDGARv8``.
    year : int, optional
        Inventory year.
    substances : sequence of str, optional
        Substance names used to narrow matching EDGAR files when possible.
    use_short_category_names : bool, default True
        Forwarded to ``emiproc.inventories.edgar.EDGARv8``.

    Examples
    --------
    >>> from artist.oem import load_edgar_inventory
    >>> inv = load_edgar_inventory("./edgar", year=2022)
    """
    _require_emiproc()
    from emiproc.inventories.edgar import EDGARv8

    local_dir = Path(local_dir)
    edgar_files = local_dir / pattern
    if substances:
        substances = [substances] if isinstance(substances, str) else list(substances)
        if len(substances) == 1 and list(local_dir.glob("*{}*.nc".format(substances[0]))):
            edgar_files = local_dir / "*{}*.nc".format(substances[0])

    return EDGARv8(
        edgar_files,
        use_short_category_names=use_short_category_names,
        year=year,
    )


def attach_edgar_temporal_profiles(inventory, aux_data_path):
    """
    Read EDGAR auxiliary temporal profiles and attach them to an inventory.

    Parameters
    ----------
    inventory : emiproc inventory
        EDGAR inventory object.
    aux_data_path : str or path-like
        EDGAR auxiliary profile directory.

    Returns
    -------
    emiproc inventory
        The same inventory object, with profiles attached.
    """
    _require_emiproc()
    from emiproc.inventories.edgar.temporal import read_edgar_auxilary_profiles

    profiles, indexes = read_edgar_auxilary_profiles(Path(aux_data_path), inventory=inventory)
    inventory.set_profiles(profiles, indexes)
    return inventory


def get_icon_grid(icon_grid_file=None, grid_download_url=DEFAULT_ICON_GRID_URL, download=True):
    """
    Build an emiproc ICON grid object.

    Parameters
    ----------
    icon_grid_file : str or path-like, optional
        Existing ICON grid file. If omitted, the filename is derived from
        ``grid_download_url``.
    grid_download_url : str, default DEFAULT_ICON_GRID_URL
        Public grid URL used when ``download=True`` and the file is missing.
    download : bool, default True
        Download the grid file if it does not exist.

    Returns
    -------
    tuple
        ``(icon_grid_file, icon_grid)``.
    """
    _require_emiproc()
    from emiproc.grids import ICONGrid

    if icon_grid_file is None:
        icon_grid_file = Path(".") / grid_download_url.split("/")[-1]
    icon_grid_file = Path(icon_grid_file)

    if download and not icon_grid_file.exists():
        import urllib.request

        urllib.request.urlretrieve(grid_download_url, icon_grid_file)

    return icon_grid_file, ICONGrid(icon_grid_file)


def remap_to_icon(inventory, icon_grid, weights_file=".remap_weights", to_projected_crs=True):
    """
    Remap an inventory to an ICON grid with emiproc.

    Parameters
    ----------
    inventory : emiproc inventory
        Source inventory.
    icon_grid : emiproc.grids.ICONGrid
        Target ICON grid.
    weights_file : str or path-like, default ".remap_weights"
        Remapping weights cache file.
    to_projected_crs : bool, default True
        Convert inventory geometry to the projected WGS84 CRS before remapping.
    """
    _require_emiproc()
    from emiproc.grids import WGS84_PROJECTED
    from emiproc.regrid import remap_inventory

    if to_projected_crs:
        inventory.to_crs(WGS84_PROJECTED)
    return remap_inventory(inventory, icon_grid, weights_file=weights_file)


def set_constant_vertical_profiles(inventory, height=2.0):
    """
    Attach a single-layer constant vertical profile to every substance.

    Parameters
    ----------
    inventory : emiproc inventory
        Inventory on the target grid.
    height : float, default 2.0
        Representative emission height.
    """
    _require_emiproc()
    from emiproc.profiles.vertical_profiles import VerticalProfile

    for substance in inventory.substances:
        inventory.set_profile(
            VerticalProfile(np.array([1.0]), height=np.array([height])),
            substance=substance,
        )
    return inventory


def country_profiles_to_cells(inventory, aux_data_path, icon_grid_name, weight_filepath=None):
    """
    Convert country-level EDGAR temporal profiles to ICON cell profiles.

    Parameters
    ----------
    inventory : emiproc inventory
        Remapped inventory.
    aux_data_path : str or path-like
        EDGAR auxiliary profile directory.
    icon_grid_name : str
        Name of the target ICON grid.
    weight_filepath : str or path-like, optional
        Country-mask weights file. Defaults to the notebook naming convention.
    """
    _require_emiproc()
    from emiproc.inventories.utils import country_to_cells

    aux_data_path = Path(aux_data_path)
    if weight_filepath is None:
        weight_filepath = aux_data_path / "country_mask_weights_country_{}".format(icon_grid_name)
    return country_to_cells(inventory, country_mask_kwargs={"weight_filepath": Path(weight_filepath)})


def temporal_scaled_array(inventory, time_range, sum_over_cells=True):
    """
    Return a temporally scaled emissions array for a time range.

    Parameters
    ----------
    inventory : emiproc inventory
        Inventory with temporal profiles.
    time_range : pandas.DatetimeIndex or sequence
        Output timestamps.
    sum_over_cells : bool, default True
        Forwarded to emiproc.
    """
    _require_emiproc()
    from emiproc.exports.utils import get_temporally_scaled_array

    return get_temporally_scaled_array(
        inventory,
        time_range=pd.DatetimeIndex(time_range),
        sum_over_cells=sum_over_cells,
    )


def interpolate_profiles(inventory, output_type="THREE_CYCLES"):
    """
    Interpolate temporal profiles to an emiproc export profile type.

    Parameters
    ----------
    inventory : emiproc inventory
        Inventory with temporal profiles.
    output_type : str or enum, default "THREE_CYCLES"
        ``TemporalProfilesInterpolated`` value.
    """
    _require_emiproc()
    from emiproc.inventories.utils import TemporalProfilesInterpolated, interpolate_temporal_profiles

    return interpolate_temporal_profiles(
        inventory,
        output_type=_enum_value(TemporalProfilesInterpolated, output_type),
    )


def export_icon_oem(inventory, icon_grid_file, output_dir, temporal_profiles_type="THREE_CYCLES"):
    """
    Export an ICON OEM emissions dataset with emiproc.

    Parameters
    ----------
    inventory : emiproc inventory
        Inventory already remapped to the ICON grid.
    icon_grid_file : str or path-like
        ICON grid file used by ICON.
    output_dir : str or path-like
        Directory for OEM files.
    temporal_profiles_type : str or enum, default "THREE_CYCLES"
        ``TemporalProfilesTypes`` value passed to emiproc.
    """
    _require_emiproc()
    from emiproc.exports.icon import TemporalProfilesTypes, export_icon_oem as _export_icon_oem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _export_icon_oem(
        inv=inventory,
        icon_grid_file=Path(icon_grid_file),
        output_dir=output_dir,
        temporal_profiles_type=_enum_value(TemporalProfilesTypes, temporal_profiles_type),
    )


def map_edgar_to_icon(
    edgar_directory=None,
    icon_grid_file=None,
    output_dir=None,
    aux_data_path=None,
    year=None,
    substances=None,
    download=True,
    pattern="*.nc",
    weights_file=".remap_weights",
    vertical_height=2.0,
    temporal_profiles_type="THREE_CYCLES",
    open_output=True,
    output_filename="oem_gridded_emissions.nc",
    edgar_dir=None,
):
    """
    Run the EDGAR-to-ICON OEM workflow from local EDGAR files.

    Parameters
    ----------
    edgar_directory : str or path-like
        Directory containing EDGAR files, or download target when
        ``download=True``.
    icon_grid_file : str or path-like
        ICON grid file used for remapping and OEM export.
    output_dir : str or path-like
        Directory where emiproc writes OEM output.
    aux_data_path : str or path-like, optional
        EDGAR auxiliary temporal profile directory. If omitted, only constant
        vertical profiles are set.
    year : int, optional
        EDGAR inventory year.
    substances : sequence of str, optional
        Substances to download when ``download=True``.
    download : bool, default True
        Download EDGAR files before loading the inventory.
    pattern : str, default "*.nc"
        EDGAR file pattern.
    weights_file : str or path-like, default ".remap_weights"
        Remapping weights cache file.
    vertical_height : float, default 2.0
        Constant vertical profile height.
    temporal_profiles_type : str or enum, default "THREE_CYCLES"
        Temporal profile type for interpolation and OEM export.
    open_output : bool, default True
        If True, open and return the gridded OEM NetCDF output.
    output_filename : str, default "oem_gridded_emissions.nc"
        Expected gridded OEM output filename.

    Returns
    -------
    xarray.Dataset or pathlib.Path
        Opened gridded emissions dataset when ``open_output=True``; otherwise
        the gridded emissions path. Temporal and vertical profile files remain
        in ``output_dir``.

    Examples
    --------
    >>> from artist.oem import map_edgar_to_icon
    >>> gridded_emissions = map_edgar_to_icon(
    ...     edgar_directory="./edgar",
    ...     icon_grid_file="icon_grid.nc",
    ...     output_dir="./output",
    ...     year=2022,
    ...     substances=["CH4", "CO2"],
    ... )
    """
    if edgar_directory is None:
        edgar_directory = edgar_dir
    if edgar_directory is None:
        raise TypeError("map_edgar_to_icon() missing required edgar_directory argument.")
    if icon_grid_file is None:
        raise TypeError("map_edgar_to_icon() missing required icon_grid_file argument.")
    if output_dir is None:
        raise TypeError("map_edgar_to_icon() missing required output_dir argument.")

    edgar_directory = Path(edgar_directory)
    output_dir = Path(output_dir)

    if download:
        if substances is None:
            raise ValueError("substances must be provided when download=True.")
        download_edgar(edgar_directory, year=year, substances=substances)

    inventory = load_edgar_inventory(edgar_directory, pattern=pattern, year=year, substances=substances)
    if aux_data_path is not None:
        inventory = attach_edgar_temporal_profiles(inventory, aux_data_path)

    icon_grid_file, icon_grid = get_icon_grid(icon_grid_file, download=False)
    remapped = remap_to_icon(inventory, icon_grid, weights_file=weights_file)
    remapped = set_constant_vertical_profiles(remapped, height=vertical_height)

    if aux_data_path is not None:
        remapped = country_profiles_to_cells(remapped, aux_data_path, icon_grid.name)
        remapped = interpolate_profiles(remapped, output_type=temporal_profiles_type)

    export_icon_oem(
        remapped,
        icon_grid_file=icon_grid_file,
        output_dir=output_dir,
        temporal_profiles_type=temporal_profiles_type,
    )

    output_path = output_dir / output_filename
    if open_output:
        return _open_oem_dataset(output_path)
    return output_path


def map_edgar(
    edgar_directory=None,
    year=None,
    species=None,
    gridfile=None,
    output_dir="./output",
    aux_data_path=None,
    download=True,
    pattern="*.nc",
    weights_file=".remap_weights",
    vertical_height=2.0,
    temporal_profiles_type="THREE_CYCLES",
    open_output=True,
    edgar=None,
):
    """
    Map EDGAR emissions for selected species to an ICON grid.

    This is the high-level workflow intended for ``ds.oem.map_edgar(...)``.

    Parameters
    ----------
    edgar_directory : str or path-like
        Directory containing EDGAR files, or download target when
        ``download=True``.
    year : int
        EDGAR inventory year.
    species : str or sequence of str
        EDGAR species/substances, for example ``"CH4"`` or
        ``["CH4", "CO2"]``.
    gridfile : str or path-like
        ICON grid file used for remapping.
    output_dir : str or path-like, default "./output"
        Directory where OEM output is written.
    aux_data_path : str or path-like, optional
        EDGAR auxiliary temporal profile directory.
    download : bool, default True
        Download EDGAR files for ``species`` before loading the inventory.
    pattern : str, default "*.nc"
        Fallback EDGAR file pattern.
    weights_file : str or path-like, default ".remap_weights"
        Remapping weights cache file.
    vertical_height : float, default 2.0
        Constant vertical profile height.
    temporal_profiles_type : str or enum, default "THREE_CYCLES"
        Temporal profile type for interpolation and OEM export.
    open_output : bool, default True
        If True, open and return the gridded OEM NetCDF output.

    Returns
    -------
    xarray.Dataset or pathlib.Path
        Opened gridded emissions dataset when ``open_output=True``; otherwise
        the gridded emissions path. Temporal and vertical profile files remain
        in ``output_dir``.

    Examples
    --------
    >>> from artist.oem import map_edgar
    >>> gridded_emissions = map_edgar(
    ...     edgar_directory="./edgar",
    ...     year=2022,
    ...     species=["CH4", "CO2"],
    ...     gridfile="icon_grid.nc",
    ... )
    """
    if edgar_directory is None:
        edgar_directory = edgar
    if edgar_directory is None:
        raise TypeError("map_edgar() missing required edgar_directory argument.")
    if year is None:
        raise TypeError("map_edgar() missing required year argument.")
    if species is None:
        raise TypeError("map_edgar() missing required species argument.")
    if gridfile is None:
        raise TypeError("map_edgar() missing required gridfile argument.")

    if isinstance(species, str):
        species = [species]

    return map_edgar_to_icon(
        edgar_directory=edgar_directory,
        icon_grid_file=gridfile,
        output_dir=output_dir,
        aux_data_path=aux_data_path,
        year=year,
        substances=species,
        download=download,
        pattern=pattern,
        weights_file=weights_file,
        vertical_height=vertical_height,
        temporal_profiles_type=temporal_profiles_type,
        open_output=open_output,
    )


def plot_raw_edgar(
    edgar_directory=None,
    year=None,
    species=None,
    pattern="*.nc",
    total_only=True,
    cmap="magma",
    edgar=None,
    coarsen=10,
    ax=None,
    **kwargs,
):
    """
    Plot raw EDGAR inventory data before ICON remapping.

    Parameters
    ----------
    edgar_directory : str or path-like or emiproc inventory
        EDGAR directory or an already loaded emiproc inventory.
    year : int, optional
        Inventory year, used when ``edgar_directory`` is a directory.
    species : str or sequence of str, optional
        Species used to narrow local EDGAR files when possible.
    pattern : str, default "*.nc"
        EDGAR file pattern.
    total_only : bool, default True
        Forwarded to ``emiproc.plots.plot_inventory``.
    cmap : str, default "magma"
        Matplotlib colormap.
    coarsen : int, default 10
        Aggregate this many raw EDGAR cells in each horizontal direction before
        plotting. This keeps global EDGAR plots responsive.
    ax : matplotlib.axes.Axes or sequence, optional
        Target axes for ARTist-native raw NetCDF plotting.
    **kwargs
        Extra keyword arguments passed to plotting.

    Examples
    --------
    >>> from artist.oem import plot_raw_edgar
    >>> plot_raw_edgar(edgar_directory="./edgar", year=2022, species=["CH4"])
    """
    import matplotlib.pyplot as plt

    if edgar_directory is None:
        edgar_directory = edgar
    if edgar_directory is None:
        raise TypeError("plot_raw_edgar() missing required edgar_directory argument.")

    if isinstance(edgar_directory, (str, Path)):
        substances = [species] if isinstance(species, str) else species
        if substances is None:
            raise ValueError("species is required when plotting raw EDGAR files from a directory.")
        substances = list(substances)
        edgar_directory = Path(edgar_directory)
        axes = np.atleast_1d(ax).tolist() if ax is not None else None
        if axes is not None and len(axes) < len(substances):
            raise ValueError("Not enough axes were provided for the requested species.")
        if axes is None:
            _, axes = plt.subplots(len(substances), 1, squeeze=False, figsize=(10, 3.8 * len(substances)))
            axes = axes.ravel().tolist()

        for axis, substance in zip(axes, substances):
            files = sorted(edgar_directory.glob("*{}*.nc".format(substance)))
            if not files:
                files = sorted(edgar_directory.glob(pattern))
            total = None
            units = None
            for path in files:
                try:
                    with xr.open_dataset(path) as dataset:
                        if "emissions" not in dataset:
                            continue
                        data = dataset["emissions"]
                        if str(data.attrs.get("substance", substance)) != substance:
                            continue
                        if year is not None and str(data.attrs.get("year", year)) != str(year):
                            continue
                        units = data.attrs.get("units", units)
                        data = data.load()
                except OSError:
                    continue
                total = data if total is None else total + data

            if total is None:
                raise FileNotFoundError(
                    "No raw EDGAR emission files found for species {} in {}.".format(
                        substance,
                        edgar_directory,
                    )
                )

            if coarsen and coarsen > 1:
                total = total.coarsen(lat=coarsen, lon=coarsen, boundary="trim").sum()
            positive = total.where(total > 0)
            image = axis.pcolormesh(
                total["lon"],
                total["lat"],
                positive,
                shading="auto",
                cmap=cmap,
                **kwargs,
            )
            axis.set_title("Raw EDGAR {} emissions{}".format(substance, " ({})".format(year) if year else ""))
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            label = "Emissions"
            if units:
                label = "{} [{}]".format(label, units)
            axis.figure.colorbar(image, ax=axis, label=label)
        return axes[0] if len(axes) == 1 else axes

    _require_emiproc()
    from emiproc.plots import plot_inventory

    before_figures = set(plt.get_fignums())
    ax = plot_inventory(edgar_directory, total_only=total_only, cmap=cmap, **kwargs)
    if ax is None:
        new_figures = [plt.figure(num) for num in plt.get_fignums() if num not in before_figures]
        axes = [figure.axes[0] for figure in new_figures if figure.axes]
        if axes:
            ax = axes[0] if len(axes) == 1 else axes
        else:
            ax = plt.gca()
    return ax


def _plot_dataarray_from_dataset(emissions, variable=None, selectors=None):
    if isinstance(emissions, xr.DataArray):
        da = emissions
    else:
        if variable is None:
            data_vars = list(emissions.data_vars)
            if not data_vars:
                raise ValueError("emissions dataset has no data variables to plot.")
            variable = data_vars[0]
        da = emissions[variable]

    if selectors:
        da = da.sel(selectors)

    cell_dim = None
    for dim in ("ncells", "cell", "cell_index"):
        if dim in da.dims:
            cell_dim = dim
            break
    if cell_dim is None:
        matches = [dim for dim in da.dims if da.sizes[dim] > 3]
        if len(matches) == 1:
            cell_dim = matches[0]
        else:
            raise ValueError("Could not identify the ICON cell dimension in the emissions data.")

    indexers = {dim: 0 for dim in da.dims if dim != cell_dim and da.sizes[dim] > 1}
    if indexers:
        da = da.isel(indexers)
    return da.squeeze(drop=True), cell_dim


def plot_mapped_emissions(
    emissions,
    gridfile,
    variable=None,
    selectors=None,
    ax=None,
    cmap="magma",
    vrange=None,
    ltranslon=False,
    add_colorbar=True,
    edgecolor="face",
    linewidth=0.0,
    projection=None,
    map_extent=None,
):
    """
    Plot gridded OEM emissions after mapping to an ICON grid.

    Parameters
    ----------
    emissions : xarray.Dataset or xarray.DataArray
        Gridded OEM emissions returned by ``map_edgar`` or opened from
        ``oem_gridded_emissions.nc``.
    gridfile : str or path-like
        ICON grid file matching the OEM output.
    variable : str, optional
        Variable to plot. Defaults to the first data variable.
    selectors : dict, optional
        Coordinate selections applied before plotting.
    ax : matplotlib.axes.Axes, optional
        Target axes. A new axes is created when omitted.
    cmap : str, default "magma"
        Matplotlib colormap.
    vrange : sequence, optional
        Two-value color normalization range.
    ltranslon : bool, default False
        Transform grid longitudes to [0, 360].
    add_colorbar : bool, default True
        Add a colorbar to the axes.
    edgecolor : str, default "face"
        Polygon edge color.
    linewidth : float, default 0.0
        Polygon edge width.
    projection : cartopy.crs.Projection, optional
        Map projection used by the target axes. When provided, or when `ax` is
        a Cartopy GeoAxes, polygons are interpreted as lon/lat coordinates and
        transformed from PlateCarree.
    map_extent : sequence, optional
        `[lon_min, lon_max, lat_min, lat_max]` plot extent.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the mapped emission plot.

    Examples
    --------
    >>> from artist.oem import plot_mapped_emissions
    >>> ax = plot_mapped_emissions(gridded_emissions, "icon_grid.nc")
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.cm as cm
    from matplotlib.collections import PolyCollection

    _, vlon, vlat, _, _ = add_grid(gridfile, ltranslon=ltranslon)
    vertices_lon = np.asarray(vlon.values)
    vertices_lat = np.asarray(vlat.values)
    if vertices_lon.shape[0] != vertices_lat.shape[0]:
        raise ValueError("ICON grid vertex longitude/latitude arrays are not aligned.")

    da, cell_dim = _plot_dataarray_from_dataset(emissions, variable=variable, selectors=selectors)
    values = np.asarray(da.transpose(cell_dim).values).ravel()
    if values.size != vertices_lon.shape[0] and values.size == vertices_lon.shape[-1]:
        vertices_lon = vertices_lon.T
        vertices_lat = vertices_lat.T
    if values.size != vertices_lon.shape[0]:
        raise ValueError(
            "Emission values and ICON grid cells are not aligned: got {} values "
            "and {} grid cells.".format(values.size, vertices_lon.shape[0])
        )

    valid = np.isfinite(values)
    values = values[valid]
    vertices_lon = vertices_lon[valid]
    vertices_lat = vertices_lat[valid]

    if ax is None:
        _, ax = plt.subplots()

    if vrange is not None and len(vrange) > 0:
        norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[-1])
    else:
        norm = mpl.colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = scalar_map.to_rgba(values)
    polygons = [np.column_stack((vertices_lon[i], vertices_lat[i])) for i in range(vertices_lon.shape[0])]

    collection = PolyCollection(
        polygons,
        facecolor=colors,
        closed=True,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=_plate_carree_transform(ax, projection),
    )
    ax.add_collection(collection, autolim=True)
    if map_extent is not None:
        if hasattr(ax, "set_extent"):
            transform = _plate_carree_transform(ax, projection)
            ax.set_extent(map_extent, crs=transform)
        else:
            ax.set_xlim([map_extent[0], map_extent[1]])
            ax.set_ylim([map_extent[2], map_extent[3]])
    elif hasattr(ax, "projection"):
        ax.set_global()
    else:
        ax.autoscale_view()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if add_colorbar:
        plt.colorbar(scalar_map, ax=ax)
    return ax


def _plate_carree_transform(ax=None, projection=None):
    if projection is None and not hasattr(ax, "projection"):
        return None
    try:
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError(
            "Cartopy is required for projected OEM emission plots. "
            "Install cartopy or plot on regular Matplotlib axes."
        ) from exc
    return ccrs.PlateCarree()
