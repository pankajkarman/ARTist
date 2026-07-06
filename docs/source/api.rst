API Reference
=============

Accessors
---------

ARTist registers xarray accessors when the package is imported:

.. code-block:: python

   import artist

Dataset accessor: ``ds.icon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ds.icon`` provides ICON dataset helpers:

``add_grid(gridfile, ltranslon=False)``
   Attach native ICON grid coordinates from a grid file.

``get_dz()``
   Compute model-layer thickness from interface heights.

``look_up(key="mixed")``
   Return dataset variable names containing a search key.

``distance(origin, destination, radius=6371.0)``
   Compute great-circle distance between lon/lat coordinates.

``nearest_gridpoints(coordinate)``
   Find nearest ICON native cells for lon/lat coordinates.

``show_slice_line(points, gridpoints, grid_stride=5)``
   Plot a vertical slice path and selected gridpoints.

``noncyl_xticks(ax, ticks, fontsize=None)``
   Draw longitude ticks for non-cylindrical Cartopy projections.

``noncyl_yticks(ax, ticks, fontsize=None)``
   Draw latitude ticks for non-cylindrical Cartopy projections.

``noncyl_gridlines(ax, xticks=None, yticks=None, fontsize=None, **gridline_kwargs)``
   Draw gridlines and edge ticks for non-cylindrical Cartopy maps.

DataArray grid accessor: ``da.icon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``da.icon`` provides native-grid helpers for individual fields:

``regrid(gridfile, lon, lat, method="linear", ltranslon=True)``
   Interpolate a native ICON field to a regular lon/lat grid.

``tri_data(gridfile, cmap=None, vrange=[], ltranslon=True)``
   Build native-cell triangle vertices and colors for plotting.

``tri_plot(gridfile, ax, cmap=None, vrange=[], ltranslon=False, add_colorbar=True, map_extent=None)``
   Plot a field on ICON triangular cells.

ART tracer accessor: ``da.art``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``da.art`` provides ICON-ART tracer diagnostics:

``tracer_load(rho, dz)``
   Compute column tracer load.

``select_plume(thresh, fill_value=0)``
   Keep values above a plume threshold.

``plume_center(v_cell, z_mc, dim=None, skipna=True)``
   Compute plume center height.

``plume_top(z_mc, height_dim="height", fill_value=0)``
   Compute the highest plume-cell height.

``plume_bottom(z_mc, height_dim="height", fill_value=0)``
   Compute the lowest plume-cell height.

``value_at_plume_top(z_mc, val, height_dim="height", fill_value=0)``
   Sample another field at the plume top.

``max_conc_height(z_mc, height_dim="height", fill_value=0)``
   Find the height where plume concentration is maximal.

``plot_hov(height, gridpoints, start_t=0, end_t=100, levels=80, point_size=90)``
   Plot a time-height view at one native gridpoint.

``plot_slice(height, gridpoints, t, deg_E_start, deg_E_end, deg_N, n=4000, levels=80, point_size=35)``
   Plot a vertical tracer slice.

``quick_plot(gridfile=None, cmap="coolwarm", levels=10, projection=None)``
   Quickly plot a field on a Cartopy map.

ART dataset accessor: ``ds.art``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ds.art`` provides dataset-level ICON-ART optical diagnostics:

``rayleigh_part(wavelength, height_ref=90, scale_height=1.0)``
   Compute Rayleigh extinction and backscatter coefficients.

``att_bsct(wavelength)``
   Compute attenuated backscatter for aerosol tracers.

``att_bsct_sph(wavelength)``
   Compute attenuated backscatter using spherical aerosol fractions.

``aod(wavelength)``
   Compute layer aerosol optical depth.

``aod_misr(wavelength, frac="all")``
   Compute MISR-style layer aerosol optical depth by fraction.

``ssa(wavelength)``
   Compute mass-weighted single-scattering albedo.

Utility Functions
-----------------

``artist.utils.add_grid(gridfile, ltranslon=False)``
   Open an ICON grid file and return converted vertex and cell-center
   coordinates.

``artist.utils.distance(origin, destination, radius=6371.0)``
   Compute great-circle distance in kilometers.
