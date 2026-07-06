xarray Accessors
================

ARTist follows xarray's accessor pattern. After ``import artist``, methods are
available from xarray objects without changing their type.

Dataset Accessor: ``ds.icon``
-----------------------------

Use ``ds.icon`` for dataset-wide ICON grid and metadata operations:

.. code-block:: python

   ds.icon.add_grid("icon_grid.nc")
   dz = ds.icon.get_dz()
   ash_variables = ds.icon.look_up("ash")
   distance_km = ds.icon.distance([8.4, 49.0], [13.4, 52.5])

DataArray Accessor: ``da.icon``
-------------------------------

Use ``da.icon`` for native-grid operations on a single field:

.. code-block:: python

   regular = da.icon.regrid("icon_grid.nc", lon, lat)
   triangles, colors, cmap = da.icon.tri_data("icon_grid.nc")

ART Tracer Accessor: ``da.art``
-------------------------------

Use ``da.art`` for ART tracer diagnostics:

.. code-block:: python

   plume = da.art.select_plume(1e-9)
   top = plume.art.plume_top(ds["z_mc"])
   load = da.art.tracer_load(ds["rho"], dz)
