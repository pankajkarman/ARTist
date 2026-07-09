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
   regional = ds.icon.sellonlat(-100, 40, -20, 60)

DataArray Accessor: ``da.icon``
-------------------------------

Use ``da.icon`` for native-grid operations on a single field:

.. code-block:: python

   da = ds["ash_mixed_acc"]
   regular = da.icon.regrid(lon, lat)
   triangles, colors, cmap = da.icon.tri_data()

OEM Dataset Accessor: ``ds.oem``
--------------------------------

Use ``ds.oem`` to map EDGAR emissions to an ICON grid for OEM input files:

.. code-block:: python

   ds = xr.Dataset()
   ds.icon.add_grid("icon_grid.nc")
   gridded_emissions = ds.oem.map_edgar(
       edgar_directory="./edgar",
       year=2022,
       species=["CH4", "CO2"],
   )
   ax = ds.oem.plot_mapped_emissions(gridded_emissions)

ART Tracer Accessor: ``da.art``
-------------------------------

Use ``da.art`` for ART tracer diagnostics:

.. code-block:: python

   plume = da.art.select_plume(1e-9)
   top = plume.art.plume_top(ds["z_mc"])
   load = da.art.tracer_load(ds["rho"], dz)
