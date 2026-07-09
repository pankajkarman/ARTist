OEM Emissions
=============

The ``ds.oem`` accessor prepares ICON online emission module inputs from EDGAR
inventories using ``emiproc``. It can download or load EDGAR NetCDF files,
remap emissions to the ICON grid already attached with ``ds.icon.add_grid(...)``,
and write the gridded emissions plus temporal and vertical profile files used
by OEM.

``emiproc`` is an optional dependency; install it only when you need EDGAR/OEM
preprocessing:

.. code-block:: bash

   pip install emiproc

The ``emiproc`` path uses the usual geospatial and NetCDF stack, including
``geopandas``, ``shapely``, ``pyogrio``, ``netCDF4``, ``rasterio``, and
``dask``.

Map EDGAR To ICON
-----------------

.. code-block:: python

   import xarray as xr
   import artist

   ds = xr.Dataset()
   ds.icon.add_grid("icon_grid.nc")

   gridded_emissions = ds.oem.map_edgar(
       edgar_directory="./edgar",
       year=2022,
       species=["CH4", "CO2", "CO"],
       output_dir="./output",
       aux_data_path="./edgar/aux",
   )

``ds.icon.add_grid(...)`` remembers the ICON grid file path, so
``ds.oem.map_edgar(...)`` can use the same grid. The return value is the
gridded emissions dataset. Temporal profiles such as ``dayofweek.nc``,
``hourofday.nc``, and ``monthofyear.nc`` plus ``vertical_profiles.nc`` are
written by ``emiproc`` into ``output_dir``.

Plot Emissions
--------------

Plot the raw EDGAR inventory before remapping:

.. code-block:: python

   ds.oem.plot_raw_edgar()

``plot_raw_edgar`` reuses the EDGAR directory, year, and species from the last
``ds.oem.map_edgar(...)`` call. Pass them explicitly only when plotting a
different inventory.

Plot the gridded emissions after ICON remapping:

.. code-block:: python

   ax = ds.oem.plot_mapped_emissions(gridded_emissions)

Use Low-Level Helpers
---------------------

The same workflow is also available as functions in ``artist.oem``:

.. code-block:: python

   from artist.oem import load_edgar_inventory, get_icon_grid, remap_to_icon

   inv = load_edgar_inventory("./edgar", year=2022, substances=["CH4"])
   gridfile, icon_grid = get_icon_grid("icon_grid.nc", download=False)
   mapped = remap_to_icon(inv, icon_grid)
