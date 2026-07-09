Working With ICON Grids
=======================

ICON grid files store cell centers and vertices in radians. ARTist provides
helpers to convert those coordinates to degrees and use them with xarray data.

Attach Grid Coordinates
-----------------------

.. code-block:: python

   ds.icon.add_grid("icon_grid.nc")

   clon = ds.icon.clon
   clat = ds.icon.clat

Find Nearest Native Cells
-------------------------

.. code-block:: python

   points = [[13.0, 52.0], [14.0, 53.0]]
   gridpoints = ds.icon.nearest_gridpoints(points)

Regrid A DataArray
------------------

.. code-block:: python

   ds.icon.add_grid("icon_grid.nc")
   regular = ds["ash_mixed_acc"].icon.regrid(lon, lat)
