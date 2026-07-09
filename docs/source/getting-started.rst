Getting Started
===============

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install icon-artist

or directly from the repository:

.. code-block:: bash

   pip install git+https://github.com/pankajkarman/ARTist.git

Plotting utilities require Cartopy in addition to the core dependencies.
When using conda, Cartopy is usually easiest to install from conda-forge:

.. code-block:: bash

   conda install -c conda-forge cartopy

Dependencies
------------

The core package depends on ``numpy``, ``scipy``, ``pandas``, ``xarray``, and
``matplotlib``. These cover grid helpers, interpolation, diagnostics, and basic
plotting.

Optional plotting on geographic projections uses ``cartopy``:

.. code-block:: bash

   conda install -c conda-forge cartopy

EDGAR emission preprocessing for ICON's online emission module uses
``emiproc``:

.. code-block:: bash

   pip install emiproc

The ``emiproc`` workflow also uses geospatial and NetCDF packages such as
``geopandas``, ``shapely``, ``pyogrio``, ``netCDF4``, ``rasterio``, and
``dask``. Installing this stack from conda-forge is usually the most reliable
option on fresh scientific Python environments.

Importing ARTist
----------------

ARTist registers xarray accessors when imported:

.. code-block:: python

   import xarray as xr
   import artist

   ds = xr.open_dataset("icon_art_output.nc")
   ds.icon.add_grid("icon_grid.nc")
   da = ds["ash_mixed_acc"]

Native-Grid Plot
----------------

Plot a DataArray on native ICON triangular cells:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(12, 6))
   da.icon.tri_plot(ax)

Regrid To A Regular Lon/Lat Grid
--------------------------------

Interpolate from native ICON cells to a regular latitude-longitude grid:

.. code-block:: python

   import numpy as np

   lon = np.linspace(0, 20, 101)
   lat = np.linspace(40, 60, 81)

   regular = da.icon.regrid(lon, lat)
   regular.plot()
