Plotting
========

Plotting methods are exposed on the same accessors as the data-processing
methods, while their implementation lives in ``artist.__plot``.

Native Triangular Cells
-----------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(12, 6))
   da.icon.tri_plot("icon_grid.nc", ax)

PolyCollection Backend
----------------------

For large native-grid fields or nested-domain outlines, ``da.viz.tricontourf``
can draw ICON cell polygons directly with Matplotlib ``PolyCollection`` after
``ds.icon.add_grid(...)`` has attached the grid:

.. code-block:: python

   import cartopy.crs as ccrs
   import matplotlib.pyplot as plt

   projection = ccrs.Robinson()
   fig, ax = plt.subplots(subplot_kw={"projection": projection})
   da.viz.tricontourf(
       ax,
       backend="polycollection",
       projection=projection,
       edgecolor="face",
   )

Quick Map
---------

.. code-block:: python

   ax = da.art.quick_plot(gridfile="icon_grid.nc")

Slice Line
----------

.. code-block:: python

   points = [[13.0, 52.0], [14.0, 53.0]]
   gridpoints = ds.icon.nearest_gridpoints(points)
   ax = ds.icon.show_slice_line(points, gridpoints)

Non-Cylindrical Gridlines
-------------------------

.. code-block:: python

   import cartopy.crs as ccrs
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
   ds.icon.noncyl_gridlines(
       ax,
       xticks=range(-180, 181, 60),
       yticks=range(-90, 91, 30),
   )

Vertical Slice
--------------

.. code-block:: python

   da.art.plot_slice(
       ds["z_mc"],
       gridpoints,
       t=0,
       deg_E_start=13,
       deg_E_end=14,
       deg_N=52,
   )
