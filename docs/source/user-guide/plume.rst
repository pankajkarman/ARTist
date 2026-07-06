Plume Diagnostics
=================

The ``da.art`` accessor contains xarray-native plume arithmetic methods adapted
from ICON-ART analysis scripts.

Select A Plume
--------------

.. code-block:: python

   plume = ds["ash_mixed_acc"].art.select_plume(1e-9)

Plume Height Diagnostics
------------------------

.. code-block:: python

   top = plume.art.plume_top(ds["z_mc"])
   bottom = plume.art.plume_bottom(ds["z_mc"])
   max_height = plume.art.max_conc_height(ds["z_mc"])

Center Of Mass
--------------

.. code-block:: python

   center = plume.art.plume_center(
       ds["cell_volume"],
       ds["z_mc"],
       dim=("height", "ncells"),
   )

Value At Plume Top
------------------

.. code-block:: python

   temp_at_top = plume.art.value_at_plume_top(ds["z_mc"], ds["temp"])
