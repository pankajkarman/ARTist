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

Plume Mass And Columns
----------------------

Dataset-level helpers adapted from ``for_Dorsa.py`` are available from
``ds.art``:

.. code-block:: python

   plume_mass = ds.art.plume_mass("ash_mixed_acc", "ash_mixed_acc", 0.1)
   so2_du = ds.art.vmr_to_du("TRSO2_chemtr")

Value At Plume Top
------------------

.. code-block:: python

   temp_at_top = plume.art.value_at_plume_top(ds["z_mc"], ds["temp"])

Optical Diagnostics
-------------------

Dataset-level optical forward operators are available from ``ds.art``:

.. code-block:: python

   alpha, beta = ds.art.rayleigh_part(532)
   attenuated = ds.art.att_bsct(532)
   layer_aod = ds.art.aod(532)
   column_aod = layer_aod.sum("height")
   acc_aod = ds.art.aod_misr(532, frac="acc")
   single_scattering_albedo = ds.art.ssa(532)

Sulfate-only AOD and aerosol microphysics diagnostics are also available:

.. code-block:: python

   sulfate_aod = ds.art.sulfate_aod(532).sum("height")
   sulfate_aod_8547 = ds.art.saod(8547).sum("height")
   dcdt_acc, dcdt_coa, dcdt = ds.art.coating_fraction()
   r_eff_ash = ds.art.effective_radius("ash")
   r_eff_sulfate = ds.art.reff_sulfate()
