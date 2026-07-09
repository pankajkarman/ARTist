ARTist documentation
====================

.. raw:: html

   <section class="artist-hero">
     <p class="artist-tagline">
       ARTist makes working with ICON-ART output in Python simple, xarray-native,
       and ready for post-processing workflows.
     </p>
     <p class="artist-meta">Version: 0.0.1</p>
     <p class="artist-links">
       Useful links:
       <a href="https://github.com/pankajkarman/ARTist">Code Repository</a> |
       <a href="https://github.com/pankajkarman/ARTist/issues">Issues</a> |
       <a href="https://github.com/pankajkarman/ARTist/releases">Releases</a>
     </p>
   </section>

Importing :mod:`artist` registers accessors on xarray objects:

.. code-block:: python

   import xarray as xr
   import artist

   ds = xr.open_dataset("icon_art_output.nc")
   da = ds["ash_mixed_acc"]

.. raw:: html

   <section class="artist-card-grid">
     <article class="artist-card">
       <div class="artist-card-icon">▶</div>
       <h2>Get started!</h2>
       <p>New to ARTist? Start here with installation instructions and a brief overview of the accessor workflow.</p>
       <p><a href="getting-started.html">Getting Started</a></p>
     </article>
     <article class="artist-card">
       <div class="artist-card-icon">▦</div>
       <h2>User guide</h2>
       <p>Work with ICON grids, ART tracers, plume diagnostics, native-grid plots, regular lon/lat regridding, and EDGAR/OEM preprocessing.</p>
       <p><a href="user-guide/index.html">User Guide</a></p>
     </article>
     <article class="artist-card">
       <div class="artist-card-icon">ƒ</div>
       <h2>API reference</h2>
       <p>Review public accessors, utility functions, plotting helpers, parameters, return values, and examples.</p>
       <p><a href="api.html">API Reference</a></p>
     </article>
     <article class="artist-card">
       <div class="artist-card-icon">+</div>
       <h2>Contribute</h2>
       <p>Build the documentation locally and follow the docstring style when adding new accessor methods.</p>
       <p><a href="development.html">Development</a></p>
     </article>
   </section>

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   user-guide/index
   api
   development
