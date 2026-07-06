Development
===========

Build Documentation Locally
---------------------------

Install documentation dependencies:

.. code-block:: bash

   pip install -r docs/requirements.txt

Build HTML pages:

.. code-block:: bash

   cd docs
   make html

On Windows, use:

.. code-block:: bat

   cd docs
   make.bat html

The generated site is written to ``docs/build/html``.

Documentation Style
-------------------

ARTist docstrings use NumPy-style sections:

.. code-block:: text

   Parameters
   ----------
   name : type
       Description.

   Returns
   -------
   type
       Description.

   Examples
   --------
   >>> import artist
   >>> result = obj.accessor.method(...)

When adding new accessor methods, include a short example that can be copied
into a notebook.
