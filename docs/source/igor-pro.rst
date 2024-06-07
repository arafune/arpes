Working with Igor
=================

What about Waves?
-----------------

Igor Pro’s ``wave`` abstraction proved to be incredibly useful in the
analysis of physics data. Unlike some other disciplines, in physics it
is import to keep track of the axis coordinates, dimensions, and units
for volumetric data, especially in ARPES.

One of the principal difficulties in making fluent and natural data
analysis suites for ARPES outside Igor Pro has been finding an adequate
replacement for this abstraction, as carrying around data and
coordinates separately is far too cumbersome. Thankfully, the Python
scientific programming community has built the excellent open source
library ``xarray`` <http://xarray.pydata.org/en/stable/>`_, which
offers first class support for labelled, unitful volumetric data.

You can learn about how PyARPES uses this library to provide a robust
data model for ARPES in the documentation on :doc:`PyARPES spectra </spectra>`.

Installing the patched ``igorpy``
---------------------------------

You’ll need to install a patched copy of ``igorpy`` to get Igor support
in PyARPES. You can do this with

.. code:: bash

   pip install https://github.com/arafune/igorpy

Importing Data from Igor
------------------------

In addition to offering a viable alternative for the Igor wave data
model, PyARPES offers limited support for loading a subset of Igor data,
notably binary waves and packed experiment files containing a single
wave into PyARPES using a patched copy of the
``igorpy`` <https://github.com/arafune/igorpy>`_ library.

An API for directly interfacing with Igor Pro binaries is available in
``arpes.load_pxt``.
