"""Establishes the PyARPES data model by extending the `xarray` types.

This is another core part of PyARPES. It provides a lot of extensions to
what comes out of the box in xarray. Some of these are useful generics,
generally on the .T extension, others collect and manipulate metadata,
interface with plotting routines, provide functional programming utilities,
etc.

If `f` is an ARPES spectrum, then `f.S` should provide a nice representation of your data
in a Jupyter cell. This is a complement to the text based approach that merely printing `f`
offers. Note, as of PyARPES v3.x.y, the xarray version has been bumped and this representation
is no longer necessary as one is provided upstream.

The main accessors are .S, .G, and .F.

The `.S` accessor:
    The `.S` accessor contains functionality related to spectroscopy. Utilities
    which only make sense in this context should be placed here, while more generic
    tools should be placed elsewhere.

The `.G` accessor:
    This a general purpose collection of tools which exists to provide conveniences over
    what already exists in the xarray data model. As an example, there are various tools
    for simultaneous iteration of data and coordinates here, as well as for vectorized
    application of functions to data or coordinates.

The `.F` accessor:
    This is an accessor which contains tools related to interpreting curve fitting
    results (assumed the return of S.modelfit).
    In particular there are utilities for vectorized extraction of parameters,
    for plotting several curve fits, or for selecting "worst" or "best" fits according
    to some measure.
"""

from __future__ import annotations

from .accessor.fit import ARPESDatasetFitToolAccessor, ARPESFitToolsAccessor
from .accessor.general import GenericDataArrayAccessor, GenericDatasetAccessor
from .accessor.spectroscopy import ARPESDataArrayAccessor, ARPESDatasetAccessor
