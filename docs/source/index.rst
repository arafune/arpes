PyARPES corrected (V4)
=======================
**V4 Release: Non maintainer update**
After V3 release, the origiinal author/maintainer seems to have relinquished maintenance of PyArpes.
While I have posted several pull-requests and comments, very unfortunately, he has not responded at all.
Furthermore, he does not seems to conduct ARPES related scientific work.
That is likely th main reason he has not maintained it.
There is no motivation to maintain codes that is not being used.

Since the previous version release, the  python ecosystem has been improved significantly.
Conda is no longer necessarily the best system, especially for Macintosh users.
In fact, I believe that it can sometimes cause confusion.
Therefore,  I have decided to drop the Conda-related support. Instead,
I strongly recommend to use "rye".

While I have added many type hints in the codes to improve usability and maintainability,
I still feel that the current package is not very user-friencdly for less experienced Python users.
I recommend igor to analyze, especially for students.



**December 2020, V3 Release**: The current relase focuses on improving
usage and workflow for less experienced Python users, lifting version
incompatibilities with dependencies, and ironing out edges in the user
experience.

For the most part, existing users of PyARPES should have no issues
upgrading, but we now require Python 3.8 instead of 3.7. We now provide
a conda environment specification which makes this process simpler, see
the installation notes below. It is recommended that you make a new
environment when you upgrade.

.. raw:: html

   <figure>
     <iframe width="560" height="315" src="https://www.youtube.com/embed/Gd0qJuInzvE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
     </iframe>
     <figcaption>
       You can find more usage and example videos here.
     </figcaption>
   </figure>

PyARPES is an open-source data analysis library for angle-resolved
photoemission spectroscopic (ARPES) research and tool development. While
the scope of what can be achieved with PyARPES is general, PyARPES
focuses on creating a productive programming and analysis environment
for ARPES and its derivatives (Spin-ARPES, ultrafast/Tr-ARPES,
ARPE-microspectroscopy, etc).

As part of this mission, PyARPES aims to reduce the feedback cycle for
scientists between data collection and producing publication quality
analyses and figures. Additionally, PyARPES aims to be a platform on
which new types of ARPES and spectroscopic analyses can be rapidly
prototyped and tested.

For these reasons, PyARPES includes out of the box a **large variety of
analysis tools** for

1.  Applying corrections to ARPES data
2.  Doing gap analysis
3.  Performing sophisticated band analysis
4.  Performing rapid and automated curve fitting, even over several
    dataset dimensions
5.  Background subtraction
6.  Dataset collation and combination
7.  Producing common types of ARPES figures and reference figures
8.  Converting to momentum space
9.  Interactively masking, selecting, laying fits, and exploring data
10. Plotting data onto Brillouin zones

These are in addition to facilities for derivatives, symmetrization, gap
fitting, Fermi-Dirac normalization, the minimum gradient method, and
others.

By default, PyARPES supports a variety of data formats from synchrotron
and laser-ARPES sources including ARPES at the Advanced Light Source
(ALS), the data produced by Scienta Omicron GmbH’s “SES Wrapper”, data
and experiment files from Igor Pro (see in particular the section on
:doc:`importing Igor Data </igor-pro>`), NeXuS files, and others.
Additional data formats can be added via a user plugin system.

If PyARPES helps you in preparing a conference presentation or
publication, please respect the guidelines for citation laid out in the
notes on :doc:`user contribution </contributing>`. Contributions and
suggestions from the community are also welcomed warmly.

Secondary to providing a healthy and sane analysis environment, PyARPES
is a testbed for new analysis and correction techniques, and as such
ships with ``scikit-learn`` and ``open-cv`` as compatible dependencies
for machine learning. ``cvxpy`` can also be included for convex
optimization tools.

Copyright © 2018-2020 by Conrad Stansbury, all rights reserved. Logo
design, Michael Khachatrian

Copyright © after 2023 by Ryuichi Arafune, all rights reserved.


Gettinng started
================

See the section on the docs site about
`contributing <https://arpes.readthedocs.io/contributing>`__ for
information on adding to PyARPES and rebuilding documentation from
source.

* :doc:`installation`
* :doc:`migration-guide`
* :doc:`faq`
* :doc:`example-videos`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Installation + Technical Notes

   installation
   getting-started
   faq
   example-videos

API reference
=============

* :doc:`api`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API reference

   api


Tutorial
========

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Tutorial

   Example Data <notebooks/tutorial-data>
   Jupyter Crash Course <notebooks/jupyter-crash-course>
   Data Exploration <notebooks/basic-data-exploration>
   Data Manipulation <notebooks/data-manipulation-intermediate>
   `xarray` Extensions Pt. 1 <notebooks/custom-dot-s-functionality>
   `xarray` Extensions Pt. 2 <notebooks/custom-dot-t-functionality>
   Curve Fitting <notebooks/curve-fitting>
   Fermi Edge Corrections <notebooks/fermi-edge-correction>
   Momentum Conversion <notebooks/converting-to-kspace>
   Nano XPS Analysis <notebooks/full-analysis-xps>

Detailed Guides
===============
.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Tutorial


* :doc:`loading-data`
* :doc:`interactive`
* :doc:`workspaces`
* :doc:`statistics`
* :doc:`curve-fitting`
* :doc:`customization`
* :doc:`advanced-plotting`
* :doc:`writing-plugins-basic`
* :doc:`writing-plugins`
* :doc:`igor-pro`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Detailed Guides

   loading-data
   interactive
   workspaces
   statistics
   curve-fitting
   customization
   advanced-plotting
   writing-plugins-basic
   writing-plugins
   igor-pro

Plotting
========

* :doc:`stack-plots`
* :doc:`brillouin-zones`
* :doc:`fermi-surfaces`
* :doc:`3d-cut-plots`
* :doc:`spin-arpes`
* :doc:`tr-arpes`
* :doc:`annotations`
* :doc:`plotting-utilities`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Plotting

   stack-plots
   brillouin-zones
   fermi-surfaces
   3d-cut-plots
   spin-arpes
   tr-arpes
   annotations
   plotting-utilities

ARPES
=====

* :doc:`spectra`
* :doc:`momentum-conversion`
* :doc:`th-arpes`
* :doc:`single-particle-spectral`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: ARPES

   spectra
   momentum-conversion
   th-arpes
   single-particle-spectral



Reference
=========

* :doc:`migration-guide`
* :doc:`writing-plugins-basic`
* :doc:`writing-plugins`
* :doc:`data-provenance`
* :doc:`modeling`
* :doc:`cmp-stack`
* :doc:`contributing`
* :doc:`dev-guide`
* :doc:`api`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Reference

   migration-guide
   writing-plugins-basic
   writing-plugins
   data-provenance
   modeling
   cmp-stack
   contributing
   dev-guide
   api


Changelog
=========

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Changelog

   CHANGELOG
   
