.. currentmodule:: arpes

This page documents the PyARPES API by generation from signatures and
docstrings. You can use this and the source code to supplement the remainder 
of the PyARPES documentation.

Please note that this is not a complete API documentation page. Less used and 
internal APIs are not documented here.

.. contents::

Data-loading functions
======================

.. autosummary::
   :toctree: generated/

   io.load_data
   io.load_example_data
   io.example_data


Xarray_extensions
=================

Establishes the PyARPES data model by extending the `xarray` types.

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

The `.G.` accessor:
    This a general purpose collection of tools which exists to provide conveniences over
    what already exists in the xarray data model. As an example, there are various tools
    for simultaneous iteration of data and coordinates here, as well as for vectorized
    application of functions to data or coordinates.

The `.F` accessor:
    This is an accessor which contains tools related to interpreting curve fitting
    results. In particular there are utilities for vectorized extraction of parameters,
    for plotting several curve fits, or for selecting "worst" or "best" fits according
    to some measure.

The .S accessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autoclass::   arpes.xarray_extensions.ARPESAngleProperty
   :members:
   :exclude_members: _*, __*
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESPhysicalProperty
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESInfoProperty
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESProvenanceProperty
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESPropertyBase
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESOffsetProperty
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESDataArrayAccessorBase
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESDataArrayAccessor
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.ARPESDatasetAccessor
   :members:
   :show-inheritance:



The .G accessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available as methods via ``.G`` accessor.


.. autoclass::   arpes.xarray_extensions.GenericAccessorBase
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.GenericDatasetAccessor
   :members:
   :show-inheritance:

.. autoclass::   arpes.xarray_extensions.GenericDataArrayAccessor
   :members:
   :show-inheritance:

The .F accessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available as methods via ``.F`` accessor.

.. autoclass:: arpes.xarray_extensions.ARPESDatasetFitToolAccessor
   :members:
   :show-inheritance:

.. autoclass:: arpes.xarray_extensions.ARPESFitToolsAccessor
   :members:
   :show-inheritance:
   

Momentum Conversion
===================

Small-Angle Approximated and Volumetric Related
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   utilities.conversion.core.convert_to_kspace
   utilities.conversion.forward.convert_coordinate_forward
   utilities.conversion.forward.convert_through_angular_point
   utilities.conversion.forward.convert_through_angular_pair

Exact Forward Transforms
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   utilities.conversion.forward.convert_coordinates_to_kspace_forward


Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   utilities.conversion.fast_interp.Interpolator
   utilities.conversion.bounds_calculations.full_angles_to_k
   utilities.conversion.remap_manipulator.remap_coords_to

Conversion Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not need to call these directly, but they are good references for anyone 
looking to implement a different kind of coordinate transform.

.. autosummary::
   :toctree: generated/

   utilities.conversion.base.CoordinateConverter
   utilities.conversion.kx_ky_conversion.ConvertKp
   utilities.conversion.kx_ky_conversion.ConvertKxKy
   utilities.conversion.kz_conversion.ConvertKpKz


General Analysis
================

Axis Methods
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.general.rebin
   analysis.general.symmetrize_axis
   analysis.general.condense
   preparation.axis_preparation.normalize_dim
   preparation.axis_preparation.sort_axis

Experimental Resolution Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.resolution.total_resolution_estimate
   analysis.resolution.thermal_broadening_estimate
   analysis.resolution.beamline_resolution_estimate
   analysis.resolution.analyzer_resolution_estimate
   


Self-Energy
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.self_energy.to_self_energy
   analysis.self_energy.fit_for_self_energy
   analysis.self_energy.estimate_bare_band
   analysis.self_energy.quasiparticle_lifetime
   

Time-Resolved ARPES Related
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.tarpes.find_t_for_max_intensity
   analysis.tarpes.relative_change
   analysis.tarpes.normalized_relative_change
   analysis.tarpes.build_crosscorrelation
   analysis.tarpes.delaytime_fs
   analysis.tarpes.position_mm_to_delaytime_fs


Spin-ARPES Related
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.sarpes.to_intensity_polarization
   analysis.sarpes.to_up_down
   analysis.sarpes.normalize_sarpes_photocurrent


Fermi Edge Related
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.gap.symmetrize
   analysis.gap.determine_broadened_fermi_distribution

Derivatives
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.derivative.dn_along_axis
   analysis.derivative.curvature
   analysis.derivative.minimum_gradient

Smoothing and Filtering
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.filters.gaussian_filter_arr
   analysis.filters.boxcar_filter_arr
   analysis.filters.savitzky_golay_filter

Deconvolution
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.deconvolution.deconvolve_ice
   analysis.deconvolution.deconvolve_rl
   analysis.deconvolution.make_psf1d
   analysis.deconvolution.make_psf

Masks
~~~~~

.. autosummary::
   :toctree: generated/

   analysis.mask.apply_mask
   analysis.mask.apply_mask_to_coords
   analysis.mask.polys_to_mask
   analysis.mask.raw_poly_to_mask

Fermi Surface Pockets
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.pocket.curves_along_pocket
   analysis.pocket.edcs_along_pocket
   analysis.pocket.radial_edcs_along_pocket
   analysis.pocket.pocket_parameters
   

Background Removal
~~~~~~~~~~~~~~~~~~

Shirley Backgrounds
-------------------

.. autosummary::
   :toctree: generated/

   analysis.shirley.calculate_shirley_background
   analysis.shirley.remove_shirley_background

Convex Hull Backgrounds
-----------------------

.. autosummary::
   :toctree: generated/
   
   analysis.background.calculate_background_hull
   analysis.background.remove_background_hull

Incoherent Backgrounds
----------------------

.. autosummary::
   :toctree: generated/
   
   correction.background.remove_incoherent_background
   correction.coords.shift_by


Array Alignment
~~~~~~~~~~~~~~~

Subpixel Correlation Based Alignment
------------------------------------

.. autosummary::
   :toctree: generated/

   analysis.align.align


Machine Learning
================

ML Based Decompositions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   analysis.decomposition.decomposition_along
   analysis.decomposition.pca_along
   analysis.decomposition.nmf_along
   analysis.decomposition.factor_analysis_along
   analysis.decomposition.ica_along

Deep Learning and PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for interpretation of results akin to those in fast.ai.

.. autosummary::
   :toctree: generated/

   deep_learning.interpret.Interpretation
   deep_learning.interpret.InterpretationItem

IO tools.

.. autosummary::
   :toctree: generated/

   deep_learning.io.from_portable_bin
   deep_learning.io.to_portable_bin

Transform pipelines.

.. autosummary::
   :toctree: generated/
   
   deep_learning.transforms.Identity
   deep_learning.transforms.ReversibleLambda
   deep_learning.transforms.ComposeBoth

Data Generation
~~~~~~~~~~~~~~~

Here are tools for simulating ARPES from theoretical models, including 
finite statistics, resolution modeling, and detector effects.

.. autosummary::
   :toctree: generated/

   simulation.sample_from_distribution
   simulation.cloud_to_arr
   simulation.SpectralFunctionMFL
   simulation.SpectralFunctionPhaseCoherent

   simulation.DetectorEffect
   simulation.NonlinearDetectorEffect


Data Loading Plugins
====================

.. autosummary::
   :toctree: generated/

   endstations.plugin.ALG_main.ALGMainChamber
   endstations.plugin.ALG_spin_ToF.SpinToFEndstation
   endstations.plugin.ANTARES.ANTARESEndstation
   endstations.plugin.Elettra_spectromicroscopy.SpectromicroscopyElettraEndstation
   endstations.plugin.example_data.ExampleDataEndstation
   endstations.plugin.fallback.FallbackEndstation
   endstations.plugin.HERS.HERSEndstation
   endstations.plugin.igor_export.IgorExportEndstation
   endstations.plugin.igor_plugin.IgorEndstation
   endstations.plugin.kaindl.KaindlEndstation
   endstations.plugin.MAESTRO.MAESTROMicroARPESEndstation
   endstations.plugin.MAESTRO.MAESTRONanoARPESEndstation
   endstations.plugin.MBS.MBSEndstation
   endstations.plugin.merlin.BL403ARPESEndstation
   endstations.plugin.SToF_DLD.SToFDLDEndstation
   endstations.plugin.SPD_main.SPDEndstation
   endstations.plugin.DSNP_UMCS.DSNP_UMCSEndstation


Curve Fitting
=============


General Curve Fitting Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   fits.utilities.result_to_hints

Models
~~~~~~

.. automodule:: arpes.fits
   :members:
.. autosummary::
   :toctree: generated/
   :recursive:

  fit_models.fermi_edge
  fit_models.dirac
  fit_models.bands
  fit_models.functional_forms
  fit_models.misc
  fit_models.two_dimensional

Plotting
========

Interactive Utilities: holoviews Based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: arpes.plotting
.. autosummary::
   :toctree: generated/

   holoviews.concat_along_phi_ui
   holoviews.profile_view
   holoviews.fit_inspection

General Utilities/Matplotlib Quality of Life
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An incomplete list of useful plotting utilities from ``arpes.utilities.plotting``.

.. autosummary::
   :toctree: generated/

   utils.path_for_plot
   utils.savefig
   utils.simple_ax_grid
   utils.invisible_axes
   utils.remove_colorbars
   utils.frame_with
   utils.unchanged_limits
   utils.plot_arr
   utils.imshow_mask
   utils.latex_escape
   utils.fancy_labels
   utils.summarize

Interactive Utilities: Matplotlib Based
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stack Plots
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   stack_plot.stack_dispersion_plot
   stack_plot.flat_stack_plot
   stack_plot.offset_scatter_plot

Spin-ARPES Plots
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   spin.spin_polarized_spectrum
   spin.spin_colored_spectrum
   spin.spin_difference_spectrum

Count-based and ToF Plots
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   tof.plot_with_std
   tof.scatter_with_std


Reference Plots
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   spatial.plot_spatial_reference
   spatial.reference_scan_spatial


Curve Fitting Plots
~~~~~~~~~~~~~~~~~~~

.. autosummary::

   fits.plot_fit
   fits.plot_fits
   parameter.plot_parameter

False Color Plots
~~~~~~~~~~~~~~~~~

.. autosummary::

   false_color.false_color_plot

Plotting with Brillouin Zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

   bz.bz_plot
   bz.plot_data_to_bz
   bz.overplot_standard
   bz.plot_plane_to_bz

Plot Annotations
~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated/

   annotations.annotate_cuts
   annotations.annotate_point
   annotations.annotate_experimental_conditions
