Migration Guide
===============

Migrating to PyARPES v5.0
~~~~~~~~~~~~~~~~~~~~~~~~~

The most prominent difference from the prvious version of pyarpes is dropping `broadcast_model` for fitting.
Instead, use xarray-lmfit for curve fitting.

In the previous version, `broadcast_mode` is used like that:

.. code-block:: python

  params = {
      "ip1_center": {"value": 8.70, "min": 8.5, "max": 9.0},
      "ip1_gamma": {"value": 0.02, "min": 0.01, "max": 1.0},
      "ip1_sigma": {"value": resolution, "vary": False},
      "ip2_center": {"value": 9.2300, "min": 9.10, "max": 9.5},
      "ip2_gamma": {"value": 0.02, "min": 0.01, "max": 1.0},
      "ip2_sigma": {"value": resolution, "vary": False},
  }

  fit_ = broadcast_model(
      [VoigtModel, VoigtModel],
      arpes_data,
      "phi",
      params=params,
      prefixes=("ip1_", "ip2_"),
  )


They are migrated to:

.. code-block:: python

  params = {
      "ip1_center": {"value": 8.70, "min": 8.5, "max": 9.0},
      "ip1_gamma": {"value": 0.02, "min": 0.01, "max": 1.0},
      "ip1_sigma": {"value": resolution, "vary": False},
      "ip2_center": {"value": 9.2300, "min": 9.10, "max": 9.5},
      "ip2_gamma": {"value": 0.02, "min": 0.01, "max": 1.0},
      "ip2_sigma": {"value": resolution, "vary": False},
  }

  models = VoigtModel(prefix="ip1_") +  VoigtModel(prefix="ip2_")
  fit_ = arpes.S.modelfit("eV",  models=models, params=models.make_params(**mono_params), progress=True)


In the previous version, the return of broadcast_model is the Dataset that contains:

* results
* data
* residual
* norm_residual

In the current version, the return of S.modelfit depends on the type. (Note that the return of DataArray.S.modelfit() and Dataset.S.modelfit() is slightly different.)

* results -> modelfit_results (or spectrum_modelfit_results, more accurately [var]_modelfit_results)
* data -> modelfit_data  (or spectrum_modelfit_results, more accuratly  [var]_modelfit_results)
* residual  is not stored, but "modelfit_best_fit" is stored.  So residual can be easily got from modelfit_data - modelfit_best_fit
* norm_residual is not stored, but norm_residual can be easily got from (modelfit_data - modelfit_best_fit)/modelfit_data




Migrating to PyARPES v3
~~~~~~~~~~~~~~~~~~~~~~~

You no longer need to provide data spreadsheets. See the documentation at :doc:`loading-data` for details
on the the data loading API.

Many improvements have been made to performance. For the most part, these changes are completely 
transparent, as in momentum conversion which is 10-50x faster than in PyARPES v2. However, PyARPES
v3 uses multiprocessing for large groups of curve fits, through the `parallel=True/False` kwarg to 
`arpes.fits.utilities.broadcast_model`. If you do not want to use parallel curve fitting, simply pass
`False` to this kwarg when you do your curve fitting.

A downside to parallel curve fitting is that there is a substantial memory overhead: about 200MB / core
on your computer. As most high-core computers also have more memory headroom, we felt this an appropriate 
default behavior. Again, you can avoid the overhead with the parallelization kwarg.

For more detailed changes see the :doc:`CHANGELOG`.
