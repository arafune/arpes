Major Changes from 3.0.1

- Most important change:

  - Use correct method to convert from the angle to moementum. (Original approach was found to be incorrect)

- New feature
  - Provide SPD_main.py & prodigy_itx.py
  - Provide IF\_\_UMCS.py & prodigy_xy.py
  - Introduce Type annotation, which is major trend in the current python coding.
    - Users are requested to know type of the object treated. Especially, the users should know the difference between xr.DataArray & xr.Dataset
  - Introduce a new attrs, "energy_notation", which determines either "Kinetic" energy or "Binding" energy
    - Add new method: S.switch_energy_notation(self, nonlinear_order=1)

* Dataset.S.spectra returns the list of the xr.DataArrays whose dims contains "eV". (See xarray_extensions.py)
  Add a new method S.swap_angle_unit() to change the angle unit (deg <-> radian)

- Coding guide line

* Do not carelessly set default=None
* Replace algorithms to make them simpler and more efficient

  - stack_plot.py/flat_stack_plot
  - analysis/general.py/rebin

* Removing

  Many files/methods/class have been removed because of inconsistency, legacy style, useless. Here is just a short list the ones I remember now:

  - Remove arpes.all

    - Certainly, this it is indeed a lazy and carefree approach, but it's too rough method that leads to a bugs and does not mathc the current pythonic style.

  - Remove utilities/attrs.py

    - The functions in this module have not been used and are unlikely to be used in the future.

  - modules that use the Bokeh.

    There is a dependency problem among bokeh, tornard, and Jupyter, which I cannot fix because I'm haven't use Bokeh.

    - arpes/plotting/band_tool.py
    - arpes/plotting/curvature_tool.py
    - arpes/plotting/fit_inspection_tool.py
    - arpes/plotting/comparison_tool.py
    - arpes/plotting/dyn_tool.py
    - arpes/plotting/interactive_utils.py
    - arpes/plotting/interactive.py
    - arpes/plotting/path_tool.py
    - arpes/plotting/mask_tool.py

  - Remove MappableDict class
  - Remove overlapped_stack_dispersion_plot
    - use stack_dispersion_plot with appropriate args
  - Remove G.extent in xarray_extensions, which is not so usuful
  - Remove scan_row property
  - Remove original_id method, as I cannot figure out the purpose.
  - Remove lmfit_plot.py. The original ModelResult.plot() is sufficiently useful, and no reason for keeping to maintain this patched version.
  - Remove condensed_attrs: We should not enfoce camelcase on attributes, while original version did. Rather, the snake_case would be better from the modern pythonic viewpoint.
  - Remove `trace` arg for debugging. This technique may be sharp, but not so well fitted the current python trend; typing oriented.

Fix from 3.0.1

- bug of concatenating in broadcast_model
- Import error in BZ plotting example #7

Minor Changes from 3.0.1

- Remove beta arg from filters.curvature
