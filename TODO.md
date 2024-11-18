# TODO (2023/06/27)

As the original author does not seem to maintain this package, every bug must be
fixed by myself. Here is the guideline to maintain the code.

- More typing annotation
- Use ruff for linting and follow the message as possible
- Follow Zen of Python: "There should be one-- and preferably only one --obvious
  way to do it. Although that way may not be obvious at first unless you're Dutch."
- Don't be afraid to remove the function (I'll bet you don't use that one.)

- Consider using np.nan instead of None

- Follow the recent python (>=3.11)

  - User pathlib.Path instead os.path

- Check the functionality

  - ConversionKxKy class
  - .band_analysis_utils import param_getter, param_stderr_getter
  - Check type of the argument set at lf.Mmodel: Is it really lf.Model?
    lf.ModelResult is better?
