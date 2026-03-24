# PyARPES repository instructions

## Build, test, and lint commands

- Install the development environment with `uv sync --all-extras --dev` from the repository root. This is the setup used in CI.
- Run the full test suite with `uv run pytest -vv --color=yes --cov=./ --cov-report=xml`. The project `pytest` config already sets `pythonpath=src` and disables Numba JIT for tests.
- Run a single test with `uv run pytest tests/test_provenance.py::test_attach_id -vv` or filter within a file via `uv run pytest tests/test_provenance.py -k attach_id -vv`.
- Run lint checks with `uv run ruff check src`. CI runs Ruff against `src`, while local hooks also run Ruff and Ruff format for Python files.
- Format Python code with `uv run ruff format src tests`.
- Build the docs with `uv run --directory docs make html`.

## High-level architecture

- `src/arpes/io.py` is the user-facing entry point for loading data. `load_data()` and `load_example_data()` return xarray objects and delegate actual file ingestion to endstation plugins.
- Plugin resolution flows through `src/arpes/endstations/registry.py` and `src/arpes/plugin_loader.py`. Concrete loaders live in `src/arpes/endstations/plugin/` and are selected by `location` / endstation alias.
- Runtime state is centralized in `src/arpes/configuration/manager.py` and exposed through `src/arpes/configuration/interface.py`. Workspace detection is filesystem-driven, and `local_config.py` can override config/settings for a local environment.
- The core data model is xarray-based. `src/arpes/xarray_extensions/` registers the `.S`, `.G`, and `.F` accessors, and most analysis, fitting, plotting, and coordinate-manipulation APIs assume those accessors are available.
- The package is notebook-first. `README.rst` and `plotting/ui/` show that interactive workflows are expected to run in Jupyter or marimo using HoloViews/Panel rather than Qt.
- Analysis logic is split by concern: numerical analysis in `src/arpes/analysis/`, corrections in `src/arpes/correction/`, coordinate/momentum conversion in `src/arpes/utilities/conversion/`, fit models in `src/arpes/fits/`, and plotting/UI in `src/arpes/plotting/`.
- Provenance is a cross-cutting concern. `src/arpes/provenance.py` attaches IDs, records parent relationships for derived data, and writes plot provenance alongside saved figures.

## Key conventions

- In scripts, call `arpes.initialize()` explicitly before relying on plugins, workspace detection, or xarray accessors. Import-time auto-initialization only happens in Jupyter / marimo environments.
- When loading files, prefer passing `location` explicitly to `load_data()`. The fallback loader will brute-force plugins and warn, but deterministic code in this repository usually specifies the endstation alias directly.
- New file readers should follow the existing endstation plugin pattern: subclass the appropriate `EndstationBase` family, define `PRINCIPAL_NAME` and `ALIASES`, export the class via the module `__all__`, and normalize metadata/coordinates in plugin post-processing.
- Treat xarray metadata as part of the public data model. Functions are expected to preserve meaningful attrs/coords, and loaded datasets conventionally expose a `.spectrum` data variable.
- If a new transformation creates derived xarray data, use the helpers in `src/arpes/provenance.py` (`update_provenance`, `provenance`, `provenance_multiple_parents`, plot provenance helpers) instead of ad hoc attrs updates.
- Put xarray-oriented convenience APIs on the appropriate accessor when they extend the data model: spectroscopy-specific behavior on `.S`, general-purpose helpers on `.G`, and model-fit helpers on `.F`. This matches the existing split in `src/arpes/xarray_extensions/`.
- Tests rely heavily on bundled fixture data and shared fixtures rather than external resources. Reuse `tests/conftest.py` fixtures such as `example_data` accessors and `sandbox_configuration` when touching loaders, provenance, or workspace-aware behavior.
- Workspace-aware code should assume the repository's established directory semantics: workspaces are detected from nearby `data/` or `Data/` directories, and `ConfigManager` derives related `datasets/` and `figures/` paths from the workspace root.
- Follow the repository style configuration from `pyproject.toml`: Ruff is the enforced linter/formatter, line length is 100, strings are double-quoted, and docstrings follow the Google convention.
