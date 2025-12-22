"""Tests for arpes.plotting public API (__init__.py).

These tests verify that:
- All names in __all__ are importable via lazy import.
- Lazy import resolves objects from the correct modules.
- Invalid attribute access raises AttributeError.
- Basic import of the plotting package does not trigger circular imports.
"""

from __future__ import annotations

import pytest

from arpes import plotting


def test_plotting_module_importable() -> None:
    """The plotting package itself should be importable."""
    # If this fails, circular imports are likely broken.
    assert plotting is not None


@pytest.mark.parametrize("name", plotting.__all__)
def test_public_api_importable(name: str) -> None:
    """All names listed in __all__ should be accessible via lazy import."""
    obj = getattr(plotting, name)
    assert obj is not None


@pytest.mark.parametrize(
    "name, expected_module_prefix",
    [
        ("h_gradient_fill", "arpes.plotting.decoration"),
        ("v_gradient_fill", "arpes.plotting.decoration"),
        ("SmoothingApp", "arpes.plotting.ui"),
        ("DifferentiateApp", "arpes.plotting.ui"),
        ("plot_dos", "arpes.plotting.dos"),
        ("plot_core_levels", "arpes.plotting.dos"),
        ("spin_polarized_spectrum", "arpes.plotting.spin"),
        ("flat_stack_plot", "arpes.plotting.stack_plot"),
        ("savefig", "arpes.plotting.utils"),
    ],
)
def test_public_api_resolves_correct_module(name: str, expected_module_prefix: str) -> None:
    """Lazy import should resolve objects from the correct module."""
    obj = getattr(plotting, name)
    assert obj.__module__.startswith(expected_module_prefix)


def test_invalid_attribute_raises_attribute_error() -> None:
    """Accessing an unknown attribute should raise AttributeError."""
    with pytest.raises(AttributeError):
        _ = plotting.this_attribute_does_not_exist


def test_hasattr_behavior() -> None:
    """hasattr should reflect lazy-loaded public API correctly."""
    assert hasattr(plotting, "plot_dispersion")
    assert not hasattr(plotting, "definitely_not_here")
