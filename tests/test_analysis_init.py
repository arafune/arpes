import pytest
import types

import arpes.analysis as analysis


@pytest.mark.parametrize("name", analysis.__all__)
def test_lazy_import_all_attributes(name):
    """Test that all names in __all__ can be accessed via lazy import
    and return some object (function, class, etc.).
    """
    attr = getattr(analysis, name)
    assert attr is not None


def test_lazy_import_invalid_attribute():
    """
    Test that accessing a non-existent attribute raises AttributeError
    """
    with pytest.raises(AttributeError):
        _ = analysis.non_existent_attribute


def test_lazy_import_type_checking():
    """
    Test that TYPE_CHECKING imports do not break (optional, mostly static check)
    """
    assert isinstance(analysis.__all__, list)
    assert "fit_bands" in analysis.__all__
