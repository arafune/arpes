"""Top level module for PyARPES."""

# pylint: disable=unused-import
from __future__ import annotations

import warnings

# Use both version conventions for people's sanity.
VERSION = "4.0.0 beta1"
__version__ = VERSION


__all__ = ["check", "__version__"]


def check() -> None:
    """Verifies certain aspects of the installation and provides guidance broken installations."""

    def verify_qt_tool() -> str | None:
        pip_command = "pip install pyqtgraph"
        warning = (
            "Using qt_tool, the PyARPES version of Image Tool, requires "
            f"pyqtgraph and PySide6:\n\n\tYou can install with: {pip_command}"
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pyqtgraph
        except ImportError:
            return warning
        return None

    def verify_igor_pro() -> str | None:
        pip_command = "pip install https://github.com/arafune/igorpy/tarball/bbfaea#egg=igor-0.3.1"
        warning = f"For Igor support, install igorpy with: {pip_command}"
        warning_incompatible = (
            "PyARPES requires a patched copy of igorpy, "
            "available at \n\t"
            "https://github.com/chstan/igorpy/tarball/712a4c4\n\n\tYou can install with: "
            f"{pip_command}"
        )
        try:
            import igor

            if igor.__version__ <= "0.3":
                msg = "Not using patched version of igorpy."
                raise ValueError(msg)

        except ValueError:
            return warning_incompatible
        except (ImportError, AttributeError):
            return warning

        return None

    checks = [
        ("Igor Pro Support", verify_igor_pro),
        ("qt_tool Support", verify_qt_tool),
    ]

    from colorama import Fore, Style

    print("Checking...")
    for check_name, check_fn in checks:
        initial_str = f"[ ] {check_name}"
        print(initial_str, end="", flush=True)

        failure_message = check_fn()

        print("\b" * len(initial_str) + " " * len(initial_str) + "\b" * len(initial_str), end="")

        if failure_message is None:
            print(f"{Fore.GREEN}[✔] {check_name}{Style.RESET_ALL}")
        else:
            print(
                f"{Fore.RED}[✘] {check_name}: \n\t{failure_message}{Style.RESET_ALL}",
            )
