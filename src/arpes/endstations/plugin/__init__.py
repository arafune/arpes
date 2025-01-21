"""Plugins for data loading."""

from __future__ import annotations

from .ALG_main import ALGMainChamber, electrons_per_pulse_mira
from .ALG_spin_ToF import SpinToFEndstation
from .ANTARES import ANTARESEndstation
from .BL10_SARPES import BL10012SARPESEndstation
from .DSNP_UMCS import DSNP_UMCSEndstation
from .Elettra_spectromicroscopy import SpectromicroscopyElettraEndstation
from .example_data import ExampleDataEndstation
from .fallback import FallbackEndstation
from .HERS import HERSEndstation
from .igor_export import IgorExportEndstation
from .igor_plugin import IgorEndstation
from .kaindl import KaindlEndstation
from .MAESTRO import MAESTRONanoARPESEndstation
from .MBS import MBSEndstation
from .merlin import BL403ARPESEndstation
from .SPD_main import SPDEndstation
from .SSRF_NSRL import NSRLEndstation, SSRFEndstation
from .SToF_DLD import SToFDLDEndstation
