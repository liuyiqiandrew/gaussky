"""Angular power-spectrum interfaces and simple analytic models."""

from gaussky.conventions import HEALPY_POLARIZED_ORDER, SpectrumPair

from .base import (
    AngularPowerSpectrum,
    validate_healpy_cls,
)
from .cmb import CMBCl
from .powerlaw import PowerLawCl

__all__ = [
    "AngularPowerSpectrum",
    "CMBCl",
    "HEALPY_POLARIZED_ORDER",
    "PowerLawCl",
    "SpectrumPair",
    "validate_healpy_cls",
]
