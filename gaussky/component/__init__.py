"""Gaussian sky component samplers."""

from .base import GaussianComponent
from .synchrotron import SimplePowerLawSynchrotron

__all__ = [
    "GaussianComponent",
    "SimplePowerLawSynchrotron",
]
