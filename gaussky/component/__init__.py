"""Gaussian sky component samplers."""

from .base import GaussianComponent, NoiseComponent
from .sed_backed import BaseSEDBackedComponent

__all__ = [
    "BaseSEDBackedComponent",
    "GaussianCMB",
    "GaussianComponent",
    "NoiseComponent",
    "SimpleModifiedBlackbodyDust",
    "SimplePowerLawSynchrotron",
    "WhiteNoise",
]


def __getattr__(name: str):
    """Lazily import concrete components when requested."""
    if name == "GaussianCMB":
        from .cmb import GaussianCMB

        return GaussianCMB
    if name == "SimpleModifiedBlackbodyDust":
        from .dust import SimpleModifiedBlackbodyDust

        return SimpleModifiedBlackbodyDust
    if name == "SimplePowerLawSynchrotron":
        from .synchrotron import SimplePowerLawSynchrotron

        return SimplePowerLawSynchrotron
    if name == "WhiteNoise":
        from .noise import WhiteNoise

        return WhiteNoise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
