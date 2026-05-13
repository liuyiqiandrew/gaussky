"""Common analytic SED models used by foreground components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import (
    SpectralEnergyDistribution,
    _positive_frequency_array,
    _validate_finite_scalar,
    _validate_positive_scalar,
)
from .sed_utils import planck_rj_spectrum, tcmb_to_trj, trj_to_tcmb


@dataclass(frozen=True)
class ModifiedBlackbodySED(SpectralEnergyDistribution):
    """Modified blackbody SED normalized in thermodynamic CMB units.

    Parameters
    ----------
    beta : float
        Dust spectral index in Rayleigh-Jeans temperature units.
    temperature_k : float
        Dust temperature in Kelvin.
    nu0_ghz : float
        Reference frequency in GHz where the scaling is normalized to one.

    Raises
    ------
    ValueError
        If ``beta`` is non-finite, ``temperature_k`` is not strictly positive,
        or ``nu0_ghz`` is not strictly positive.
    """

    beta: float
    temperature_k: float
    nu0_ghz: float

    def __post_init__(self) -> None:
        """Validate scalar SED parameters."""
        _validate_finite_scalar(self.beta, "beta")
        _validate_positive_scalar(self.temperature_k, "temperature_k")
        _validate_positive_scalar(self.nu0_ghz, "nu0_ghz")

    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the dimensionless SED scaling.

        Parameters
        ----------
        freq_ghz : array_like
            Frequency or frequencies in GHz.

        Returns
        -------
        ndarray
            Scaling from the reference frequency to ``freq_ghz`` in
            thermodynamic CMB temperature units.

        Raises
        ------
        ValueError
            If any requested frequency is not strictly positive.
        """
        freq = _positive_frequency_array(freq_ghz)
        # The spectral index and greybody factor are naturally expressed in
        # Rayleigh-Jeans temperature units; wrap them with the inverse unit
        # conversions so callers receive thermodynamic CMB scaling factors.
        unit_conversion = trj_to_tcmb(freq) * tcmb_to_trj(self.nu0_ghz)
        rj_scaling = (freq / self.nu0_ghz) ** self.beta
        rj_scaling *= planck_rj_spectrum(self.temperature_k, freq)
        rj_scaling /= planck_rj_spectrum(self.temperature_k, self.nu0_ghz)
        return unit_conversion * rj_scaling


@dataclass(frozen=True)
class PowerLawSED(SpectralEnergyDistribution):
    """Power-law SED normalized in thermodynamic CMB units.

    Parameters
    ----------
    beta : float
        Spectral index in Rayleigh-Jeans temperature units.
    nu0_ghz : float
        Reference frequency in GHz where the scaling is normalized to one.

    Raises
    ------
    ValueError
        If ``beta`` is non-finite or ``nu0_ghz`` is not strictly positive.
    """

    beta: float
    nu0_ghz: float

    def __post_init__(self) -> None:
        """Validate scalar SED parameters."""
        _validate_finite_scalar(self.beta, "beta")
        _validate_positive_scalar(self.nu0_ghz, "nu0_ghz")

    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the dimensionless SED scaling.

        Parameters
        ----------
        freq_ghz : array_like
            Frequency or frequencies in GHz.

        Returns
        -------
        ndarray
            Scaling from the reference frequency to ``freq_ghz`` in
            thermodynamic CMB temperature units.

        Raises
        ------
        ValueError
            If any requested frequency is not strictly positive.
        """
        freq = _positive_frequency_array(freq_ghz)
        # Apply the same thermodynamic/RJ convention as the modified
        # blackbody model so all SEDs share an output unit convention.
        unit_conversion = trj_to_tcmb(freq) * tcmb_to_trj(self.nu0_ghz)
        rj_scaling = (freq / self.nu0_ghz) ** self.beta
        return unit_conversion * rj_scaling
