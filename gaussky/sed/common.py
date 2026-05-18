"""Common analytic SED models used by foreground components."""

from __future__ import annotations

from collections.abc import Callable
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
        return unit_conversion * self._rj_scaling(freq)

    def scale_maps(self, maps: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale pivot-frequency maps using staged RJ/CMB conversions.

        This applies the same operation order used by the legacy ``pygsm``
        implementation: convert pivot maps from thermodynamic CMB to RJ units,
        apply the RJ modified-blackbody scaling, then convert each frequency
        channel back to thermodynamic CMB units.
        """
        return _scale_maps_like_pygsm(
            maps,
            freq_ghz,
            nu0_ghz=self.nu0_ghz,
            rj_scaling=self._rj_scaling,
        )

    def scale_cls(self, cls: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale pivot-frequency spectra using staged RJ/CMB conversions."""
        return _scale_cls_like_pygsm(
            cls,
            freq_ghz,
            nu0_ghz=self.nu0_ghz,
            rj_scaling=self._rj_scaling,
        )

    def _rj_scaling(self, freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the RJ-unit modified-blackbody scaling."""
        return (freq_ghz / self.nu0_ghz) ** self.beta * (
            planck_rj_spectrum(self.temperature_k, freq_ghz)
            / planck_rj_spectrum(self.temperature_k, self.nu0_ghz)
        )


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

    def scale_maps(self, maps: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale pivot-frequency maps using staged RJ/CMB conversions."""
        return _scale_maps_like_pygsm(
            maps,
            freq_ghz,
            nu0_ghz=self.nu0_ghz,
            rj_scaling=self._rj_scaling,
        )

    def scale_cls(self, cls: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale pivot-frequency spectra using staged RJ/CMB conversions."""
        return _scale_cls_like_pygsm(
            cls,
            freq_ghz,
            nu0_ghz=self.nu0_ghz,
            rj_scaling=self._rj_scaling,
        )

    def _rj_scaling(self, freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the RJ-unit power-law scaling."""
        return (freq_ghz / self.nu0_ghz) ** self.beta


def _positive_frequency_vector(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """Return frequencies as a one-dimensional float array."""
    freq = np.atleast_1d(_positive_frequency_array(freq_ghz))
    if freq.ndim != 1:
        raise ValueError("freq_ghz must be scalar or one-dimensional")
    return freq


def _scale_maps_like_pygsm(
    maps: ArrayLike,
    freq_ghz: ArrayLike,
    *,
    nu0_ghz: float,
    rj_scaling: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Scale maps with the same operation order as ``pygsm.Sky``."""
    freq = _positive_frequency_vector(freq_ghz)
    scaled_maps = np.array(maps, dtype=np.float64, copy=True)
    if scaled_maps.ndim == 0 or scaled_maps.shape[0] != freq.size:
        raise ValueError("maps must have a leading frequency axis matching freq_ghz")

    factor_shape = (freq.size,) + (1,) * (scaled_maps.ndim - 1)
    scaled_maps *= tcmb_to_trj(nu0_ghz)
    scaled_maps *= rj_scaling(freq).reshape(factor_shape)
    return scaled_maps * trj_to_tcmb(freq).reshape(factor_shape)


def _scale_cls_like_pygsm(
    cls: ArrayLike,
    freq_ghz: ArrayLike,
    *,
    nu0_ghz: float,
    rj_scaling: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Scale spectra with the same operation order as ``pygsm.Sky``."""
    freq = _positive_frequency_vector(freq_ghz)
    rj_cls0 = np.asarray(cls, dtype=np.float64) * tcmb_to_trj(nu0_ghz) ** 2
    rj_cls = np.tile(rj_cls0, (freq.size,) + (1,) * rj_cls0.ndim)

    factor_shape = (freq.size,) + (1,) * rj_cls0.ndim
    rj_cls *= rj_scaling(freq).reshape(factor_shape) ** 2
    return rj_cls * (trj_to_tcmb(freq) ** 2).reshape(factor_shape)
