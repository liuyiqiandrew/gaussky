"""Unit-conversion helpers for GHz-frequency SED calculations."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import _positive_frequency_array, _validate_positive_scalar

# SI constants used by the Planck and thermodynamic-temperature conversions.
H_PLANCK: Final[float] = 6.62607015e-34  # Planck constant [J s]
K_BOLTZMANN: Final[float] = 1.380649e-23  # Boltzmann constant [J/K]
T_CMB: Final[float] = 2.725  # CMB monopole temperature [K]


def planck_rj_spectrum(
    temperature_k: float, freq_ghz: ArrayLike
) -> NDArray[np.float64]:
    """Planck law in Rayleigh-Jeans brightness temperature units.

    Computes the frequency-dependent part of the Planck spectrum without the
    ``nu**2`` flux prefactor, which is the factor needed for modified blackbody
    SED scaling in RJ temperature units.

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin.
    freq_ghz : array_like
        Frequencies in GHz.

    Returns
    -------
    ndarray
        Planck spectrum evaluated at each frequency, in RJ units.

    Raises
    ------
    ValueError
        If the temperature or any frequency is not strictly positive.
    """
    _validate_positive_scalar(temperature_k, "temperature_k")
    freq = _positive_frequency_array(freq_ghz)
    x = H_PLANCK * freq * 1e9 / (K_BOLTZMANN * temperature_k)
    return freq / np.expm1(x)


def trj_to_tcmb(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """Convert Rayleigh-Jeans temperature to CMB thermodynamic temperature.

    Parameters
    ----------
    freq_ghz : array_like
        Frequencies in GHz.

    Returns
    -------
    ndarray
        Conversion factor: multiply RJ temperature by this to get CMB
        temperature.

    Raises
    ------
    ValueError
        If any frequency is not strictly positive.
    """
    freq = _positive_frequency_array(freq_ghz)
    x = H_PLANCK * freq * 1e9 / (K_BOLTZMANN * T_CMB)
    # expm1 keeps the low-frequency conversion stable where x is small.
    expm1_x = np.expm1(x)
    return expm1_x**2 / (x**2 * np.exp(x))


def tcmb_to_trj(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """Convert CMB thermodynamic temperature to Rayleigh-Jeans temperature.

    Inverse of :func:`trj_to_tcmb`.

    Parameters
    ----------
    freq_ghz : array_like
        Frequencies in GHz.

    Returns
    -------
    ndarray
        Conversion factor: multiply CMB temperature by this to get RJ
        temperature.

    Raises
    ------
    ValueError
        If any frequency is not strictly positive.
    """
    freq = _positive_frequency_array(freq_ghz)
    x = H_PLANCK * freq * 1e9 / (K_BOLTZMANN * T_CMB)
    # Keep this numerically aligned with trj_to_tcmb so the factors remain
    # reciprocal down to low GHz frequencies.
    expm1_x = np.expm1(x)
    return x**2 * np.exp(x) / expm1_x**2
