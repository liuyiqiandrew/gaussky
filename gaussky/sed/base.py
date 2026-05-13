"""Shared protocol for frequency-dependent SED scaling models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SpectralEnergyDistribution(Protocol):
    """Frequency scaling for a sky component.

    Implementations return dimensionless multiplicative factors normalized at
    ``nu0_ghz``. Frequencies are always specified in GHz.
    """

    nu0_ghz: float

    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the SED scaling at one or more frequencies.

        Parameters
        ----------
        freq_ghz : array_like
            Frequency or frequencies in GHz.

        Returns
        -------
        ndarray
            Dimensionless scaling factors with the same broadcast shape as
            ``freq_ghz``.
        """
        ...


def _validate_finite_scalar(value: float, name: str) -> float:
    """Return a scalar after finite-value validation."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_positive_scalar(value: float, name: str) -> float:
    """Return a scalar after strict positivity validation."""
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _positive_frequency_array(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """Return frequencies as a float array after positivity validation."""
    array = np.asarray(freq_ghz, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("freq_ghz must contain finite positive values")
    return array
