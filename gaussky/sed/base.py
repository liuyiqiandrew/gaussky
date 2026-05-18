"""Protocol for frequency-dependent SED scaling models."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky._validation import (
    positive_frequency_array as _positive_frequency_array,
    validate_finite_scalar as _validate_finite_scalar,
    validate_positive_scalar as _validate_positive_scalar,
)

__all__ = [
    "SpectralEnergyDistribution",
    "_positive_frequency_array",
    "_validate_finite_scalar",
    "_validate_positive_scalar",
]


class SpectralEnergyDistribution(Protocol):
    """Frequency scaling for a sky component.

    Implementations expose three related entry points:

    - :meth:`scale` returns the dimensionless per-frequency scaling factor
      (broadcastable along the frequency axis). The factor equals one at
      ``nu0_ghz`` by construction.
    - :meth:`scale_maps` applies the staged thermodynamic-CMB → RJ → SED →
      thermodynamic-CMB conversion to an array whose **leading axis is
      frequency**, matching the operation order of the legacy ``pygsm``
      implementation. Use this when scaling sampled signal maps.
    - :meth:`scale_cls` applies the equivalent staged conversion to angular
      power spectra (factors enter squared). Use this when scaling spectra.

    Frequencies are always in GHz.
    """

    nu0_ghz: float

    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Return the dimensionless per-frequency scaling factor."""

    def scale_maps(self, maps: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale maps from ``nu0_ghz`` to each ``freq_ghz`` channel.

        ``maps`` must have its leading axis equal to ``len(freq_ghz)``. The
        return value has the same shape as ``maps`` (with a leading axis
        of size ``freq_ghz.size`` when ``freq_ghz`` is a scalar promoted to
        a 1-element vector).
        """

    def scale_cls(self, cls: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        """Scale angular power spectra from ``nu0_ghz`` to each ``freq_ghz``."""
