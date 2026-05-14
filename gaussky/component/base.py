"""Shared protocol for Gaussian sky component samplers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap


@runtime_checkable
class GaussianComponent(Protocol):
    """Protocol for components that sample multi-frequency sky maps."""

    name: str

    def sample_map(
        self,
        *,
        nside: int,
        freqs_ghz: ArrayLike,
        fields: tuple[SignalField, ...],
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
    ) -> MultiFreqCompMap:
        """Sample one realization of this component."""
        ...
