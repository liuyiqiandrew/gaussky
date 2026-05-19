"""Shared protocols for Gaussian sky component samplers."""

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
        fields: tuple[SignalField, ...],
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
        seed: int | None = None,
        lmax: int | None = None,
    ) -> MultiFreqCompMap:
        """Sample one realization of this component."""
        ...


@runtime_checkable
class NoiseComponent(Protocol):
    """Protocol for instrument-noise components.

    Noise components draw independent per-channel realizations and are
    never beam-smoothed by the sampler — the total map convention is
    ``total = (beam ⊗ sky) + noise``. The method name
    :meth:`sample_noise_map` is intentionally distinct from
    :meth:`GaussianComponent.sample_map` so a noise component can never be
    routed through the signal-side helper (which always applies the beam).

    ``beam_fwhm_rad`` is accepted for provenance — implementations record
    it on the returned :class:`~gaussky.map.MultiFreqCompMap` but must not
    smooth the noise realization with it.
    """

    name: str

    def sample_noise_map(
        self,
        *,
        nside: int,
        fields: tuple[SignalField, ...],
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
        seed: int | None = None,
    ) -> MultiFreqCompMap:
        """Sample one realization of this noise component."""
        ...
