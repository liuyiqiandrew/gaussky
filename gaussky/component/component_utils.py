"""Shared helpers for Gaussian sky component samplers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import healpy as hp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import (
    HEALPIX_ORDERINGS,
    SIGNAL_FIELDS,
    HealpixOrdering,
    SignalField,
    U_K_CMB,
    U_K_CMB_SQUARED,
)
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import AngularPowerSpectrum, validate_healpy_cls
from gaussky.sed import SpectralEnergyDistribution

_FIELD_TO_HEALPY_INDEX: dict[SignalField, int] = {"T": 0, "Q": 1, "U": 2}
_MAX_NUMPY_LEGACY_SEED = 2**32 - 1


def validate_component_name(name: str) -> None:
    """Validate a component name used in map products."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if name == "":
        raise ValueError("name must not be empty")


def normalize_seed(seed: int | None) -> int | None:
    """Validate and normalize a NumPy legacy RNG seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    if seed < 0 or seed > _MAX_NUMPY_LEGACY_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_NUMPY_LEGACY_SEED}")
    return int(seed)


def _normalize_ordering(ordering: HealpixOrdering) -> HealpixOrdering:
    """Validate and canonicalize a HEALPix ordering label."""
    if not isinstance(ordering, str):
        raise TypeError("ordering must be a string")
    normalized = ordering.upper()
    if normalized not in HEALPIX_ORDERINGS:
        valid = ", ".join(HEALPIX_ORDERINGS)
        raise ValueError(
            f"Unknown HEALPix ordering {ordering!r}; expected one of {valid}"
        )
    return cast(HealpixOrdering, normalized)


def _validate_nside(nside: int) -> None:
    """Validate the HEALPix ``nside`` convention before sampling."""
    if not isinstance(nside, int):
        raise TypeError("nside must be an integer")
    if nside <= 0:
        raise ValueError("nside must be strictly positive")
    if nside & (nside - 1):
        raise ValueError("nside must be a power of two")


def _normalize_freqs(
    freqs_ghz: ArrayLike | None,
    *,
    default_ghz: float | None = None,
) -> NDArray[np.float64]:
    """Return a one-dimensional positive frequency array in GHz."""
    if freqs_ghz is None:
        if default_ghz is None:
            raise ValueError("freqs_ghz must be provided")
        freqs_ghz = [default_ghz]

    freqs = np.atleast_1d(np.asarray(freqs_ghz, dtype=np.float64))
    if freqs.ndim != 1:
        raise ValueError("freqs_ghz must be one-dimensional")
    if freqs.size == 0:
        raise ValueError("freqs_ghz must contain at least one frequency")
    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("freqs_ghz must contain finite positive values")
    return freqs


def _normalize_fields(fields: tuple[SignalField, ...]) -> tuple[SignalField, ...]:
    """Validate fields and preserve the caller's requested order."""
    normalized = tuple(fields)
    if not normalized:
        raise ValueError("fields must contain at least one field")
    for signal_field in normalized:
        if signal_field not in SIGNAL_FIELDS:
            valid = ", ".join(SIGNAL_FIELDS)
            raise ValueError(
                f"Unknown signal field {signal_field!r}; expected one of {valid}"
            )
    return normalized


def _signal_field_indices(fields: tuple[SignalField, ...]) -> list[int]:
    """Return Healpy T/Q/U indices for requested signal fields."""
    return [_FIELD_TO_HEALPY_INDEX[signal_field] for signal_field in fields]


def _normalize_beam(beam_fwhm_rad: BeamFwhm, nfreq: int) -> BeamFwhm:
    """Validate a scalar or per-frequency beam FWHM in radians."""
    if beam_fwhm_rad is None:
        return None

    beam = np.asarray(beam_fwhm_rad, dtype=np.float64)
    if beam.ndim == 0:
        value = float(beam)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("beam_fwhm_rad must be finite and non-negative")
        return value

    if beam.ndim != 1:
        raise ValueError("beam_fwhm_rad must be scalar or one-dimensional")
    if beam.shape != (nfreq,):
        raise ValueError("beam_fwhm_rad must be scalar or have shape (nfreq,)")
    if np.any(~np.isfinite(beam)) or np.any(beam < 0.0):
        raise ValueError("beam_fwhm_rad must contain finite non-negative values")
    return beam


def _signal_unit(ps: AngularPowerSpectrum) -> str:
    """Return the map unit implied by the power-spectrum unit."""
    if ps.unit == U_K_CMB_SQUARED:
        return U_K_CMB
    raise ValueError(
        f"Unsupported power-spectrum unit for Gaussian component: {ps.unit!r}"
    )


def _smooth_tqu(pivot_tqu: NDArray[np.float64], beam: float) -> NDArray[np.float64]:
    """Smooth a T/Q/U map with Healpy's polarization-aware convention."""
    return np.asarray(hp.smoothing(pivot_tqu, fwhm=beam, pol=True), dtype=np.float64)


def _reorder_ring_to_nested(maps: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reorder the trailing pixel axis from RING to NESTED."""
    original_shape = maps.shape
    flattened = maps.reshape(-1, original_shape[-1])
    reordered = hp.reorder(flattened, r2n=True)
    return np.asarray(reordered, dtype=np.float64).reshape(original_shape)


def _synfast_tqu(
    healpy_cls: list[NDArray[np.float64]],
    nside: int,
    *,
    seed: int | None,
) -> NDArray[np.float64]:
    """Run Healpy synfast with optional legacy NumPy RNG seeding."""
    if seed is None:
        return np.asarray(
            hp.synfast(healpy_cls, nside, alm=False, pol=True, new=True),
            dtype=np.float64,
        )

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return np.asarray(
            hp.synfast(healpy_cls, nside, alm=False, pol=True, new=True),
            dtype=np.float64,
        )
    finally:
        np.random.set_state(state)


def sample_gaussian_component_map(
    *,
    ps: AngularPowerSpectrum,
    sed: SpectralEnergyDistribution,
    component_name: str,
    metadata: Mapping[str, object],
    nside: int,
    fields: tuple[SignalField, ...],
    freqs_ghz: ArrayLike | None = None,
    beam_fwhm_rad: BeamFwhm = None,
    ordering: HealpixOrdering = "RING",
    coord: str | None = None,
    seed: int | None = None,
) -> MultiFreqCompMap:
    """Sample a Gaussian component and scale it with an SED.

    Parameters
    ----------
    ps : AngularPowerSpectrum
        Polarized angular power spectrum at the SED reference frequency. The
        model must return Healpy-ordered spectra compatible with polarized
        :func:`synfast`.
    sed : SpectralEnergyDistribution
        Frequency scaling normalized at the same reference frequency as ``ps``.
    component_name : str
        Component label stored in the returned map.
    metadata : mapping
        Component model parameters stored in the returned map.
    nside : int
        HEALPix resolution parameter.
    freqs_ghz : array_like or None, default=None
        Frequency channels in GHz. If ``None``, the SED pivot frequency
        ``sed.nu0_ghz`` is sampled, so the returned map has one frequency
        channel with unit SED scaling.
    fields : tuple of {"T", "Q", "U"}
        Signal fields to retain, in output order.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Beam FWHM in radians. A scalar beam is shared across channels; a
        one-dimensional array supplies one beam per frequency.
    ordering : {"RING", "NESTED"}, default="RING"
        HEALPix ordering for the returned map.
    coord : str or None, default=None
        Optional coordinate-frame label for the returned map.
    seed : int or None, default=None
        Optional NumPy legacy RNG seed used for the Healpy realization.

    Returns
    -------
    MultiFreqCompMap
        Component map with shape ``(nfreq, nfield, npix)``.
    """
    _validate_nside(nside)
    normalized_ordering = _normalize_ordering(ordering)
    freqs = _normalize_freqs(freqs_ghz, default_ghz=sed.nu0_ghz)
    normalized_fields = _normalize_fields(fields)
    beam = _normalize_beam(beam_fwhm_rad, freqs.size)
    normalized_seed = normalize_seed(seed)
    unit = _signal_unit(ps)

    lmax = 3 * nside - 1
    healpy_cls = validate_healpy_cls(ps.to_healpy_cls(lmax), lmax)

    npix = hp.nside2npix(nside)
    pivot_tqu = _synfast_tqu(healpy_cls, nside, seed=normalized_seed)
    if pivot_tqu.shape != (len(SIGNAL_FIELDS), npix):
        raise ValueError("healpy.synfast must return a T/Q/U map with shape (3, npix)")

    field_indices = _signal_field_indices(normalized_fields)
    maps = np.empty((freqs.size, len(normalized_fields), npix), dtype=np.float64)

    if beam is None:
        selected = pivot_tqu[field_indices]
        maps[...] = selected[None, :, :]
    elif np.asarray(beam).ndim == 0:
        smoothed_tqu = _smooth_tqu(pivot_tqu, float(beam))
        selected = smoothed_tqu[field_indices]
        maps[...] = selected[None, :, :]
    else:
        beam_array = np.asarray(beam, dtype=np.float64)
        for freq_index, channel_beam in enumerate(beam_array):
            smoothed_tqu = _smooth_tqu(pivot_tqu, float(channel_beam))
            maps[freq_index] = smoothed_tqu[field_indices]

    scale_maps = getattr(sed, "scale_maps", None)
    if callable(scale_maps):
        maps = np.asarray(scale_maps(maps, freqs), dtype=np.float64)
    else:
        maps *= sed.scale(freqs)[:, None, None]

    if normalized_ordering == "NESTED":
        maps = _reorder_ring_to_nested(maps)

    return MultiFreqCompMap(
        maps=maps,
        unit=unit,
        nside=nside,
        ordering=normalized_ordering,
        coord=coord,
        freqs_ghz=freqs,
        fields=normalized_fields,
        beam_fwhm_rad=beam,
        component_name=component_name,
        auxiliary_maps={},
        metadata={**metadata, "seed": normalized_seed},
    )


def sample_frequency_independent_gaussian_component_map(
    *,
    ps: AngularPowerSpectrum,
    component_name: str,
    metadata: Mapping[str, object],
    nside: int,
    freqs_ghz: ArrayLike | None,
    fields: tuple[SignalField, ...],
    beam_fwhm_rad: BeamFwhm = None,
    ordering: HealpixOrdering = "RING",
    coord: str | None = None,
    seed: int | None = None,
) -> MultiFreqCompMap:
    """Sample a Gaussian component with no frequency-dependent SED.

    The same underlying T/Q/U realization is used for every requested
    frequency. Beam smoothing may still differ per frequency channel.

    Parameters
    ----------
    ps : AngularPowerSpectrum
        Polarized angular power spectrum. The model must return Healpy-ordered
        spectra compatible with polarized :func:`synfast`.
    component_name : str
        Component label stored in the returned map.
    metadata : mapping
        Component model parameters stored in the returned map.
    nside : int
        HEALPix resolution parameter.
    freqs_ghz : array_like
        Frequency channels in GHz. They label repeated CMB maps and do not
        change the sampled sky signal.
    fields : tuple of {"T", "Q", "U"}
        Signal fields to retain, in output order.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Beam FWHM in radians. A scalar beam is shared across channels; a
        one-dimensional array supplies one beam per frequency.
    ordering : {"RING", "NESTED"}, default="RING"
        HEALPix ordering for the returned map.
    coord : str or None, default=None
        Optional coordinate-frame label for the returned map.
    seed : int or None, default=None
        Optional NumPy legacy RNG seed used for the Healpy realization.

    Returns
    -------
    MultiFreqCompMap
        Component map with shape ``(nfreq, nfield, npix)``.
    """
    _validate_nside(nside)
    normalized_ordering = _normalize_ordering(ordering)
    freqs = _normalize_freqs(freqs_ghz)
    normalized_fields = _normalize_fields(fields)
    beam = _normalize_beam(beam_fwhm_rad, freqs.size)
    normalized_seed = normalize_seed(seed)
    unit = _signal_unit(ps)

    lmax = 3 * nside - 1
    healpy_cls = validate_healpy_cls(ps.to_healpy_cls(lmax), lmax)

    npix = hp.nside2npix(nside)
    pivot_tqu = _synfast_tqu(healpy_cls, nside, seed=normalized_seed)
    if pivot_tqu.shape != (len(SIGNAL_FIELDS), npix):
        raise ValueError("healpy.synfast must return a T/Q/U map with shape (3, npix)")

    field_indices = _signal_field_indices(normalized_fields)
    maps = np.empty((freqs.size, len(normalized_fields), npix), dtype=np.float64)

    if beam is None:
        selected = pivot_tqu[field_indices]
        maps[...] = selected[None, :, :]
    elif np.asarray(beam).ndim == 0:
        smoothed_tqu = _smooth_tqu(pivot_tqu, float(beam))
        selected = smoothed_tqu[field_indices]
        maps[...] = selected[None, :, :]
    else:
        beam_array = np.asarray(beam, dtype=np.float64)
        for freq_index, channel_beam in enumerate(beam_array):
            smoothed_tqu = _smooth_tqu(pivot_tqu, float(channel_beam))
            maps[freq_index] = smoothed_tqu[field_indices]

    if normalized_ordering == "NESTED":
        maps = _reorder_ring_to_nested(maps)

    return MultiFreqCompMap(
        maps=maps,
        unit=unit,
        nside=nside,
        ordering=normalized_ordering,
        coord=coord,
        freqs_ghz=freqs,
        fields=normalized_fields,
        beam_fwhm_rad=beam,
        component_name=component_name,
        auxiliary_maps={},
        metadata={**metadata, "seed": normalized_seed},
    )
