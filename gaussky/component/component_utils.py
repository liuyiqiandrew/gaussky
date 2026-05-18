"""Shared helpers for Gaussian sky component samplers."""

from __future__ import annotations

from collections.abc import Mapping

import healpy as hp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky._validation import (
    normalize_beam,
    normalize_freqs,
    normalize_ordering,
    normalize_signal_fields,
    validate_lmax,
    validate_nside,
)
from gaussky.conventions import (
    HARMONIC_FIELDS,
    SIGNAL_FIELDS,
    HarmonicField,
    HealpixOrdering,
    SignalField,
)
from gaussky.map import BeamFwhm, MultiFreqCompAlm, MultiFreqCompMap
from gaussky.ps import AngularPowerSpectrum, validate_healpy_cls
from gaussky.sed import SpectralEnergyDistribution
from gaussky.units import signal_unit_for

_FIELD_TO_HEALPY_INDEX: dict[SignalField, int] = {"T": 0, "Q": 1, "U": 2}
_HARMONIC_FIELD_TO_INDEX: dict[HarmonicField, int] = {"T": 0, "E": 1, "B": 2}
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


def _signal_field_indices(fields: tuple[SignalField, ...]) -> list[int]:
    """Return Healpy T/Q/U indices for requested signal fields."""
    return [_FIELD_TO_HEALPY_INDEX[signal_field] for signal_field in fields]


def _signal_unit(ps: AngularPowerSpectrum) -> str:
    """Return the map unit implied by the power-spectrum unit."""
    return signal_unit_for(ps.unit)


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


def _apply_beam_smoothing(
    pivot_tqu: NDArray[np.float64],
    *,
    beam: BeamFwhm,
    field_indices: list[int],
    nfreq: int,
    nfields: int,
    npix: int,
) -> NDArray[np.float64]:
    """Apply the pivot T/Q/U realization to ``(nfreq, nfields, npix)`` maps.

    Scalar or absent beams are broadcast across all frequencies; a
    per-channel beam array smooths each channel independently.
    """
    maps = np.empty((nfreq, nfields, npix), dtype=np.float64)
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
    return maps


def sample_component_map(
    *,
    ps: AngularPowerSpectrum,
    sed: SpectralEnergyDistribution | None,
    component_name: str,
    metadata: Mapping[str, object],
    nside: int,
    fields: tuple[SignalField, ...],
    freqs_ghz: ArrayLike | None = None,
    beam_fwhm_rad: BeamFwhm = None,
    ordering: HealpixOrdering = "RING",
    coord: str | None = None,
    seed: int | None = None,
    lmax: int | None = None,
) -> MultiFreqCompMap:
    """Sample a Gaussian component, optionally scaling it with an SED.

    Parameters
    ----------
    ps : AngularPowerSpectrum
        Polarized angular power spectrum. When ``sed`` is given this spectrum
        is evaluated at ``sed.nu0_ghz``; when ``sed`` is ``None`` the same
        realization is broadcast across every requested frequency channel.
    sed : SpectralEnergyDistribution or None
        Frequency scaling normalized at the same reference frequency as ``ps``.
        Pass ``None`` for a frequency-independent component (e.g. CMB).
    component_name : str
        Component label stored in the returned map.
    metadata : mapping
        Component model parameters stored in the returned map.
    nside : int
        HEALPix resolution parameter.
    fields : tuple of {"T", "Q", "U"}
        Signal fields to retain, in output order.
    freqs_ghz : array_like or None, default=None
        Frequency channels in GHz. If ``sed`` is given and this is ``None``,
        the SED pivot frequency ``sed.nu0_ghz`` is used as a one-channel
        fallback. If ``sed`` is ``None``, ``freqs_ghz`` must be supplied.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Beam FWHM in radians. A scalar beam is shared across channels; a
        one-dimensional array supplies one beam per frequency.
    ordering : {"RING", "NESTED"}, default="RING"
        HEALPix ordering for the returned map.
    coord : str or None, default=None
        Optional coordinate-frame label for the returned map.
    seed : int or None, default=None
        Optional NumPy legacy RNG seed used for the Healpy realization.
    lmax : int or None, default=None
        Band-limit for the spectrum and Healpy realization. ``None`` uses
        the standard ``3 * nside - 1`` ceiling. Lower values band-limit the
        realization; higher values let Healpy sample beyond the pixel
        resolution.

    Returns
    -------
    MultiFreqCompMap
        Component map with shape ``(nfreq, nfield, npix)``.
    """
    validate_nside(nside)
    normalized_ordering = normalize_ordering(ordering)
    default_ghz = sed.nu0_ghz if sed is not None else None
    freqs = normalize_freqs(freqs_ghz, default_ghz=default_ghz)
    normalized_fields = normalize_signal_fields(fields)
    beam = normalize_beam(beam_fwhm_rad, freqs.size)
    normalized_seed = normalize_seed(seed)
    unit = _signal_unit(ps)

    resolved_lmax = 3 * nside - 1 if lmax is None else lmax
    validate_lmax(resolved_lmax)
    healpy_cls = validate_healpy_cls(ps.to_healpy_cls(resolved_lmax), resolved_lmax)

    npix = hp.nside2npix(nside)
    pivot_tqu = _synfast_tqu(healpy_cls, nside, seed=normalized_seed)
    if pivot_tqu.shape != (len(SIGNAL_FIELDS), npix):
        raise ValueError("healpy.synfast must return a T/Q/U map with shape (3, npix)")

    field_indices = _signal_field_indices(normalized_fields)
    maps = _apply_beam_smoothing(
        pivot_tqu,
        beam=beam,
        field_indices=field_indices,
        nfreq=freqs.size,
        nfields=len(normalized_fields),
        npix=npix,
    )

    if sed is not None:
        maps = np.asarray(sed.scale_maps(maps, freqs), dtype=np.float64)

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


def _synalm_teb(
    healpy_cls: list[NDArray[np.float64]],
    *,
    lmax: int,
    seed: int | None,
) -> tuple[NDArray[np.complex128], ...]:
    """Run Healpy synalm with optional legacy NumPy RNG seeding."""
    if seed is None:
        result = hp.synalm(healpy_cls, lmax=lmax, new=True)
    else:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            result = hp.synalm(healpy_cls, lmax=lmax, new=True)
        finally:
            np.random.set_state(state)
    return tuple(np.asarray(alm, dtype=np.complex128) for alm in result)


def _harmonic_field_indices(fields: tuple[HarmonicField, ...]) -> list[int]:
    """Return Healpy T/E/B indices for requested harmonic fields."""
    return [_HARMONIC_FIELD_TO_INDEX[harmonic_field] for harmonic_field in fields]


def _normalize_harmonic_fields(
    fields: tuple[HarmonicField, ...],
) -> tuple[HarmonicField, ...]:
    """Validate harmonic-field labels and preserve the caller's requested order."""
    normalized = tuple(fields)
    if not normalized:
        raise ValueError("fields must contain at least one harmonic field")
    for harmonic_field in normalized:
        if harmonic_field not in HARMONIC_FIELDS:
            valid = ", ".join(HARMONIC_FIELDS)
            raise ValueError(
                f"Unknown harmonic field {harmonic_field!r}; expected one of {valid}"
            )
    return normalized


def sample_component_alm(
    *,
    ps: AngularPowerSpectrum,
    sed: SpectralEnergyDistribution | None,
    component_name: str,
    metadata: Mapping[str, object],
    lmax: int,
    fields: tuple[HarmonicField, ...] = ("T", "E", "B"),
    freqs_ghz: ArrayLike | None = None,
    beam_fwhm_rad: BeamFwhm = None,
    coord: str | None = None,
    seed: int | None = None,
) -> MultiFreqCompAlm:
    """Sample a Gaussian component as spherical-harmonic coefficients.

    Returns a :class:`~gaussky.map.MultiFreqCompAlm` carrying T/E/B alms in
    Healpy's polarized convention. SED-backed components multiply the
    pivot-frequency alms by the per-frequency :meth:`scale` factor; CMB-like
    components broadcast the same realization across every channel.

    Parameters
    ----------
    ps : AngularPowerSpectrum
        Polarized angular power spectrum (Healpy-ordered).
    sed : SpectralEnergyDistribution or None
        Frequency scaling normalized at the same reference as ``ps``. Pass
        ``None`` for a frequency-independent component (e.g. CMB).
    component_name : str
        Component label stored in the returned container.
    metadata : mapping
        Component model parameters stored in the returned container.
    lmax : int
        Band-limit for the realization. Must be non-negative.
    fields : tuple of {"T", "E", "B"}, default=("T","E","B")
        Harmonic fields to retain, in output order.
    freqs_ghz : array_like or None, default=None
        Frequency channels in GHz. When ``sed`` is given, ``None`` falls
        back to ``sed.nu0_ghz``. When ``sed`` is ``None``, ``freqs_ghz``
        must be supplied.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Instrument beam FWHM in radians, stored for provenance only. The
        alm path does **not** apply beam smoothing; combine with
        ``healpy.gauss_beam`` (or your own ``bl``) downstream if needed.
    coord : str or None, default=None
        Optional coordinate-frame label.
    seed : int or None, default=None
        Optional NumPy legacy RNG seed used for the Healpy realization.

    Returns
    -------
    MultiFreqCompAlm
        Container with shape ``(nfreq, nfield, nalm)`` where
        ``nalm = healpy.Alm.getsize(lmax)``.
    """
    validate_lmax(lmax)
    default_ghz = sed.nu0_ghz if sed is not None else None
    freqs = normalize_freqs(freqs_ghz, default_ghz=default_ghz)
    normalized_fields = _normalize_harmonic_fields(fields)
    beam = normalize_beam(beam_fwhm_rad, freqs.size)
    normalized_seed = normalize_seed(seed)
    unit = _signal_unit(ps)

    healpy_cls = validate_healpy_cls(ps.to_healpy_cls(lmax), lmax)
    pivot_teb = _synalm_teb(healpy_cls, lmax=lmax, seed=normalized_seed)
    if len(pivot_teb) != len(HARMONIC_FIELDS):
        raise ValueError("healpy.synalm must return one alm array per harmonic field")
    nalm = pivot_teb[0].size

    field_indices = _harmonic_field_indices(normalized_fields)
    pivot_stack = np.stack([pivot_teb[i] for i in field_indices], axis=0)
    alms = np.broadcast_to(
        pivot_stack[None, :, :], (freqs.size, len(normalized_fields), nalm)
    ).copy()

    if sed is not None:
        alms *= sed.scale(freqs)[:, None, None]

    return MultiFreqCompAlm(
        alms=alms,
        lmax=lmax,
        freqs_ghz=freqs,
        fields=normalized_fields,
        unit=unit,
        component_name=component_name,
        beam_fwhm_rad=beam,
        coord=coord,
        metadata={**metadata, "seed": normalized_seed},
    )
