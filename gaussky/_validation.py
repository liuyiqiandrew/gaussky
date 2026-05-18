"""Shared validation and normalization helpers.

This module is the single source of truth for the small scalar / array /
HEALPix-metadata validators reused across :mod:`gaussky.sed`,
:mod:`gaussky.component`, and :mod:`gaussky.map`. Subpackages must import
from here rather than re-implementing the checks to prevent the per-package
copies from drifting.

The module is private; importers should reach in directly rather than going
through public re-exports for these helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import (
    HEALPIX_ORDERINGS,
    SIGNAL_FIELDS,
    HealpixOrdering,
    SignalField,
)

BeamFwhm: TypeAlias = float | NDArray[np.float64] | None


def validate_finite_scalar(value: float, name: str) -> float:
    """Return a scalar after finite-value validation."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_positive_scalar(value: float, name: str) -> float:
    """Return a scalar after strict positivity validation."""
    if value <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def validate_nside(nside: int) -> None:
    """Validate the HEALPix ``nside`` convention."""
    if not isinstance(nside, int):
        raise TypeError("nside must be an integer")
    if nside <= 0:
        raise ValueError("nside must be strictly positive")
    if nside & (nside - 1):
        raise ValueError("nside must be a power of two")


def validate_lmax(lmax: int) -> None:
    """Validate a band-limit (must be a non-negative integer)."""
    if isinstance(lmax, bool) or not isinstance(lmax, int):
        raise TypeError("lmax must be an integer")
    if lmax < 0:
        raise ValueError("lmax must be non-negative")


def expected_npix(nside: int) -> int:
    """Return the HEALPix pixel count for ``nside``."""
    return 12 * nside**2


def readonly_float_array(values: object, name: str) -> NDArray[np.float64]:
    """Return a read-only ``float64`` copy of an array-like value."""
    del name  # accepted for signature compatibility; not used in the message
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def positive_frequency_array(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """Return frequencies as a float array after positivity validation."""
    array = np.asarray(freq_ghz, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("freq_ghz must contain finite positive values")
    return array


def normalize_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    """Copy metadata into a read-only mapping."""
    normalized: dict[str, object] = {}
    for name, value in metadata.items():
        if not isinstance(name, str):
            raise TypeError("metadata keys must be strings")
        if name == "":
            raise ValueError("metadata keys must not be empty")
        normalized[name] = value
    return MappingProxyType(normalized)


def normalize_ordering(ordering: object) -> HealpixOrdering:
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


def normalize_coord(coord: object) -> str | None:
    """Validate an optional coordinate-frame label."""
    if coord is None:
        return None
    if not isinstance(coord, str):
        raise TypeError("coord must be a string or None")
    if coord == "":
        raise ValueError("coord must not be empty")
    return coord


def normalize_signal_fields(fields: object) -> tuple[SignalField, ...]:
    """Validate and normalize a signal-field tuple.

    Accepts a single field string or an iterable of field strings; in the
    latter case the requested order is preserved.
    """
    if isinstance(fields, str):
        normalized: tuple[object, ...] = (fields,)
    else:
        normalized = tuple(cast(Iterable[object], fields))
    if not normalized:
        raise ValueError("fields must contain at least one field")
    for field in normalized:
        if field not in SIGNAL_FIELDS:
            valid = ", ".join(SIGNAL_FIELDS)
            raise ValueError(f"Unknown signal field {field!r}; expected one of {valid}")
    return cast(tuple[SignalField, ...], normalized)


def normalize_freqs(
    freqs_ghz: ArrayLike | None,
    *,
    default_ghz: float | None = None,
) -> NDArray[np.float64]:
    """Return a one-dimensional positive frequency array in GHz.

    Parameters
    ----------
    freqs_ghz : array_like or None
        Requested frequency channels. ``None`` falls back to a one-element
        array containing ``default_ghz`` if that argument was provided.
    default_ghz : float or None, default=None
        Pivot frequency used when ``freqs_ghz`` is ``None``. ``None`` means
        no fallback is available and ``freqs_ghz`` must be provided.

    Returns
    -------
    ndarray
        Read-only ``float64`` one-dimensional array of positive frequencies.
    """
    if freqs_ghz is None:
        if default_ghz is None:
            raise ValueError("freqs_ghz must be provided")
        freqs_ghz = [default_ghz]

    freqs = np.atleast_1d(readonly_float_array(freqs_ghz, "freqs_ghz"))
    if freqs.ndim != 1:
        raise ValueError("freqs_ghz must be one-dimensional")
    if freqs.size == 0:
        raise ValueError("freqs_ghz must contain at least one frequency")
    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("freqs_ghz must contain finite positive values")
    return freqs


def normalize_beam(beam_fwhm_rad: BeamFwhm, nfreq: int) -> BeamFwhm:
    """Validate a scalar or per-frequency beam FWHM in radians."""
    if beam_fwhm_rad is None:
        return None

    beam = np.asarray(beam_fwhm_rad, dtype=np.float64)
    if beam.ndim == 0:
        value = float(beam)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("beam_fwhm_rad must be finite and non-negative")
        return value

    normalized = readonly_float_array(beam_fwhm_rad, "beam_fwhm_rad")
    if normalized.ndim != 1:
        raise ValueError("beam_fwhm_rad must be scalar or one-dimensional")
    if normalized.shape != (nfreq,):
        raise ValueError("beam_fwhm_rad must be scalar or have shape (nfreq,)")
    if np.any(~np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("beam_fwhm_rad must contain finite non-negative values")
    return normalized


def beam_values(beam_fwhm_rad: BeamFwhm, nfreq: int) -> NDArray[np.float64] | None:
    """Represent a beam value as a per-frequency array for comparisons."""
    if beam_fwhm_rad is None:
        return None
    beam = np.asarray(beam_fwhm_rad, dtype=np.float64)
    if beam.ndim == 0:
        return np.full(nfreq, float(beam), dtype=np.float64)
    return beam


def select_beam(beam_fwhm_rad: BeamFwhm, index: int) -> BeamFwhm:
    """Return the beam value that corresponds to a selected frequency."""
    if beam_fwhm_rad is None or np.asarray(beam_fwhm_rad).ndim == 0:
        return beam_fwhm_rad
    return float(np.asarray(beam_fwhm_rad, dtype=np.float64)[index])


def arrays_close(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Compare arrays while treating shape mismatches as incompatibility."""
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=rtol, atol=atol)
    )
