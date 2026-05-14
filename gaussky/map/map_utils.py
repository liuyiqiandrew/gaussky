"""Validation and normalization helpers for HEALPix map containers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from gaussky.conventions import (
    HEALPIX_ORDERINGS,
    SIGNAL_FIELDS,
    HealpixOrdering,
    SignalField,
)

BeamFwhm: TypeAlias = float | NDArray[np.float64] | None

SUPPORTED_SIGNAL_FIELDS: tuple[SignalField, ...] = SIGNAL_FIELDS
SUPPORTED_HEALPIX_ORDERINGS: tuple[HealpixOrdering, ...] = HEALPIX_ORDERINGS


def readonly_float_array(values: object, name: str) -> NDArray[np.float64]:
    """Return a read-only ``float64`` copy of an array-like value."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
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


def normalize_freqs(freqs_ghz: object) -> NDArray[np.float64]:
    """Validate and normalize a one-dimensional frequency grid."""
    freqs = np.atleast_1d(readonly_float_array(freqs_ghz, "freqs_ghz"))
    if freqs.ndim != 1:
        raise ValueError("freqs_ghz must be one-dimensional")
    if freqs.size == 0:
        raise ValueError("freqs_ghz must contain at least one frequency")
    if np.any(~np.isfinite(freqs)) or np.any(freqs <= 0.0):
        raise ValueError("freqs_ghz must contain finite positive values")
    return freqs


def normalize_signal_fields(fields: object) -> tuple[SignalField, ...]:
    """Validate and normalize a signal-field tuple."""
    if isinstance(fields, str):
        normalized: tuple[object, ...] = (fields,)
    else:
        normalized = tuple(cast(Iterable[object], fields))
    if not normalized:
        raise ValueError("fields must contain at least one field")
    for field in normalized:
        if field not in SUPPORTED_SIGNAL_FIELDS:
            valid = ", ".join(SUPPORTED_SIGNAL_FIELDS)
            raise ValueError(f"Unknown signal field {field!r}; expected one of {valid}")
    return cast(tuple[SignalField, ...], normalized)


def validate_nside(nside: int) -> None:
    """Validate the HEALPix ``nside`` convention."""
    if not isinstance(nside, int):
        raise TypeError("nside must be an integer")
    if nside <= 0:
        raise ValueError("nside must be strictly positive")
    if nside & (nside - 1):
        raise ValueError("nside must be a power of two")


def expected_npix(nside: int) -> int:
    """Return the HEALPix pixel count for ``nside``."""
    return 12 * nside**2


def normalize_ordering(ordering: object) -> HealpixOrdering:
    """Validate and normalize a HEALPix ordering label."""
    if not isinstance(ordering, str):
        raise TypeError("ordering must be a string")
    normalized = ordering.upper()
    if normalized not in SUPPORTED_HEALPIX_ORDERINGS:
        valid = ", ".join(SUPPORTED_HEALPIX_ORDERINGS)
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


def normalize_beam(beam_fwhm_rad: BeamFwhm, nfreq: int) -> BeamFwhm:
    """Validate a scalar or per-frequency beam FWHM value."""
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
