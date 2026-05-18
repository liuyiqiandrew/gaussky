"""Map-package re-exports of the shared validation helpers."""

from __future__ import annotations

from gaussky._validation import (
    BeamFwhm,
    arrays_close,
    beam_values,
    expected_npix,
    normalize_beam,
    normalize_coord,
    normalize_freqs,
    normalize_metadata,
    normalize_ordering,
    normalize_signal_fields,
    readonly_float_array,
    select_beam,
    validate_nside,
)
from gaussky.conventions import (
    HEALPIX_ORDERINGS,
    SIGNAL_FIELDS,
    HealpixOrdering,
    SignalField,
)

SUPPORTED_SIGNAL_FIELDS: tuple[SignalField, ...] = SIGNAL_FIELDS
SUPPORTED_HEALPIX_ORDERINGS: tuple[HealpixOrdering, ...] = HEALPIX_ORDERINGS

__all__ = [
    "BeamFwhm",
    "SUPPORTED_HEALPIX_ORDERINGS",
    "SUPPORTED_SIGNAL_FIELDS",
    "arrays_close",
    "beam_values",
    "expected_npix",
    "normalize_beam",
    "normalize_coord",
    "normalize_freqs",
    "normalize_metadata",
    "normalize_ordering",
    "normalize_signal_fields",
    "readonly_float_array",
    "select_beam",
    "validate_nside",
]
