"""Container for multi-frequency spherical-harmonic coefficient products.

The alm container is a sibling to :class:`gaussky.map.MultiFreqCompMap`. It
stores band-limited Gaussian realizations in harmonic space rather than as
pixelated HEALPix maps. Components produce alms via
:func:`gaussky.component.component_utils.sample_component_alm` and callers
can convert to pixel space at any chosen ``nside`` via ``healpy.alm2map``
(or apply transfer functions, custom smoothing, etc. before doing so).

Conventions:

- ``alms`` has shape ``(nfreq, nharm, nalm)`` with ``nalm =
  healpy.Alm.getsize(lmax, mmax)`` and ``nharm == len(fields)``.
- ``fields`` is a tuple of harmonic fields (``"T"``, ``"E"``, ``"B"``) in
  the caller's requested order.
- ``alms`` are stored as a read-only ``complex128`` array.
- ``beam_fwhm_rad`` is informational only; the alm path does not apply
  beam smoothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

import healpy as hp  # type: ignore[import-not-found]
import numpy as np
from numpy.typing import NDArray

from gaussky._validation import (
    BeamFwhm,
    normalize_beam,
    normalize_coord,
    normalize_freqs,
    normalize_metadata,
    validate_lmax,
)
from gaussky.conventions import HARMONIC_FIELDS, HarmonicField


@dataclass(frozen=True, kw_only=True, eq=False)
class MultiFreqCompAlm:
    """Multi-frequency spherical-harmonic coefficient container.

    Parameters
    ----------
    alms : ndarray
        Complex-valued alm array with shape ``(nfreq, nharm, nalm)``.
        Stored as a read-only ``complex128`` copy.
    lmax : int
        Maximum multipole.
    mmax : int or None, default=None
        Maximum ``m`` for the harmonic transform. ``None`` means
        ``mmax == lmax``.
    freqs_ghz : ndarray
        Frequency channels in GHz.
    fields : tuple of {"T", "E", "B"}
        Harmonic fields stored along axis 1.
    unit : str
        Unit of the corresponding pixel-space map.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Instrument beam FWHM in radians, stored for provenance only.
    coord : str or None, default=None
        Optional coordinate-frame label.
    component_name : str
        Component label, mirroring :class:`MultiFreqCompMap.component_name`.
    metadata : mapping, default={}
        Read-only mapping of model parameters and provenance.
    """

    alms: NDArray[np.complex128]
    lmax: int
    freqs_ghz: NDArray[np.float64]
    fields: tuple[HarmonicField, ...]
    unit: str
    component_name: str
    mmax: int | None = None
    beam_fwhm_rad: BeamFwhm = None
    coord: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the alm shape and metadata contract."""
        validate_lmax(self.lmax)
        resolved_mmax = self.lmax if self.mmax is None else self.mmax
        validate_lmax(resolved_mmax)
        if resolved_mmax > self.lmax:
            raise ValueError("mmax must not exceed lmax")
        object.__setattr__(self, "mmax", resolved_mmax)

        freqs = normalize_freqs(self.freqs_ghz)
        fields = self._normalize_harmonic_fields(self.fields)

        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if self.unit == "":
            raise ValueError("unit must not be empty")

        if not isinstance(self.component_name, str):
            raise TypeError("component_name must be a string")
        if self.component_name == "":
            raise ValueError("component_name must not be empty")

        alms = np.array(self.alms, dtype=np.complex128, copy=True)
        nalm_expected = int(hp.Alm.getsize(self.lmax, resolved_mmax))
        expected_shape = (freqs.size, len(fields), nalm_expected)
        if alms.shape != expected_shape:
            raise ValueError(
                "alms must have shape (nfreq, nharm, nalm); "
                f"expected {expected_shape}, got {alms.shape}"
            )
        if np.any(~np.isfinite(alms.real)) or np.any(~np.isfinite(alms.imag)):
            raise ValueError("alms must contain finite values")
        alms.setflags(write=False)
        object.__setattr__(self, "alms", alms)

        object.__setattr__(self, "freqs_ghz", freqs)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "coord", normalize_coord(self.coord))
        object.__setattr__(
            self, "beam_fwhm_rad", normalize_beam(self.beam_fwhm_rad, freqs.size)
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @staticmethod
    def _normalize_harmonic_fields(
        fields: object,
    ) -> tuple[HarmonicField, ...]:
        """Validate harmonic-field labels and preserve the caller's order."""
        if isinstance(fields, str):
            normalized: tuple[object, ...] = (fields,)
        else:
            normalized = tuple(fields)  # type: ignore[arg-type]
        if not normalized:
            raise ValueError("fields must contain at least one harmonic field")
        for field_label in normalized:
            if field_label not in HARMONIC_FIELDS:
                valid = ", ".join(HARMONIC_FIELDS)
                raise ValueError(
                    f"Unknown harmonic field {field_label!r}; expected one of {valid}"
                )
        return tuple(normalized)  # type: ignore[return-value]

    @property
    def nfreq(self) -> int:
        """Number of frequency channels."""
        return self.freqs_ghz.size

    @property
    def nfields(self) -> int:
        """Number of stored harmonic fields."""
        return len(self.fields)

    @property
    def nalm(self) -> int:
        """Number of alm coefficients per (frequency, field) pair."""
        return self.alms.shape[-1]

    def copy_with(self: Self, **changes: object) -> Self:
        """Return a revalidated copy with selected dataclass fields replaced."""
        from dataclasses import fields as dataclass_fields, replace
        from typing import Any, cast

        valid_fields = {dc_field.name for dc_field in dataclass_fields(self)}
        unknown_fields = set(changes) - valid_fields
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown alm-container field replacement(s): {names}")
        return replace(self, **cast(Any, changes))


__all__ = ["MultiFreqCompAlm"]
