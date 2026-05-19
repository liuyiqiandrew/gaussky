"""Shared HEALPix map container protocols and base dataclasses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from typing import Any, Protocol, Self, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from gaussky.conventions import HealpixOrdering, SignalField

from .map_utils import (
    SUPPORTED_HEALPIX_ORDERINGS,
    SUPPORTED_SIGNAL_FIELDS,
    BeamFwhm,
    arrays_close as _arrays_close,
    beam_values as _beam_values,
    expected_npix as _expected_npix,
    normalize_beam as _normalize_beam,
    normalize_coord as _normalize_coord,
    normalize_freqs as _normalize_freqs,
    normalize_metadata as _normalize_metadata,
    normalize_ordering as _normalize_ordering,
    normalize_signal_fields as _normalize_signal_fields,
    readonly_float_array as _readonly_float_array,
    select_beam as _select_beam,
    validate_nside as _validate_nside,
)


@runtime_checkable
class HealpixMapContainer(Protocol):
    """Protocol for generic HEALPix map products.

    Generic HEALPix maps only guarantee pixelization metadata and a trailing
    pixel axis. They may represent signal maps, spectral-index maps, masks, or
    other auxiliary products.
    """

    maps: NDArray[np.float64]
    nside: int
    ordering: HealpixOrdering
    coord: str | None

    @property
    def npix(self) -> int:
        """Number of HEALPix pixels in each map."""

    @property
    def npix_expected(self) -> int:
        """Expected pixel count for ``nside``."""

    def assert_same_pixelization(self, other: HealpixMapContainer) -> None:
        """Validate matching HEALPix pixelization metadata."""

    def copy_with(self, **changes: object) -> Self:
        """Return a copy with selected dataclass fields replaced."""


@runtime_checkable
class SignalMapContainer(HealpixMapContainer, Protocol):
    """Protocol for signal-style multi-frequency HEALPix maps.

    Signal maps store data with shape ``(nfreq, nfield, npix)``. Frequencies are
    in GHz, beam full-width at half-maximum values are in radians, and fields
    are constrained to ``T``, ``Q``, and ``U``.
    """

    freqs_ghz: NDArray[np.float64]
    fields: tuple[SignalField, ...]
    unit: str
    beam_fwhm_rad: BeamFwhm

    @property
    def nfreq(self) -> int:
        """Number of frequency channels."""

    @property
    def nfields(self) -> int:
        """Number of signal fields."""

    def field_index(self, field: SignalField) -> int:
        """Return the axis index for a signal field."""

    def freq_index(self, freq_ghz: float, *, atol: float = 0.0) -> int:
        """Return the axis index for a frequency channel."""

    def select_field(self, field: SignalField) -> Self:
        """Return a one-field signal map container."""

    def select_freq(self, freq_ghz: float, *, atol: float = 0.0) -> Self:
        """Return a one-frequency signal map container."""

    def assert_compatible(
        self,
        other: SignalMapContainer,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        """Validate that another signal map has matching grid and metadata."""


@dataclass(frozen=True, kw_only=True, eq=False)
class BaseHealpixMap(HealpixMapContainer):
    """Validated generic HEALPix map container.

    Parameters
    ----------
    maps : ndarray
        Map array with trailing pixel axis ``npix``. The stored array is a
        read-only ``float64`` copy.
    nside : int
        HEALPix resolution parameter. Must be a positive power of two.
    ordering : {"RING", "NESTED"}, default="RING"
        HEALPix pixel ordering.
    coord : str or None, default=None
        Optional coordinate-frame label.

    Raises
    ------
    ValueError
        If the map is missing a trailing pixel axis or the trailing axis does
        not match ``healpy.nside2npix(nside)``.
    TypeError
        If ``nside``, ``ordering``, or ``coord`` have unsupported types.
    """

    maps: NDArray[np.float64]
    nside: int
    ordering: HealpixOrdering = "RING"
    coord: str | None = None

    def __post_init__(self) -> None:
        """Normalize inputs and validate the trailing HEALPix pixel axis."""
        maps = _readonly_float_array(self.maps, "maps")
        _validate_nside(self.nside)
        expected_npix = _expected_npix(self.nside)

        if maps.ndim == 0:
            raise ValueError("maps must have a trailing HEALPix pixel axis")
        if maps.shape[-1] != expected_npix:
            raise ValueError(
                "maps trailing axis must match npix; "
                f"expected {expected_npix}, got {maps.shape[-1]}"
            )

        object.__setattr__(self, "maps", maps)
        object.__setattr__(self, "ordering", _normalize_ordering(self.ordering))
        object.__setattr__(self, "coord", _normalize_coord(self.coord))

    @property
    def npix(self) -> int:
        """Number of HEALPix pixels on the trailing axis."""
        return self.maps.shape[-1]

    @property
    def npix_expected(self) -> int:
        """Expected HEALPix pixel count for ``nside``."""
        return _expected_npix(self.nside)

    def assert_same_pixelization(self, other: HealpixMapContainer) -> None:
        """Validate that another map has matching HEALPix metadata.

        Parameters
        ----------
        other : HealpixMapContainer
            Map to compare with this container.

        Raises
        ------
        ValueError
            If ``nside``, pixel ordering, or coordinate frame differ.
        """
        if self.nside != other.nside:
            raise ValueError("nside values are incompatible")
        if self.ordering != other.ordering:
            raise ValueError("HEALPix orderings are incompatible")
        if self.coord != other.coord:
            raise ValueError("coordinate frames are incompatible")

    def copy_with(self, **changes: object) -> Self:
        """Return a copy with selected dataclass fields replaced.

        Parameters
        ----------
        **changes
            Field values passed to :func:`dataclasses.replace`.

        Returns
        -------
        BaseHealpixMap
            New container of the same concrete class, revalidated and with
            arrays copied into read-only storage.

        Raises
        ------
        ValueError
            If an unknown field name is supplied or replacement values fail
            validation.
        """
        valid_fields = {field.name for field in dataclass_fields(self)}
        unknown_fields = set(changes) - valid_fields
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise ValueError(f"Unknown map field replacement(s): {names}")
        return replace(self, **cast(Any, changes))

    def __repr__(self) -> str:
        """Return a short representation that omits the array body."""
        return (
            f"{type(self).__name__}("
            f"shape={self.maps.shape}, "
            f"dtype={self.maps.dtype}, "
            f"nside={self.nside}, "
            f"ordering={self.ordering!r}, "
            f"coord={self.coord!r})"
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class AuxiliaryHealpixMap(BaseHealpixMap):
    """Generic auxiliary HEALPix product such as a spectral-index map.

    Parameters
    ----------
    unit : str or None, default=None
        Optional physical unit of the auxiliary product.
    metadata : mapping, default={}
        Read-only metadata for provenance or model parameters.
    """

    unit: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optional auxiliary metadata."""
        super().__post_init__()
        if self.unit is not None:
            if not isinstance(self.unit, str):
                raise TypeError("unit must be a string or None")
            if self.unit == "":
                raise ValueError("unit must not be empty")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    def __repr__(self) -> str:
        """Return a short representation that omits the array body."""
        return (
            f"{type(self).__name__}("
            f"shape={self.maps.shape}, "
            f"dtype={self.maps.dtype}, "
            f"unit={self.unit!r}, "
            f"nside={self.nside}, "
            f"ordering={self.ordering!r}, "
            f"coord={self.coord!r})"
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class BaseSignalMap(BaseHealpixMap, SignalMapContainer):
    """Validated multi-frequency signal HEALPix map container.

    Parameters
    ----------
    freqs_ghz : ndarray
        Frequency channels in GHz. A scalar input is promoted to a length-one
        frequency axis.
    fields : tuple of {"T", "Q", "U"}
        Signal fields stored along axis 1.
    unit : str
        Unit of the map values.
    maps : ndarray
        Signal map array with shape ``(nfreq, nfield, npix)``.
    beam_fwhm_rad : float, ndarray, or None, default=None
        Beam FWHM in radians. Values may be scalar, per frequency with shape
        ``(nfreq,)``, or absent.
    """

    freqs_ghz: NDArray[np.float64]
    fields: tuple[SignalField, ...]
    unit: str
    beam_fwhm_rad: BeamFwhm = None

    def __post_init__(self) -> None:
        """Normalize signal metadata and validate the full map-axis contract."""
        freqs_ghz = _normalize_freqs(self.freqs_ghz)
        fields = _normalize_signal_fields(self.fields)
        super().__post_init__()

        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if self.unit == "":
            raise ValueError("unit must not be empty")

        if self.maps.ndim != 3:
            raise ValueError("maps must have shape (nfreq, nfield, npix)")
        expected_shape = (freqs_ghz.size, len(fields), self.npix_expected)
        if self.maps.shape != expected_shape:
            raise ValueError(
                "maps must have shape (nfreq, nfield, npix); "
                f"expected {expected_shape}, got {self.maps.shape}"
            )

        object.__setattr__(self, "freqs_ghz", freqs_ghz)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "beam_fwhm_rad",
            _normalize_beam(self.beam_fwhm_rad, freqs_ghz.size),
        )

    @property
    def nfreq(self) -> int:
        """Number of frequency channels."""
        return self.freqs_ghz.size

    @property
    def nfields(self) -> int:
        """Number of stored signal fields."""
        return len(self.fields)

    def field_index(self, field: SignalField) -> int:
        """Return the axis index for a signal field.

        Parameters
        ----------
        field : {"T", "Q", "U"}
            Field label to locate.

        Returns
        -------
        int
            Position of ``field`` along the signal-field axis.

        Raises
        ------
        ValueError
            If ``field`` is not present in this container.
        """
        try:
            return self.fields.index(field)
        except ValueError as error:
            raise ValueError(f"Field {field!r} is not present") from error

    def freq_index(self, freq_ghz: float, *, atol: float = 0.0) -> int:
        """Return the axis index for a frequency channel.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz to locate.
        atol : float, default=0.0
            Absolute tolerance used when matching the stored grid.

        Returns
        -------
        int
            Position of ``freq_ghz`` along the frequency axis.

        Raises
        ------
        ValueError
            If no channel matches, or if multiple channels match within
            ``atol``.
        """
        if atol < 0.0:
            raise ValueError("atol must be non-negative")

        matches = np.flatnonzero(
            np.isclose(self.freqs_ghz, freq_ghz, rtol=0.0, atol=atol)
        )
        if matches.size == 0:
            raise ValueError(f"Frequency {freq_ghz!r} GHz is not present")
        if matches.size > 1:
            raise ValueError(f"Frequency {freq_ghz!r} GHz matches multiple channels")
        return int(matches[0])

    def select_field(self, field: SignalField) -> Self:
        """Return a one-field view as a new read-only signal container."""
        index = self.field_index(field)
        return self.copy_with(maps=self.maps[:, index : index + 1, :], fields=(field,))

    def select_freq(self, freq_ghz: float, *, atol: float = 0.0) -> Self:
        """Return a one-frequency signal map container."""
        index = self.freq_index(freq_ghz, atol=atol)
        return self.copy_with(
            freqs_ghz=self.freqs_ghz[index : index + 1],
            maps=self.maps[index : index + 1, :, :],
            beam_fwhm_rad=_select_beam(self.beam_fwhm_rad, index),
        )

    def assert_compatible(
        self,
        other: SignalMapContainer,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        """Validate that another signal map has matching grid and metadata.

        Parameters
        ----------
        other : SignalMapContainer
            Signal map to compare with this container.
        rtol, atol : float, default=0.0
            Relative and absolute tolerances for frequency and beam comparisons.

        Raises
        ------
        ValueError
            If the maps are not compatible for arithmetic or stacking.
        """
        if rtol < 0.0 or atol < 0.0:
            raise ValueError("rtol and atol must be non-negative")

        self.assert_same_pixelization(other)
        if self.unit != other.unit:
            raise ValueError("map units are incompatible")
        if self.fields != other.fields:
            raise ValueError("map fields are incompatible")
        if not _arrays_close(self.freqs_ghz, other.freqs_ghz, rtol=rtol, atol=atol):
            raise ValueError("frequency grids are incompatible")

        self_beam = _beam_values(self.beam_fwhm_rad, self.nfreq)
        other_beam = _beam_values(other.beam_fwhm_rad, other.nfreq)
        if self_beam is None or other_beam is None:
            if self_beam is not other_beam:
                raise ValueError("beam FWHM values are incompatible")
        elif not _arrays_close(self_beam, other_beam, rtol=rtol, atol=atol):
            raise ValueError("beam FWHM values are incompatible")

    def __repr__(self) -> str:
        """Return a short representation that omits the map array body."""
        return (
            f"{type(self).__name__}("
            f"shape={self.maps.shape}, "
            f"freqs_ghz={self.freqs_ghz.tolist()}, "
            f"fields={self.fields}, "
            f"unit={self.unit!r}, "
            f"nside={self.nside}, "
            f"ordering={self.ordering!r}, "
            f"coord={self.coord!r})"
        )
