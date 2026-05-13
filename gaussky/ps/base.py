"""Shared interfaces and containers for angular power spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import HEALPY_POLARIZED_ORDER, SpectrumPair


class AngularPowerSpectrum(Protocol):
    """Angular ``C_ell`` definition for a Gaussian sky component."""

    unit: str

    def cl(self, pair: SpectrumPair, ell: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the angular power spectrum.

        Parameters
        ----------
        pair : {"TT", "EE", "BB", "TE", "EB", "TB"}
            Spectrum component to evaluate.
        ell : array_like
            Multipole or multipoles.

        Returns
        -------
        ndarray
            ``C_ell`` values in ``self.unit`` with the broadcast shape of
            ``ell``.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class ClSpectra:
    """Named collection of one-dimensional ``C_ell`` arrays.

    Parameters
    ----------
    ells : ndarray
        One-dimensional multipole grid.
    pairs : tuple of SpectrumPair
        Spectrum-pair labels for the first axis of ``values``.
    values : ndarray
        Spectra with shape ``(npair, nell)``.
    unit : str
        Physical unit of all spectra.
    """

    ells: NDArray[np.float64]
    pairs: tuple[SpectrumPair, ...]
    values: NDArray[np.float64]
    unit: str

    def __post_init__(self) -> None:
        """Normalize arrays and validate the named spectra."""
        ells = np.array(self.ells, dtype=np.float64, copy=True)
        values = np.array(self.values, dtype=np.float64, copy=True)
        pairs = tuple(self.pairs)

        if ells.ndim != 1:
            raise ValueError("ells must be one-dimensional")
        if np.any(ells < 0.0):
            raise ValueError("ells must be non-negative")
        if not pairs:
            raise ValueError("pairs must contain at least one spectrum pair")
        if len(set(pairs)) != len(pairs):
            raise ValueError("pairs must be unique")
        unknown_pairs = set(pairs) - set(HEALPY_POLARIZED_ORDER)
        if unknown_pairs:
            names = ", ".join(sorted(unknown_pairs))
            raise ValueError(f"Unknown spectrum pair(s): {names}")
        if values.shape != (len(pairs), ells.size):
            raise ValueError("values must have shape (npair, nell)")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if self.unit == "":
            raise ValueError("unit must not be empty")

        ells.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "ells", ells)
        object.__setattr__(self, "pairs", pairs)
        object.__setattr__(self, "values", values)

    def cls_for(self, pair: SpectrumPair) -> NDArray[np.float64]:
        """Return the spectrum array for one pair label.

        Parameters
        ----------
        pair : SpectrumPair
            Spectrum pair to locate.

        Returns
        -------
        ndarray
            Read-only one-dimensional ``C_ell`` array.
        """
        try:
            index = self.pairs.index(pair)
        except ValueError as error:
            raise ValueError(f"Spectrum pair {pair!r} is not present") from error
        return self.values[index]

    def to_healpy_cls(
        self,
        *,
        order: tuple[SpectrumPair, ...] = HEALPY_POLARIZED_ORDER,
    ) -> list[NDArray[np.float64]]:
        """Return spectra as a list in Healpy-compatible pair ordering."""
        return [self.cls_for(pair) for pair in order]


def cl_spectra_from_model(
    spectrum: AngularPowerSpectrum,
    lmax: int,
    *,
    pairs: tuple[SpectrumPair, ...] = HEALPY_POLARIZED_ORDER,
) -> ClSpectra:
    """Evaluate a power-spectrum model on the integer multipole grid.

    Parameters
    ----------
    spectrum : AngularPowerSpectrum
        Spectrum model to evaluate.
    lmax : int
        Maximum multipole. The returned spectra have length ``lmax + 1``.
    pairs : tuple of SpectrumPair, default=HEALPY_POLARIZED_ORDER
        Spectrum pairs to include.

    Returns
    -------
    ClSpectra
        Named spectra with one row per requested pair.

    Raises
    ------
    ValueError
        If ``lmax`` is negative.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")

    ells = np.arange(lmax + 1, dtype=np.float64)
    values = np.vstack([spectrum.cl(pair, ells) for pair in pairs])
    return ClSpectra(ells=ells, pairs=pairs, values=values, unit=spectrum.unit)
