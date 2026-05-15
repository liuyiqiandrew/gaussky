"""Shared interfaces and containers for angular power spectra."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import HEALPY_POLARIZED_ORDER


class AngularPowerSpectrum(Protocol):
    """Angular ``C_ell`` model for Gaussian sky simulation.

    Implementations provide spectra on the integer multipole grid requested by
    simulation code in the polarized ordering consumed by
    :func:`healpy.synfast`.
    """

    unit: str

    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]:
        """Return spectra up to ``lmax`` in Healpy polarized ordering.

        Parameters
        ----------
        lmax : int
            Maximum multipole. Returned arrays must have length ``lmax + 1``
            and cover multipoles ``0`` through ``lmax``.

        Returns
        -------
        list of ndarray
            ``C_ell`` arrays ordered as ``TT, EE, BB, TE, EB, TB``.
        """
        ...


def validate_healpy_cls(
    healpy_cls: Sequence[ArrayLike],
    lmax: int,
    *,
    atol: float = 0.0,
) -> list[NDArray[np.float64]]:
    """Validate and normalize polarized Healpy ``C_ell`` arrays.

    Parameters
    ----------
    healpy_cls : sequence of array_like
        Six spectra ordered as ``TT, EE, BB, TE, EB, TB``.
    lmax : int
        Maximum multipole. Each spectrum must have length ``lmax + 1``.
    atol : float, default=0.0
        Absolute tolerance for small negative covariance eigenvalues.

    Returns
    -------
    list of ndarray
        Read-only ``float64`` arrays in Healpy polarized ordering.

    Raises
    ------
    ValueError
        If the spectra are not shaped for Healpy, contain non-finite values, or
        define a non-positive-semidefinite T/E/B covariance.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    if len(healpy_cls) != len(HEALPY_POLARIZED_ORDER):
        raise ValueError(
            "healpy_cls must contain six spectra ordered as TT, EE, BB, TE, EB, TB"
        )

    expected_shape = (lmax + 1,)
    normalized_cls: list[NDArray[np.float64]] = []
    for pair, values in zip(HEALPY_POLARIZED_ORDER, healpy_cls, strict=True):
        array = np.array(values, dtype=np.float64, copy=True)
        if array.ndim != 1:
            raise ValueError(f"{pair} spectrum must be one-dimensional")
        if array.shape != expected_shape:
            raise ValueError(
                f"{pair} spectrum must have length {lmax + 1}; "
                f"got shape {array.shape}"
            )
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{pair} spectrum must contain finite values")
        array.setflags(write=False)
        normalized_cls.append(array)

    covariance = _alm_covariance_from_healpy_cls(normalized_cls)
    eigvals = np.linalg.eigvalsh(covariance)
    if np.any(eigvals < -atol):
        raise ValueError("Power-spectrum covariance is not positive semidefinite")

    return normalized_cls


def _alm_covariance_from_healpy_cls(
    healpy_cls: Sequence[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Return T/E/B covariance matrices from normalized Healpy spectra."""
    nells = healpy_cls[0].size
    matrices = np.zeros((nells, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = healpy_cls[0]
    matrices[:, 1, 1] = healpy_cls[1]
    matrices[:, 2, 2] = healpy_cls[2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = healpy_cls[3]
    matrices[:, 1, 2] = matrices[:, 2, 1] = healpy_cls[4]
    matrices[:, 0, 2] = matrices[:, 2, 0] = healpy_cls[5]
    return matrices
