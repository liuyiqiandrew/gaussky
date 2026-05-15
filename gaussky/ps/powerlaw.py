"""Power-law models for polarized angular power spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import (
    HEALPY_POLARIZED_ORDER,
    SpectrumPair,
    U_K_CMB_SQUARED,
)

from .base import AngularPowerSpectrum


_AMPLITUDE_FIELDS: dict[SpectrumPair, str] = {
    "TT": "amp_tt",
    "EE": "amp_ee",
    "BB": "amp_bb",
    "TE": "amp_te",
    "EB": "amp_eb",
    "TB": "amp_tb",
}

_SLOPE_FIELDS: dict[SpectrumPair, str] = {
    "TT": "alpha_tt",
    "EE": "alpha_ee",
    "BB": "alpha_bb",
    "TE": "alpha_te",
    "EB": "alpha_eb",
    "TB": "alpha_tb",
}


def _normalize_pair(pair: str) -> SpectrumPair:
    """Validate and canonicalize a spectrum-pair label."""
    normalized = pair.upper()
    if normalized not in HEALPY_POLARIZED_ORDER:
        valid = ", ".join(HEALPY_POLARIZED_ORDER)
        raise ValueError(f"Unknown spectrum pair {pair!r}; expected one of {valid}")
    return cast(SpectrumPair, normalized)


@dataclass(frozen=True)
class PowerLawCl(AngularPowerSpectrum):
    """Power-law angular ``C_ell`` model.

    The amplitudes are ``C_ell`` values at ``ell0``. This class does not
    interpret amplitudes as ``D_ell = ell (ell + 1) C_ell / 2 pi`` unless
    ``is_cell`` is set to ``False``. The :meth:`cl` method is retained as a
    convenience for analytic evaluations, while :meth:`to_healpy_cls` is the
    simulation-facing protocol used by map samplers.

    Parameters
    ----------
    amp_tt, amp_ee, amp_bb, amp_te, amp_eb, amp_tb : float, default=0.0
        Power-spectrum amplitudes at ``ell0`` in ``unit``.
    alpha_tt, alpha_ee, alpha_bb, alpha_te, alpha_eb, alpha_tb : float,
        default=0.0
        Power-law slopes for each spectrum component.
    ell0 : float, default=80.0
        Reference multipole for all amplitudes.
    ell_min : int, default=2
        Multipoles below this value are forced to zero.
    unit : str, default="uK_CMB^2"
        Unit of the returned ``C_ell`` values.
    is_cell : bool, default=True
        If ``False``, amplitudes are interpreted as ``D_ell`` values and are
        converted to ``C_ell`` on output.

    Raises
    ------
    ValueError
        If the reference multipole or minimum multipole is invalid, if any
        amplitude or slope is non-finite, or if an auto-spectrum amplitude is
        negative.
    """

    amp_tt: float = 0.0
    amp_ee: float = 0.0
    amp_bb: float = 0.0
    amp_te: float = 0.0
    amp_eb: float = 0.0
    amp_tb: float = 0.0
    alpha_tt: float = 0.0
    alpha_ee: float = 0.0
    alpha_bb: float = 0.0
    alpha_te: float = 0.0
    alpha_eb: float = 0.0
    alpha_tb: float = 0.0
    ell0: float = 80.0
    ell_min: int = 2
    unit: str = U_K_CMB_SQUARED
    is_cell: bool = True

    def __post_init__(self) -> None:
        """Validate scalar model parameters."""
        if self.ell0 <= 0.0:
            raise ValueError("ell0 must be strictly positive")
        if self.ell_min < 0:
            raise ValueError("ell_min must be non-negative")

        for pair in HEALPY_POLARIZED_ORDER:
            amp = getattr(self, _AMPLITUDE_FIELDS[pair])
            alpha = getattr(self, _SLOPE_FIELDS[pair])
            if not np.isfinite(amp):
                raise ValueError(f"{_AMPLITUDE_FIELDS[pair]} must be finite")
            if not np.isfinite(alpha):
                raise ValueError(f"{_SLOPE_FIELDS[pair]} must be finite")
            if pair in {"TT", "EE", "BB"} and amp < 0.0:
                raise ValueError(f"{_AMPLITUDE_FIELDS[pair]} must be non-negative")

    def cl(self, pair: SpectrumPair, ell: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``C_ell`` for a spectrum pair.

        Parameters
        ----------
        pair : {"TT", "EE", "BB", "TE", "EB", "TB"}
            Spectrum component to evaluate.
        ell : array_like
            Multipole or multipoles.

        Returns
        -------
        ndarray
            ``C_ell`` values with the same shape as ``ell``. Multipoles below
            ``ell_min`` are zero.

        Raises
        ------
        ValueError
            If ``pair`` is unknown or any multipole is negative.
        """
        normalized = _normalize_pair(pair)
        ell_array = np.asarray(ell, dtype=np.float64)
        if np.any(ell_array < 0.0):
            raise ValueError("ell must be non-negative")

        values = np.zeros_like(ell_array, dtype=np.float64)
        # Keep the low-ell convention explicit: the model is defined only above
        # ell_min, while masked entries remain exactly zero.
        mask = ell_array >= self.ell_min

        if np.any(mask):
            amp = getattr(self, _AMPLITUDE_FIELDS[normalized])
            alpha = getattr(self, _SLOPE_FIELDS[normalized])
            values[mask] = amp * (ell_array[mask] / self.ell0) ** alpha
            if not self.is_cell:
                # Some literature quotes amplitudes in D_ell; downstream code
                # consumes C_ell, so apply the standard conversion factor here.
                dl2cl = np.zeros_like(ell_array)
                dl2cl[mask] = 2 * np.pi / (ell_array[mask] * (ell_array[mask] + 1.0))
                values *= dl2cl

        return values

    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]:
        """Build spectra in Healpy's polarized ordering.

        Parameters
        ----------
        lmax : int
            Maximum multipole. Returned arrays have length ``lmax + 1``.

        Returns
        -------
        list of ndarray
            Spectra ordered as ``TT, EE, BB, TE, EB, TB``.

        Raises
        ------
        ValueError
            If ``lmax`` is negative.
        """
        if lmax < 0:
            raise ValueError("lmax must be non-negative")

        ells = np.arange(lmax + 1, dtype=np.float64)
        return [self.cl(pair, ells) for pair in HEALPY_POLARIZED_ORDER]

    def alm_covariance(self, ell: ArrayLike) -> NDArray[np.float64]:
        """Return T/E/B covariance matrices at each multipole.

        Parameters
        ----------
        ell : array_like
            Multipole or multipoles.

        Returns
        -------
        ndarray
            Array with shape ``ell.shape + (3, 3)`` using field order
            ``T, E, B``.

        Raises
        ------
        ValueError
            If any multipole is negative.
        """
        ell_array = np.asarray(ell, dtype=np.float64)
        matrices = np.zeros(ell_array.shape + (3, 3), dtype=np.float64)

        # Healpy stores spectra as named pair arrays; the covariance matrix
        # needs the corresponding symmetric T/E/B block at each ell.
        matrices[..., 0, 0] = self.cl("TT", ell_array)
        matrices[..., 1, 1] = self.cl("EE", ell_array)
        matrices[..., 2, 2] = self.cl("BB", ell_array)
        matrices[..., 0, 1] = matrices[..., 1, 0] = self.cl("TE", ell_array)
        matrices[..., 0, 2] = matrices[..., 2, 0] = self.cl("TB", ell_array)
        matrices[..., 1, 2] = matrices[..., 2, 1] = self.cl("EB", ell_array)
        return matrices

    def validate_positive_semidefinite(
        self, ell: ArrayLike, *, atol: float = 0.0
    ) -> None:
        """Validate that T/E/B covariance matrices are positive semidefinite.

        Parameters
        ----------
        ell : array_like
            Multipole or multipoles to check.
        atol : float, default=0.0
            Absolute tolerance for tiny negative eigenvalues from roundoff.

        Raises
        ------
        ValueError
            If any checked covariance matrix has an eigenvalue below
            ``-atol``.
        """
        eigvals = np.linalg.eigvalsh(self.alm_covariance(ell))
        if np.any(eigvals < -atol):
            raise ValueError("Power-spectrum covariance is not positive semidefinite")
