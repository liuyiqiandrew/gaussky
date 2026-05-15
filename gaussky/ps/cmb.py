"""Template-backed CMB angular power spectra."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from gaussky.conventions import HEALPY_POLARIZED_ORDER, SpectrumPair, U_K_CMB_SQUARED

from .base import AngularPowerSpectrum


DEFAULT_CMB_SPEC_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent / "data" / "cmb_spec"
)
_LENSED_NO_TENSOR_FILENAME: Final[str] = "camb_lens_nobb.dat"
_LENSED_R1_FILENAME: Final[str] = "camb_lens_r1.dat"
_CAMB_FILE_ORDER: Final[tuple[SpectrumPair, ...]] = ("TT", "EE", "BB", "TE")


@dataclass(frozen=True, kw_only=True)
class _CMBTemplateSet:
    """Loaded CAMB ``D_ell`` templates on a shared integer multipole grid."""

    ells: NDArray[np.int64]
    lensed_no_tensor_dl: NDArray[np.float64]
    lensed_r1_dl: NDArray[np.float64]
    source_dir: Path

    def __post_init__(self) -> None:
        """Validate template shapes and freeze array storage."""
        ells = np.array(self.ells, dtype=np.int64, copy=True)
        lensed_no_tensor_dl = np.array(
            self.lensed_no_tensor_dl, dtype=np.float64, copy=True
        )
        lensed_r1_dl = np.array(self.lensed_r1_dl, dtype=np.float64, copy=True)

        if ells.ndim != 1:
            raise ValueError("CMB template multipoles must be one-dimensional")
        if ells.size == 0:
            raise ValueError("CMB templates must contain at least one multipole")
        if not np.array_equal(ells, np.arange(ells[-1] + 1)):
            raise ValueError("CMB template multipoles must cover 0..lmax")

        expected_shape = (len(_CAMB_FILE_ORDER), ells.size)
        if lensed_no_tensor_dl.shape != expected_shape:
            raise ValueError("lensed-no-tensor template has an invalid shape")
        if lensed_r1_dl.shape != expected_shape:
            raise ValueError("r=1 template has an invalid shape")

        ells.setflags(write=False)
        lensed_no_tensor_dl.setflags(write=False)
        lensed_r1_dl.setflags(write=False)
        object.__setattr__(self, "ells", ells)
        object.__setattr__(self, "lensed_no_tensor_dl", lensed_no_tensor_dl)
        object.__setattr__(self, "lensed_r1_dl", lensed_r1_dl)

    @property
    def lmax(self) -> int:
        """Maximum multipole available in the bundled templates."""
        return int(self.ells[-1])


def _read_camb_dl_template(path: Path) -> NDArray[np.float64]:
    """Read one CAMB ``D_ell`` template as ``(TT, EE, BB, TE, ell)``."""
    raw = np.atleast_2d(np.loadtxt(path))
    if raw.ndim != 2 or raw.shape[1] != len(_CAMB_FILE_ORDER) + 1:
        raise ValueError(
            f"{path} must have columns ell, TT, EE, BB, TE; got shape {raw.shape}"
        )

    ell_values = raw[:, 0]
    ell_indices = ell_values.astype(np.int64)
    if not np.allclose(ell_values, ell_indices):
        raise ValueError(f"{path} contains non-integer multipoles")
    if np.any(ell_indices < 0):
        raise ValueError(f"{path} contains negative multipoles")
    if len(set(ell_indices.tolist())) != ell_indices.size:
        raise ValueError(f"{path} contains duplicate multipoles")

    lmax = int(ell_indices.max(initial=0))
    first_ell = int(ell_indices[0])
    if first_ell not in {0, 1}:
        raise ValueError(f"{path} must start at ell=0 or ell=1")
    if not np.array_equal(ell_indices, np.arange(first_ell, lmax + 1)):
        raise ValueError(f"{path} must contain consecutive multipoles")

    dl = np.zeros((len(_CAMB_FILE_ORDER), lmax + 1), dtype=np.float64)
    dl[:, ell_indices] = raw[:, 1:].T
    return dl


@lru_cache(maxsize=None)
def _load_cmb_templates(template_dir: str) -> _CMBTemplateSet:
    """Load and cache the CMB template pair from one directory."""
    source_dir = Path(template_dir)
    lensed_no_tensor = _read_camb_dl_template(source_dir / _LENSED_NO_TENSOR_FILENAME)
    lensed_r1 = _read_camb_dl_template(source_dir / _LENSED_R1_FILENAME)
    if lensed_no_tensor.shape != lensed_r1.shape:
        raise ValueError("CMB template files must share the same multipole range")

    ells = np.arange(lensed_no_tensor.shape[1], dtype=np.int64)
    return _CMBTemplateSet(
        ells=ells,
        lensed_no_tensor_dl=lensed_no_tensor,
        lensed_r1_dl=lensed_r1,
        source_dir=source_dir,
    )


def _dl_to_cl_scale(lmax: int) -> NDArray[np.float64]:
    """Return the per-ell conversion from ``D_ell`` to ``C_ell``."""
    ells = np.arange(lmax + 1, dtype=np.float64)
    scale = np.zeros_like(ells)
    valid = ells >= 2.0
    scale[valid] = 2.0 * np.pi / (ells[valid] * (ells[valid] + 1.0))
    return scale


@dataclass(frozen=True, kw_only=True)
class CMBCl(AngularPowerSpectrum):
    """CMB angular power spectrum from CAMB lensing and ``r=1`` templates.

    The bundled CAMB files store ``D_ell = ell (ell + 1) C_ell / 2 pi`` with
    columns ``ell, TT, EE, BB, TE``. This class combines the lensed no-tensor
    and lensed ``r=1`` templates as

    ``D_ell = a_lens * D_ell^lens + r_tensor * (D_ell^r1 - D_ell^lens)``

    and then converts to ``C_ell``. The CMB model uses ``TB = EB = 0``.

    Parameters
    ----------
    a_lens : float, default=1.0
        Lensing-template amplitude. ``1`` gives the template value and ``0``
        removes the lensed no-tensor contribution in this linear combination.
    r_tensor : float, default=0.0
        Tensor-to-scalar ratio multiplying the difference between the ``r=1``
        and no-tensor templates.
    template_dir : str or Path, default=DEFAULT_CMB_SPEC_DIR
        Directory containing ``camb_lens_nobb.dat`` and ``camb_lens_r1.dat``.
    unit : str, default="uK_CMB^2"
        Unit of the returned ``C_ell`` values.

    Raises
    ------
    ValueError
        If scalar parameters are non-finite, negative, or if ``unit`` is not
        supported.
    """

    a_lens: float = 1.0
    r_tensor: float = 0.0
    template_dir: str | Path = DEFAULT_CMB_SPEC_DIR
    unit: str = U_K_CMB_SQUARED

    def __post_init__(self) -> None:
        """Validate scalar CMB spectrum parameters."""
        if not np.isfinite(self.a_lens) or self.a_lens < 0.0:
            raise ValueError("a_lens must be finite and non-negative")
        if not np.isfinite(self.r_tensor) or self.r_tensor < 0.0:
            raise ValueError("r_tensor must be finite and non-negative")
        if self.unit != U_K_CMB_SQUARED:
            raise ValueError(f"CMBCl only supports unit {U_K_CMB_SQUARED!r}")

    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]:
        """Return CMB ``C_ell`` spectra through ``lmax``.

        Parameters
        ----------
        lmax : int
            Maximum multipole to return. Must not exceed the template range.

        Returns
        -------
        list of ndarray
            Spectra ordered as ``TT, EE, BB, TE, EB, TB`` and expressed in
            ``uK_CMB^2``. ``TB`` and ``EB`` are exactly zero.

        Raises
        ------
        ValueError
            If ``lmax`` is negative or larger than the loaded templates.
        """
        if lmax < 0:
            raise ValueError("lmax must be non-negative")

        templates = _load_cmb_templates(str(Path(self.template_dir)))
        if lmax > templates.lmax:
            raise ValueError(
                f"lmax={lmax} exceeds the CMB template range "
                f"(lmax={templates.lmax})"
            )

        combined_dl = self.a_lens * templates.lensed_no_tensor_dl[
            :, : lmax + 1
        ] + self.r_tensor * (
            templates.lensed_r1_dl[:, : lmax + 1]
            - templates.lensed_no_tensor_dl[:, : lmax + 1]
        )
        combined_cl = combined_dl * _dl_to_cl_scale(lmax)[None, :]

        values = np.zeros((len(HEALPY_POLARIZED_ORDER), lmax + 1), dtype=np.float64)
        for source_index, pair in enumerate(_CAMB_FILE_ORDER):
            target_index = HEALPY_POLARIZED_ORDER.index(pair)
            values[target_index] = combined_cl[source_index]

        return [values[index] for index in range(len(HEALPY_POLARIZED_ORDER))]
