"""Per-channel Gaussian white-noise component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap

from ..base import NoiseComponent
from ..component_utils import (
    normalize_seed,
    sample_per_channel_noise_map,
    validate_component_name,
)


@dataclass(frozen=True, kw_only=True)
class WhiteNoise(NoiseComponent):
    """Per-channel Gaussian white-noise component.

    Each (frequency, field, pixel) entry is an independent draw from
    ``Normal(0, sigma_pixel)`` where
    ``sigma_pixel = sigma_uK_arcmin / sqrt(pixel_area_arcmin²)``.
    Output unit is ``"uK_CMB"``.

    Parameters
    ----------
    sigma_uK_arcmin : float or ndarray
        Per-channel noise sensitivity in μK·arcmin. A scalar is shared
        across all sampled frequency channels; a one-dimensional array
        must have length equal to the resolved ``freqs_ghz`` at sample
        time.
    name : str, default="white_noise"
        Component name stored in sampled map products.
    """

    sigma_uK_arcmin: float | NDArray[np.float64]
    name: str = "white_noise"

    def __post_init__(self) -> None:
        """Validate the component name and the sensitivity entries.

        Raises
        ------
        ValueError
            If any sensitivity entry is non-finite or non-positive, or if
            the input is more than one-dimensional.
        """
        validate_component_name(self.name)
        sigma = np.asarray(self.sigma_uK_arcmin, dtype=np.float64)
        if sigma.ndim > 1:
            raise ValueError("sigma_uK_arcmin must be scalar or one-dimensional")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma_uK_arcmin must contain finite positive values")

        if sigma.ndim == 0:
            normalized: float | NDArray[np.float64] = float(sigma)
        else:
            normalized_array = sigma.astype(np.float64, copy=True)
            normalized_array.setflags(write=False)
            normalized = normalized_array
        object.__setattr__(self, "sigma_uK_arcmin", normalized)

    def sample_noise_map(
        self,
        *,
        nside: int,
        fields: tuple[SignalField, ...],
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
        seed: int | None = None,
    ) -> MultiFreqCompMap:
        """Draw a multi-frequency white-noise realization.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter.
        fields : tuple of {"T", "Q", "U"}
            Signal fields to retain, in output order.
        freqs_ghz : array_like or None, default=None
            Frequency channels in GHz. Required — white noise has no
            implicit pivot frequency to fall back to.
        beam_fwhm_rad : float, ndarray, or None, default=None
            Stored on the returned map for provenance only; not applied.
        ordering : {"RING", "NESTED"}, default="RING"
            HEALPix ordering for the returned map.
        coord : str or None, default=None
            Optional coordinate-frame label for the returned map.
        seed : int or None, default=None
            Optional seed for :func:`numpy.random.default_rng`.

        Returns
        -------
        MultiFreqCompMap
            Noise map with shape ``(nfreq, nfield, npix)`` and unit
            ``"uK_CMB"``.
        """
        normalized_seed = normalize_seed(seed)
        sigma = cast(
            "float | NDArray[np.float64]",
            self.sigma_uK_arcmin,
        )
        return sample_per_channel_noise_map(
            sigma_uK_arcmin=sigma,
            component_name=self.name,
            metadata=self._metadata(),
            nside=nside,
            fields=fields,
            freqs_ghz=freqs_ghz,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
            seed=normalized_seed,
        )

    def _metadata(self) -> dict[str, object]:
        """Return the model parameters carried by sampled noise maps."""
        sigma = self.sigma_uK_arcmin
        if isinstance(sigma, np.ndarray):
            return {"sigma_uK_arcmin": tuple(float(value) for value in sigma)}
        return {"sigma_uK_arcmin": float(sigma)}


__all__ = ["WhiteNoise"]
