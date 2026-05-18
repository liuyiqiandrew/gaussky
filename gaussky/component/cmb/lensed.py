"""Frequency-independent Gaussian CMB component backed by lensing templates."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import CMBCl

from ..base import GaussianComponent
from ..component_utils import sample_component_map, validate_component_name


@dataclass(frozen=True, kw_only=True)
class GaussianCMB(GaussianComponent):
    """Frequency-independent Gaussian CMB component.

    The component samples one CMB T/Q/U realization from :class:`gaussky.ps.CMBCl`
    and repeats it across all requested frequency channels. Frequencies label
    the output channels and allow per-channel beam smoothing; they do not apply
    an SED or require a pivot frequency.

    Parameters
    ----------
    ps : CMBCl, default=CMBCl()
        CMB angular power-spectrum model used for the Gaussian realization.
    name : str, default="cmb"
        Component name stored in sampled map products.
    """

    ps: CMBCl = field(default_factory=CMBCl)
    name: str = "cmb"

    def __post_init__(self) -> None:
        """Validate component parameters."""
        validate_component_name(self.name)

    def sample_map(
        self,
        *,
        nside: int,
        fields: tuple[SignalField, ...],
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
        seed: int | None = None,
        lmax: int | None = None,
    ) -> MultiFreqCompMap:
        """Sample a multi-frequency CMB map realization.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter.
        freqs_ghz : array_like
            Frequency channels in GHz. CMB signal amplitudes are independent of
            this grid, but the returned map keeps one channel per frequency.
            Unlike SED-backed components, this value is required because CMB has
            no pivot frequency.
        fields : tuple of {"T", "Q", "U"}
            Signal fields to retain, in output order.
        beam_fwhm_rad : float, ndarray, or None, default=None
            Beam FWHM in radians. A scalar beam is shared across channels; a
            one-dimensional array supplies one beam per frequency.
        ordering : {"RING", "NESTED"}, default="RING"
            HEALPix ordering for the returned map.
        coord : str or None, default=None
            Optional coordinate-frame label for the returned map.
        seed : int or None, default=None
            Optional NumPy legacy RNG seed used for the Healpy realization.

        Returns
        -------
        MultiFreqCompMap
            Component map with shape ``(nfreq, nfield, npix)``.
        """
        return sample_component_map(
            ps=self.ps,
            sed=None,
            component_name=self.name,
            metadata={
                "a_lens": self.ps.a_lens,
                "r_tensor": self.ps.r_tensor,
                "template_dir": str(self.ps.template_dir),
            },
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
            seed=seed,
            lmax=lmax,
        )


__all__ = ["GaussianCMB"]
