"""Dust component implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import PowerLawCl
from gaussky.sed import ModifiedBlackbodySED

from .base import GaussianComponent
from .component_utils import sample_gaussian_component_map, validate_component_name


@dataclass(frozen=True, kw_only=True)
class SimpleModifiedBlackbodyDust(GaussianComponent):
    """Simple Gaussian dust component with a modified-blackbody SED.

    The component samples one pivot-frequency Gaussian T/Q/U realization from
    ``ps`` and scales it to each requested frequency with
    :class:`gaussky.sed.ModifiedBlackbodySED`.

    Parameters
    ----------
    ps : PowerLawCl
        Polarized angular power spectrum at ``nu0_ghz``.
    beta_d : float
        Dust spectral index in Rayleigh-Jeans temperature units.
    temp_d : float
        Dust temperature in Kelvin.
    nu0_ghz : float
        Reference frequency in GHz where the SED scaling is one.
    name : str, default="dust"
        Component name stored in sampled map products.
    """

    ps: PowerLawCl
    beta_d: float
    temp_d: float
    nu0_ghz: float
    name: str = "dust"
    _sed: ModifiedBlackbodySED = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate component parameters and cache the derived SED."""
        validate_component_name(self.name)
        object.__setattr__(
            self,
            "_sed",
            ModifiedBlackbodySED(
                beta=self.beta_d,
                temperature_k=self.temp_d,
                nu0_ghz=self.nu0_ghz,
            ),
        )

    @property
    def sed(self) -> ModifiedBlackbodySED:
        """Modified-blackbody SED derived from dust parameters."""
        return self._sed

    def sample_map(
        self,
        *,
        nside: int,
        fields: tuple[SignalField, ...],
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
    ) -> MultiFreqCompMap:
        """Sample a multi-frequency dust map realization.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter.
        freqs_ghz : array_like or None, default=None
            Frequency channels in GHz. If ``None``, only ``nu0_ghz`` is sampled
            and the SED scaling is one.
        fields : tuple of {"T", "Q", "U"}
            Signal fields to retain, in output order.
        beam_fwhm_rad : float, ndarray, or None, default=None
            Beam FWHM in radians. A scalar beam is shared across channels; a
            one-dimensional array supplies one beam per frequency.
        ordering : {"RING", "NESTED"}, default="RING"
            HEALPix ordering for the returned map.
        coord : str or None, default=None
            Optional coordinate-frame label for the returned map.

        Returns
        -------
        MultiFreqCompMap
            Component map with shape ``(nfreq, nfield, npix)``.
        """
        return sample_gaussian_component_map(
            ps=self.ps,
            sed=self.sed,
            component_name=self.name,
            metadata={
                "beta_d": self.beta_d,
                "temp_d": self.temp_d,
                "nu0_ghz": self.nu0_ghz,
            },
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
        )


__all__ = ["SimpleModifiedBlackbodyDust"]
