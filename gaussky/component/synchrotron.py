"""Synchrotron component implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import PowerLawCl
from gaussky.sed import PowerLawSED

from .base import GaussianComponent
from .component_utils import sample_gaussian_component_map, validate_component_name


@dataclass(frozen=True, kw_only=True)
class SimplePowerLawSynchrotron(GaussianComponent):
    """Simple Gaussian synchrotron component with a power-law SED.

    The component samples one pivot-frequency Gaussian T/Q/U realization from
    ``ps`` and scales it to each requested frequency with
    :class:`gaussky.sed.PowerLawSED`.

    Parameters
    ----------
    ps : PowerLawCl
        Polarized angular power spectrum at ``nu0_ghz``.
    beta_s : float
        Synchrotron spectral index in Rayleigh-Jeans temperature units.
    nu0_ghz : float
        Reference frequency in GHz where the SED scaling is one.
    name : str, default="synchrotron"
        Component name stored in sampled map products.
    """

    ps: PowerLawCl
    beta_s: float
    nu0_ghz: float
    name: str = "synchrotron"
    _sed: PowerLawSED = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate component parameters and cache the derived SED."""
        validate_component_name(self.name)
        object.__setattr__(
            self,
            "_sed",
            PowerLawSED(beta=self.beta_s, nu0_ghz=self.nu0_ghz),
        )

    @property
    def sed(self) -> PowerLawSED:
        """Power-law SED derived from ``beta_s`` and ``nu0_ghz``."""
        return self._sed

    def sample_map(
        self,
        *,
        nside: int,
        freqs_ghz: ArrayLike,
        fields: tuple[SignalField, ...],
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
    ) -> MultiFreqCompMap:
        """Sample a multi-frequency synchrotron map realization.

        Parameters
        ----------
        nside : int
            HEALPix resolution parameter.
        freqs_ghz : array_like
            Frequency channels in GHz.
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
            metadata={"beta_s": self.beta_s, "nu0_ghz": self.nu0_ghz},
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
        )
