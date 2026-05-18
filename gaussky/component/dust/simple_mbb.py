"""Simple Gaussian dust component with a modified-blackbody SED."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gaussky.ps import PowerLawCl
from gaussky.sed import ModifiedBlackbodySED

from ..sed_backed import BaseSEDBackedComponent


@dataclass(frozen=True, kw_only=True)
class SimpleModifiedBlackbodyDust(BaseSEDBackedComponent):
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

    def _build_sed(self) -> ModifiedBlackbodySED:
        return ModifiedBlackbodySED(
            beta=self.beta_d,
            temperature_k=self.temp_d,
            nu0_ghz=self.nu0_ghz,
        )

    def _metadata(self) -> Mapping[str, object]:
        return {
            "beta_d": self.beta_d,
            "temp_d": self.temp_d,
            "nu0_ghz": self.nu0_ghz,
        }

    @property
    def sed(self) -> ModifiedBlackbodySED:
        """Modified-blackbody SED derived from dust parameters."""
        return self._sed  # type: ignore[return-value]


__all__ = ["SimpleModifiedBlackbodyDust"]
