"""Simple Gaussian synchrotron component with a power-law SED."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from gaussky.ps import PowerLawCl
from gaussky.sed import PowerLawSED

from ..sed_backed import BaseSEDBackedComponent


@dataclass(frozen=True, kw_only=True)
class SimplePowerLawSynchrotron(BaseSEDBackedComponent):
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

    def _build_sed(self) -> PowerLawSED:
        return PowerLawSED(beta=self.beta_s, nu0_ghz=self.nu0_ghz)

    def _metadata(self) -> Mapping[str, object]:
        return {"beta_s": self.beta_s, "nu0_ghz": self.nu0_ghz}

    @property
    def sed(self) -> PowerLawSED:
        """Power-law SED derived from ``beta_s`` and ``nu0_ghz``."""
        return self._sed  # type: ignore[return-value]


__all__ = ["SimplePowerLawSynchrotron"]
