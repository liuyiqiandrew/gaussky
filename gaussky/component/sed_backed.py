"""Shared scaffolding for SED-backed Gaussian components.

Concrete foreground components (synchrotron, dust, and future PySM-style
models) all follow the same recipe: take a polarized angular power spectrum
``ps`` at a pivot frequency ``nu0_ghz``, derive an
:class:`~gaussky.sed.SpectralEnergyDistribution`, and forward to the unified
:func:`~gaussky.component.component_utils.sample_component_map` helper. This
module factors that recipe into a single base dataclass so each new component
only has to declare its parameters, its SED factory, and a metadata mapping.

CMB-style frequency-independent components do **not** subclass this base;
they delegate directly to ``sample_component_map(sed=None, ...)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from numpy.typing import ArrayLike

from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import AngularPowerSpectrum
from gaussky.sed import SpectralEnergyDistribution

from .base import GaussianComponent
from .component_utils import sample_component_map, validate_component_name


@dataclass(frozen=True, kw_only=True)
class BaseSEDBackedComponent(GaussianComponent):
    """Frozen-dataclass base for components that pair a spectrum with an SED.

    Subclasses must:

    - declare their model parameters (including their own narrowly-typed
      ``ps`` and a default value for ``name``);
    - implement :meth:`_build_sed` to return the SED instance derived from
      those parameters;
    - implement :meth:`_metadata` to expose those parameters in the sampled
      :class:`~gaussky.map.MultiFreqCompMap`'s ``metadata`` mapping.

    The base owns the ``__post_init__`` name validation, caches the derived
    SED on a frozen instance via ``object.__setattr__``, exposes it through
    the read-only :attr:`sed` property, and provides a default
    :meth:`sample_map` that forwards to
    :func:`~gaussky.component.component_utils.sample_component_map`. Subclasses
    that need extra validation should override ``__post_init__`` and call
    ``super().__post_init__()`` first.
    """

    ps: AngularPowerSpectrum
    name: str
    _sed: SpectralEnergyDistribution = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the component name and cache the derived SED."""
        validate_component_name(self.name)
        object.__setattr__(self, "_sed", self._build_sed())

    def _build_sed(self) -> SpectralEnergyDistribution:
        """Return the SED implied by this component's parameters."""
        raise NotImplementedError(f"{type(self).__name__} must implement _build_sed()")

    def _metadata(self) -> Mapping[str, object]:
        """Return the model parameters carried by the sampled map."""
        return {}

    @property
    def sed(self) -> SpectralEnergyDistribution:
        """Frequency-scaling SED derived from this component's parameters."""
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
        seed: int | None = None,
        lmax: int | None = None,
    ) -> MultiFreqCompMap:
        """Sample a multi-frequency map realization using the cached SED."""
        return sample_component_map(
            ps=self.ps,
            sed=self.sed,
            component_name=self.name,
            metadata=self._metadata(),
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
            seed=seed,
            lmax=lmax,
        )


__all__ = ["BaseSEDBackedComponent"]
