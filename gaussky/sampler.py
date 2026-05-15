from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from numpy.typing import ArrayLike

from gaussky.component.base import GaussianComponent
from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap, MultiFreqTotalMap


class Sampler:
    def __init__(
        self,
        nside: int,
    ) -> None:
        self.nside: int = nside

    @overload
    def sample(
        self,
        comp: GaussianComponent,
        fields: tuple[SignalField, ...] = ("T", "Q", "U"),
        ordering: HealpixOrdering = "RING",
        beam_fwhm_rad: BeamFwhm = 0.0,
        coord: str | None = None,
        freqs_ghz: ArrayLike | None = None,
    ) -> MultiFreqCompMap:
        ...

    @overload
    def sample(
        self,
        comp: Sequence[GaussianComponent],
        fields: tuple[SignalField, ...] = ("T", "Q", "U"),
        ordering: HealpixOrdering = "RING",
        beam_fwhm_rad: BeamFwhm = 0.0,
        coord: str | None = None,
        freqs_ghz: ArrayLike | None = None,
    ) -> MultiFreqTotalMap:
        ...

    def sample(
        self,
        comp: GaussianComponent | Sequence[GaussianComponent],
        fields: tuple[SignalField, ...] = ("T", "Q", "U"),
        ordering: HealpixOrdering = "RING",
        beam_fwhm_rad: BeamFwhm = 0.0,
        coord: str | None = None,
        freqs_ghz: ArrayLike | None = None,
    ) -> MultiFreqCompMap | MultiFreqTotalMap:
        """Sample one component map or a total map from component maps."""
        if isinstance(comp, GaussianComponent):
            return self._sample_component(
                comp,
                fields=fields,
                ordering=ordering,
                beam_fwhm_rad=beam_fwhm_rad,
                coord=coord,
                freqs_ghz=freqs_ghz,
            )

        if isinstance(comp, Sequence):
            components = tuple(comp)
            if not components:
                raise ValueError("components must contain at least one component")

            component_maps = tuple(
                self._sample_component(
                    component,
                    fields=fields,
                    ordering=ordering,
                    beam_fwhm_rad=beam_fwhm_rad,
                    coord=coord,
                    freqs_ghz=freqs_ghz,
                )
                for component in components
            )
            return MultiFreqTotalMap.from_components(component_maps)

        raise TypeError(
            "comp must be a GaussianComponent or a sequence of GaussianComponent "
            "instances"
        )

    def _sample_component(
        self,
        component: GaussianComponent,
        *,
        fields: tuple[SignalField, ...],
        ordering: HealpixOrdering,
        beam_fwhm_rad: BeamFwhm,
        coord: str | None,
        freqs_ghz: ArrayLike | None,
    ) -> MultiFreqCompMap:
        """Sample a component with the sampler's shared map request."""
        if not isinstance(component, GaussianComponent):
            raise TypeError(
                "components must be GaussianComponent instances with a name "
                "and sample_map method"
            )

        return component.sample_map(
            nside=self.nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            ordering=ordering,
            beam_fwhm_rad=beam_fwhm_rad,
            coord=coord,
        )
