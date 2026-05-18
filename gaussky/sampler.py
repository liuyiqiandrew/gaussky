from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import numpy as np
from numpy.typing import ArrayLike

from gaussky.component.base import GaussianComponent
from gaussky.component.component_utils import normalize_seed
from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap, MultiFreqTotalMap


_CHILD_SEED_HIGH = 2**32 - 1


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
        seed: int | None = None,
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
        seed: int | None = None,
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
        seed: int | None = None,
    ) -> MultiFreqCompMap | MultiFreqTotalMap:
        """Sample one component map or a total map from component maps."""
        normalized_seed = normalize_seed(seed)
        if isinstance(comp, GaussianComponent):
            return self._sample_component(
                comp,
                fields=fields,
                ordering=ordering,
                beam_fwhm_rad=beam_fwhm_rad,
                coord=coord,
                freqs_ghz=freqs_ghz,
                seed=normalized_seed,
            )

        if isinstance(comp, Sequence):
            components = self._normalize_components(comp)
            component_seeds = self._component_seeds(
                components,
                root_seed=normalized_seed,
            )

            component_maps = tuple(
                self._sample_component(
                    component,
                    fields=fields,
                    ordering=ordering,
                    beam_fwhm_rad=beam_fwhm_rad,
                    coord=coord,
                    freqs_ghz=freqs_ghz,
                    seed=component_seeds[component.name],
                )
                for component in components
            )
            return MultiFreqTotalMap.from_components(
                component_maps,
                metadata={
                    "seed": normalized_seed,
                    "component_seeds": component_seeds,
                },
            )

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
        seed: int | None,
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
            seed=seed,
        )

    def _normalize_components(
        self,
        components: Sequence[GaussianComponent],
    ) -> tuple[GaussianComponent, ...]:
        """Validate a non-empty component sequence before sampling."""
        component_tuple = tuple(components)
        if not component_tuple:
            raise ValueError("components must contain at least one component")

        for component in component_tuple:
            if not isinstance(component, GaussianComponent):
                raise TypeError(
                    "components must be GaussianComponent instances with a name "
                    "and sample_map method"
                )

        names = [component.name for component in component_tuple]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        return component_tuple

    def _component_seeds(
        self,
        components: tuple[GaussianComponent, ...],
        *,
        root_seed: int | None,
    ) -> dict[str, int | None]:
        """Return per-component seeds derived from an optional root seed."""
        if root_seed is None:
            return {component.name: None for component in components}

        rng = np.random.RandomState(root_seed)
        child_seeds = rng.randint(0, _CHILD_SEED_HIGH, size=len(components))
        return {
            component.name: int(child_seed)
            for component, child_seed in zip(components, child_seeds, strict=True)
        }
