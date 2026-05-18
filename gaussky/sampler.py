"""High-level multi-component sky-sampling facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Final, overload

import numpy as np
from numpy.typing import ArrayLike

from gaussky.component.base import GaussianComponent
from gaussky.component.component_utils import normalize_seed
from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap, MultiFreqTotalMap


_CHILD_SEED_HIGH: Final[int] = 2**32 - 1


class _Unset:
    """Sentinel singleton meaning 'argument not supplied; use the stored default'."""

    _instance: ClassVar["_Unset | None"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET: Final[_Unset] = _Unset()


class Sampler:
    """Facade that samples one or more components into a multi-frequency sky map.

    The constructor accepts per-scene defaults for the sampling options
    (``fields``, ``freqs_ghz``, ``beam_fwhm_rad``, ``coord``, ``ordering``,
    ``seed``). ``Sampler.sample`` accepts the same keyword arguments as
    overrides; pass an explicit value to override the default, or omit the
    argument to inherit it.

    Use :meth:`with_` to obtain a sibling sampler with one or more defaults
    replaced (e.g. ``sampler.with_(seed=7)``) without mutating the original.
    """

    def __init__(
        self,
        nside: int,
        *,
        fields: tuple[SignalField, ...] = ("T", "Q", "U"),
        freqs_ghz: ArrayLike | None = None,
        beam_fwhm_rad: BeamFwhm = 0.0,
        coord: str | None = None,
        ordering: HealpixOrdering = "RING",
        seed: int | None = None,
        lmax: int | None = None,
    ) -> None:
        self.nside: int = nside
        self.fields: tuple[SignalField, ...] = fields
        self.freqs_ghz: ArrayLike | None = freqs_ghz
        self.beam_fwhm_rad: BeamFwhm = beam_fwhm_rad
        self.coord: str | None = coord
        self.ordering: HealpixOrdering = ordering
        self.seed: int | None = seed
        self.lmax: int | None = lmax

    def with_(self, **changes: Any) -> "Sampler":
        """Return a new :class:`Sampler` with the supplied defaults replaced.

        Recognized keys: ``nside``, ``fields``, ``freqs_ghz``,
        ``beam_fwhm_rad``, ``coord``, ``ordering``, ``seed``, ``lmax``.
        Unknown keys raise :class:`TypeError`.
        """
        allowed = {
            "nside",
            "fields",
            "freqs_ghz",
            "beam_fwhm_rad",
            "coord",
            "ordering",
            "seed",
            "lmax",
        }
        unknown = set(changes) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown Sampler default(s): {names}")

        return Sampler(
            nside=changes.get("nside", self.nside),
            fields=changes.get("fields", self.fields),
            freqs_ghz=changes.get("freqs_ghz", self.freqs_ghz),
            beam_fwhm_rad=changes.get("beam_fwhm_rad", self.beam_fwhm_rad),
            coord=changes.get("coord", self.coord),
            ordering=changes.get("ordering", self.ordering),
            seed=changes.get("seed", self.seed),
            lmax=changes.get("lmax", self.lmax),
        )

    @overload
    def sample(
        self,
        comp: GaussianComponent,
        *,
        fields: tuple[SignalField, ...] | _Unset = _UNSET,
        freqs_ghz: ArrayLike | None | _Unset = _UNSET,
        beam_fwhm_rad: BeamFwhm | _Unset = _UNSET,
        coord: str | None | _Unset = _UNSET,
        ordering: HealpixOrdering | _Unset = _UNSET,
        seed: int | None | _Unset = _UNSET,
        lmax: int | None | _Unset = _UNSET,
    ) -> MultiFreqCompMap: ...

    @overload
    def sample(
        self,
        comp: Sequence[GaussianComponent],
        *,
        fields: tuple[SignalField, ...] | _Unset = _UNSET,
        freqs_ghz: ArrayLike | None | _Unset = _UNSET,
        beam_fwhm_rad: BeamFwhm | _Unset = _UNSET,
        coord: str | None | _Unset = _UNSET,
        ordering: HealpixOrdering | _Unset = _UNSET,
        seed: int | None | _Unset = _UNSET,
        lmax: int | None | _Unset = _UNSET,
    ) -> MultiFreqTotalMap: ...

    def sample(
        self,
        comp: GaussianComponent | Sequence[GaussianComponent],
        *,
        fields: tuple[SignalField, ...] | _Unset = _UNSET,
        freqs_ghz: ArrayLike | None | _Unset = _UNSET,
        beam_fwhm_rad: BeamFwhm | _Unset = _UNSET,
        coord: str | None | _Unset = _UNSET,
        ordering: HealpixOrdering | _Unset = _UNSET,
        seed: int | None | _Unset = _UNSET,
        lmax: int | None | _Unset = _UNSET,
    ) -> MultiFreqCompMap | MultiFreqTotalMap:
        """Sample one component map or a summed multi-component map.

        Each keyword argument falls back to the corresponding default
        captured at construction time when omitted; pass an explicit value
        (including ``None`` for nullable arguments) to override.
        """
        resolved_fields = self.fields if isinstance(fields, _Unset) else fields
        resolved_freqs = self.freqs_ghz if isinstance(freqs_ghz, _Unset) else freqs_ghz
        resolved_beam = (
            self.beam_fwhm_rad if isinstance(beam_fwhm_rad, _Unset) else beam_fwhm_rad
        )
        resolved_coord = self.coord if isinstance(coord, _Unset) else coord
        resolved_ordering = self.ordering if isinstance(ordering, _Unset) else ordering
        resolved_seed = self.seed if isinstance(seed, _Unset) else seed
        resolved_lmax = self.lmax if isinstance(lmax, _Unset) else lmax
        normalized_seed = normalize_seed(resolved_seed)

        if isinstance(comp, GaussianComponent):
            return self._sample_component(
                comp,
                fields=resolved_fields,
                ordering=resolved_ordering,
                beam_fwhm_rad=resolved_beam,
                coord=resolved_coord,
                freqs_ghz=resolved_freqs,
                seed=normalized_seed,
                lmax=resolved_lmax,
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
                    fields=resolved_fields,
                    ordering=resolved_ordering,
                    beam_fwhm_rad=resolved_beam,
                    coord=resolved_coord,
                    freqs_ghz=resolved_freqs,
                    seed=component_seeds[component.name],
                    lmax=resolved_lmax,
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
        lmax: int | None,
    ) -> MultiFreqCompMap:
        """Dispatch one component through its ``sample_map`` method."""
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
            lmax=lmax,
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
