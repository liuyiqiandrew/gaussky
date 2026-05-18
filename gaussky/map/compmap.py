"""Component and total sky-map containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from .base import BaseSignalMap, HealpixMapContainer, SignalField
from .map_utils import normalize_metadata


def _normalize_auxiliary_maps(
    auxiliary_maps: Mapping[str, object],
    reference: HealpixMapContainer,
) -> Mapping[str, HealpixMapContainer]:
    """Validate auxiliary map containers keyed by product name."""
    normalized: dict[str, HealpixMapContainer] = {}
    for name, auxiliary_map in auxiliary_maps.items():
        if not isinstance(name, str):
            raise TypeError("auxiliary map names must be strings")
        if name == "":
            raise ValueError("auxiliary map names must not be empty")
        if not isinstance(auxiliary_map, HealpixMapContainer):
            raise TypeError(
                "auxiliary map values must be HealpixMapContainer instances"
            )

        reference.assert_same_pixelization(auxiliary_map)
        normalized[name] = auxiliary_map
    return MappingProxyType(normalized)


def _sum_component_maps(
    components: Sequence["MultiFreqCompMap"],
) -> NDArray[np.float64]:
    """Return the component sum using the shared map-axis convention."""
    return np.sum(
        [component.maps for component in components], axis=0, dtype=np.float64
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class MultiFreqCompMap(BaseSignalMap):
    """Map realization for one named sky component.

    Parameters
    ----------
    component_name : str
        Stable component label, for example ``"cmb"`` or ``"dust"``.
    auxiliary_maps : mapping, default={}
        Component-owned auxiliary products. Values must be HEALPix map
        containers with the same pixelization as this component map.
    metadata : mapping, default={}
        Read-only metadata for provenance or model parameters.

    Notes
    -----
    The inherited ``maps`` array has shape ``(nfreq, nfield, npix)``.
    """

    component_name: str
    auxiliary_maps: Mapping[str, HealpixMapContainer] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate component-specific metadata after base map validation."""
        super().__post_init__()
        if not isinstance(self.component_name, str):
            raise TypeError("component_name must be a string")
        if self.component_name == "":
            raise ValueError("component_name must not be empty")

        object.__setattr__(
            self,
            "auxiliary_maps",
            _normalize_auxiliary_maps(self.auxiliary_maps, self),
        )
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))


@dataclass(frozen=True, kw_only=True, eq=False)
class MultiFreqTotalMap(BaseSignalMap):
    """Map realization made by summing compatible component maps.

    Parameters
    ----------
    components : sequence of MultiFreqCompMap
        Component maps whose sum defines ``maps``. Components must share the
        same frequency grid, fields, HEALPix metadata, unit, beam, and
        coordinate frame.

    Raises
    ------
    ValueError
        If components are empty, duplicated by name, incompatible, or do not sum
        to the provided total map.
    """

    components: tuple[MultiFreqCompMap, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate component compatibility and total-map consistency."""
        super().__post_init__()
        components = tuple(self.components)
        if not components:
            raise ValueError("components must contain at least one component map")

        names = [component.component_name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")

        for component in components:
            self.assert_compatible(component)

        expected_maps = _sum_component_maps(components)
        if expected_maps.shape != self.maps.shape or not np.array_equal(
            expected_maps, self.maps
        ):
            raise ValueError("maps must equal the sum of component maps")

        object.__setattr__(self, "components", components)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    @classmethod
    def from_components(
        cls,
        components: Sequence[MultiFreqCompMap],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Self:
        """Build a total map from compatible component maps.

        Parameters
        ----------
        components : sequence of MultiFreqCompMap
            Component maps to sum. Names must be unique and metadata must be
            compatible for arithmetic.
        metadata : mapping or None, default=None
            Optional metadata stored on the total map.

        Returns
        -------
        MultiFreqTotalMap
            Total map whose ``maps`` array is the component sum.
        """
        component_tuple = tuple(components)
        if not component_tuple:
            raise ValueError("components must contain at least one component map")

        reference = component_tuple[0]
        names = [component.component_name for component in component_tuple]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")

        for component in component_tuple[1:]:
            reference.assert_compatible(component)

        return cls(
            freqs_ghz=reference.freqs_ghz,
            fields=reference.fields,
            unit=reference.unit,
            maps=_sum_component_maps(component_tuple),
            nside=reference.nside,
            beam_fwhm_rad=reference.beam_fwhm_rad,
            ordering=reference.ordering,
            coord=reference.coord,
            components=component_tuple,
            metadata={} if metadata is None else metadata,
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Names of the components that form this total map."""
        return tuple(component.component_name for component in self.components)

    def component(self, name: str) -> MultiFreqCompMap:
        """Return a component by name.

        Parameters
        ----------
        name : str
            Component name to locate.

        Returns
        -------
        MultiFreqCompMap
            Matching component map.

        Raises
        ------
        KeyError
            If no component has the requested name.
        """
        for component in self.components:
            if component.component_name == name:
                return component
        raise KeyError(name)

    def select_field(self, field: SignalField) -> Self:
        """Return a total map with one retained field and matching components."""
        return type(self).from_components(
            tuple(component.select_field(field) for component in self.components),
            metadata=self.metadata,
        )

    def select_freq(self, freq_ghz: float, *, atol: float = 0.0) -> Self:
        """Return a total map with one retained frequency and matching components."""
        return type(self).from_components(
            tuple(
                component.select_freq(freq_ghz, atol=atol)
                for component in self.components
            ),
            metadata=self.metadata,
        )

    def copy_with(self, **changes: object) -> Self:
        """Return a revalidated total map copy.

        If ``components`` are replaced and ``maps`` are omitted, the total is
        recomputed from the new component sequence.
        """
        if "components" in changes and "maps" not in changes:
            remaining_changes = dict(changes)
            components = remaining_changes.pop("components")
            if isinstance(components, Sequence):
                metadata = remaining_changes.pop("metadata", self.metadata)
                total = type(self).from_components(components, metadata=metadata)
                if remaining_changes:
                    return total.copy_with(**remaining_changes)
                return total
            raise TypeError("components must be a sequence of component maps")
        return super().copy_with(**changes)
