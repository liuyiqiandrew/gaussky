# `gaussky.map` — map containers

The `map` subpackage holds the validated, frozen dataclass containers every
sampled map flows through. There are five public types and one type alias:

```python
from gaussky.map import (
    AuxiliaryHealpixMap,           # generic HEALPix product (no nfreq/nfield)
    BaseHealpixMap,                # base for any HEALPix product
    BaseSignalMap,                 # base for multi-frequency signal maps
    BeamFwhm,                      # float | NDArray[float64] | None
    HealpixMapContainer,           # protocol (generic HEALPix)
    SignalMapContainer,            # protocol (signal map)
    MultiFreqCompMap,              # one component, multi-frequency
    MultiFreqTotalMap,             # sum of components, multi-frequency
    SUPPORTED_HEALPIX_ORDERINGS,
    SUPPORTED_SIGNAL_FIELDS,
)
```

## Invariants

All containers are:

- `@dataclass(frozen=True, kw_only=True, eq=False)`,
- store maps as **read-only `float64`** arrays (revalidated on construction),
- carry pixelization metadata (`nside`, `ordering`, `coord`),
- expose `copy_with(**changes)` that returns a revalidated new instance.

Signal maps add:

- `freqs_ghz` (positive 1-D `float64`, read-only);
- `fields` (tuple of `SignalField` literals in the requested order);
- `unit` (non-empty string);
- `beam_fwhm_rad` (scalar, per-frequency 1-D array, or `None`);
- shape contract `maps.shape == (nfreq, nfield, npix)`.

## Type alias

```python
from gaussky.map import BeamFwhm

# BeamFwhm = float | NDArray[np.float64] | None
```

A scalar beam means shared across all channels; a 1-D array supplies one beam
per frequency; `None` means no smoothing.

## `BaseHealpixMap`

Generic HEALPix container.

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class BaseHealpixMap:
    maps: NDArray[np.float64]
    nside: int
    ordering: HealpixOrdering = "RING"
    coord: str | None = None
```

Validation:

- `nside` is a positive power of two;
- `maps` is non-scalar and `maps.shape[-1] == 12 * nside ** 2`;
- `ordering` is canonicalized to uppercase (`"ring"` → `"RING"`);
- `coord` is `None` or a non-empty string.

Properties:

- `npix` — trailing axis length;
- `npix_expected` — `12 * nside ** 2`.

Methods:

- `assert_same_pixelization(other)` — raises if `nside`, `ordering`, or `coord`
  differ.
- `copy_with(**changes)` — returns a revalidated copy via
  `dataclasses.replace`; unknown field names raise `ValueError`.

## `AuxiliaryHealpixMap`

Generic auxiliary product (mask, spectral-index map, etc.). Same as
`BaseHealpixMap` plus:

- `unit: str | None = None` — optional unit string;
- `metadata: Mapping[str, object] = {}` — copied into a read-only mapping.

```python
import numpy as np
from gaussky.map import AuxiliaryHealpixMap

beta_p = AuxiliaryHealpixMap(
    maps=np.full(12, 1.6),
    nside=1,
    unit="dimensionless",
    metadata={"field": "beta_p"},
)
```

## `BaseSignalMap`

The shared base for multi-frequency signal maps. Adds frequency, field, unit,
and beam handling on top of `BaseHealpixMap`.

```python
@dataclass(frozen=True, kw_only=True, eq=False)
class BaseSignalMap(BaseHealpixMap, SignalMapContainer):
    freqs_ghz: NDArray[np.float64]
    fields: tuple[SignalField, ...]
    unit: str
    beam_fwhm_rad: BeamFwhm = None
```

Beyond `BaseHealpixMap`:

- `maps.ndim == 3` and `maps.shape == (nfreq, nfield, npix)`;
- `unit` is non-empty;
- `beam_fwhm_rad` is normalized (scalar or shape `(nfreq,)`).

Properties: `nfreq`, `nfields`.

Methods:

| Method                                          | What it does                                          |
|-------------------------------------------------|-------------------------------------------------------|
| `field_index(field)`                            | Position of `"T"`/`"Q"`/`"U"` in `fields`.            |
| `freq_index(freq_ghz, *, atol=0.0)`             | Index of a channel in `freqs_ghz`.                    |
| `select_field(field)`                           | New container with only that field.                   |
| `select_freq(freq_ghz, *, atol=0.0)`            | New container with only that frequency.               |
| `assert_compatible(other, *, rtol=0.0, atol=0.0)` | Validate matching grid / unit / fields / beam.      |

`assert_compatible` is what `MultiFreqTotalMap.from_components` uses under the
hood; everything that goes into a total map must pass it.

## `MultiFreqCompMap`

Single component, multi-frequency signal map. Returned by every
`GaussianComponent.sample_map(...)` call.

Extra fields beyond `BaseSignalMap`:

- `component_name: str` — non-empty;
- `auxiliary_maps: Mapping[str, HealpixMapContainer]` — read-only mapping
  whose values must share the same `nside`, `ordering`, and `coord` as this
  component map;
- `metadata: Mapping[str, object]` — copied into a read-only mapping.

```python
import numpy as np
from gaussky.map import AuxiliaryHealpixMap, MultiFreqCompMap

beta_p = AuxiliaryHealpixMap(
    maps=np.full(12, 1.6), nside=1, unit="dimensionless",
)

comp = MultiFreqCompMap(
    component_name="dust",
    freqs_ghz=np.array([150.0, 220.0]),
    fields=("Q", "U"),
    unit="uK_CMB",
    maps=np.zeros((2, 2, 12)),
    nside=1,
    beam_fwhm_rad=np.array([0.01, 0.005]),
    auxiliary_maps={"beta": beta_p},
    metadata={"model": "modified-blackbody"},
)
```

The auxiliary map's pixelization is validated against `comp`. The `metadata`
and `auxiliary_maps` mappings are wrapped in `MappingProxyType`, so attempting
`comp.metadata["x"] = ...` raises `TypeError`.

## `MultiFreqTotalMap`

The sum of compatible `MultiFreqCompMap` instances. Two ways to construct:

```python
total = MultiFreqTotalMap.from_components([sync, dust])  # canonical
```

```python
# Lower-level — caller must precompute the sum.
total = MultiFreqTotalMap(
    freqs_ghz=ref.freqs_ghz,
    fields=ref.fields,
    unit=ref.unit,
    maps=sync.maps + dust.maps,
    nside=ref.nside,
    beam_fwhm_rad=ref.beam_fwhm_rad,
    ordering=ref.ordering,
    coord=ref.coord,
    components=(sync, dust),
)
```

The constructor verifies:

- non-empty `components`;
- `component_name`s are unique;
- every component is compatible with the reference (frequencies, fields,
  units, beams, ordering, coord, nside);
- `maps == sum(c.maps for c in components)`.

Extra properties and methods:

- `component_names` — tuple of names in storage order;
- `component(name)` — fetch one component (`KeyError` if missing);
- `select_field(field)` — total map narrowed to one field (components too);
- `select_freq(freq_ghz, *, atol=0.0)` — total map narrowed to one frequency;
- `copy_with(components=..., ...)` — replacing `components` without supplying
  `maps` recomputes the sum.

## `HealpixMapContainer` / `SignalMapContainer` protocols

The runtime-checkable protocols mirror the concrete classes' public surface.
They exist so that:

- third-party storage objects can be passed into `auxiliary_maps` without
  inheriting `BaseHealpixMap`, as long as they expose the right attributes and
  methods;
- type checkers can treat any signal container uniformly.

In practice, most users only construct `MultiFreqCompMap` (via sampling) and
`MultiFreqTotalMap` (via `from_components`).

## Examples

### Slicing a total map

```python
field_only = total.select_field("Q")
freq_only = total.select_freq(150.0)
freq_only.maps.shape   # (1, nfields, npix)
```

### Inspecting an axis position

```python
i_T = total.field_index("T")          # 0
i_30 = total.freq_index(30.0)         # 0 (within atol=0.0)
total.maps[i_30, i_T]                 # 1-D pixel array
```

### Copying with edits

```python
relabelled = comp.copy_with(component_name="dust_main")
recolored  = comp.copy_with(unit="K_CMB")  # warning: no rescaling, just relabel
```

`copy_with` revalidates, so you cannot accidentally introduce a shape /
ordering / unit mismatch.
