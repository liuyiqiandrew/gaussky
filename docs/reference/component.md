# `gaussky.component` — Gaussian sky components

A component binds an `AngularPowerSpectrum` and (optionally) a
`SpectralEnergyDistribution` and exposes the sampling entry point that
`Sampler` calls. The package ships three concrete components, the
`BaseSEDBackedComponent` mixin every SED-backed component subclasses, and
two shared sampling helpers (map and alm).

```python
from gaussky.component import (
    GaussianCMB,                    # frequency-independent CMB
    GaussianComponent,              # protocol
    SimpleModifiedBlackbodyDust,    # dust foreground
    SimplePowerLawSynchrotron,      # synchrotron foreground
)
# or, equivalently, the flat top-level surface:
from gaussky import (
    GaussianCMB, GaussianComponent,
    SimpleModifiedBlackbodyDust, SimplePowerLawSynchrotron,
)
```

Internally the package is a directory of sub-packages, one per category:

```
gaussky/component/
├── base.py            # GaussianComponent Protocol
├── sed_backed.py      # BaseSEDBackedComponent mixin
├── component_utils.py # sample_component_map, sample_component_alm
├── cmb/               # cmb/__init__.py re-exports lensed.GaussianCMB
├── dust/              # dust/__init__.py re-exports simple_mbb.SimpleModifiedBlackbodyDust
└── synchrotron/       # synchrotron/__init__.py re-exports simple_powerlaw.SimplePowerLawSynchrotron
```

`gaussky/component/__init__.py` lazy-imports each concrete class through
`__getattr__` so the flat public surface stays cheap. Register new public
components there.

## The protocol

```python
@runtime_checkable
class GaussianComponent(Protocol):
    name: str

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
    ) -> MultiFreqCompMap: ...
```

Key facts:

- `name` is a non-empty string used in the returned `MultiFreqCompMap` and as
  the uniqueness key when several components are summed into a
  `MultiFreqTotalMap`.
- All arguments to `sample_map` are keyword-only.
- `freqs_ghz` defaults to `None`. SED-backed components fall back to their
  pivot frequency `nu0_ghz`; CMB requires a frequency grid (no pivot exists).
- `seed` is an optional NumPy legacy RNG seed. The helper save/restores the
  global RNG around `hp.synfast` so the draw is deterministic without
  leaking the seeded state.
- `lmax` defaults to `3 * nside - 1` when `None`; pass an explicit value to
  band-limit or to over-sample the realization.

## `BaseSEDBackedComponent`

The mixin (in `gaussky.component.sed_backed`) every SED-backed component
subclasses. It owns the `name` validation, the SED cache, the `sed`
property, and the default `sample_map` that delegates to
`sample_component_map(...)`.

```python
@dataclass(frozen=True, kw_only=True)
class BaseSEDBackedComponent(GaussianComponent):
    ps: AngularPowerSpectrum
    name: str
    _sed: SpectralEnergyDistribution = field(init=False, repr=False, compare=False)

    def _build_sed(self) -> SpectralEnergyDistribution: ...   # subclass override
    def _metadata(self) -> Mapping[str, object]: ...          # subclass override
```

Subclasses declare their model fields (including their own narrowly-typed
`ps` and a default for `name`) and implement `_build_sed` and `_metadata`.
Frequency-independent components (CMB-like) skip the mixin and implement
the Protocol directly. See the bundled components below for the pattern.

## `SimplePowerLawSynchrotron`

Subclass of `BaseSEDBackedComponent`. Pairs `PowerLawCl` with `PowerLawSED`.

| Field        | Default          | Meaning                                                  |
|--------------|------------------|----------------------------------------------------------|
| `ps`         | (required)       | `PowerLawCl` instance at `nu0_ghz`.                      |
| `beta_s`     | (required)       | Synchrotron spectral index (RJ units).                   |
| `nu0_ghz`    | (required)       | Pivot frequency in GHz.                                  |
| `name`       | `"synchrotron"`  | Component label.                                         |

The mixin's `__post_init__` validates the name and caches
`_sed = PowerLawSED(beta=beta_s, nu0_ghz=nu0_ghz)`. Access via
`component.sed`.

### Example

```python
import numpy as np
from gaussky.component import SimplePowerLawSynchrotron
from gaussky.ps import PowerLawCl

sync = SimplePowerLawSynchrotron(
    ps=PowerLawCl(amp_ee=20.0, alpha_ee=-3.0, amp_bb=4.0, alpha_bb=-3.0),
    beta_s=-3.1,
    nu0_ghz=23.0,
)

sampled = sync.sample_map(
    nside=64,
    freqs_ghz=np.array([23.0, 30.0, 90.0]),
    fields=("Q", "U"),
    beam_fwhm_rad=np.deg2rad(0.3),
)
sampled.maps.shape   # (3, 2, 49152)
sampled.unit         # 'uK_CMB'
sampled.metadata     # MappingProxyType({'beta_s': -3.1, 'nu0_ghz': 23.0})
```

## `SimpleModifiedBlackbodyDust`

Subclass of `BaseSEDBackedComponent`. Pairs `PowerLawCl` with `ModifiedBlackbodySED`.

| Field        | Default     | Meaning                                                       |
|--------------|-------------|---------------------------------------------------------------|
| `ps`         | (required)  | `PowerLawCl` at `nu0_ghz`.                                    |
| `beta_d`     | (required)  | Dust spectral index (RJ).                                     |
| `temp_d`     | (required)  | Dust temperature in K.                                        |
| `nu0_ghz`    | (required)  | Pivot frequency in GHz.                                       |
| `name`       | `"dust"`    | Component label.                                              |

### Example

```python
from gaussky.component import SimpleModifiedBlackbodyDust
from gaussky.ps import PowerLawCl

dust = SimpleModifiedBlackbodyDust(
    ps=PowerLawCl(amp_ee=70.0, alpha_ee=-2.4, amp_bb=10.0, alpha_bb=-2.4),
    beta_d=1.6,
    temp_d=19.6,
    nu0_ghz=353.0,
)
sampled = dust.sample_map(
    nside=64,
    freqs_ghz=[150.0, 220.0, 353.0, 545.0],
    fields=("T", "Q", "U"),
)
```

The dust SED is normalized at 353 GHz, so passing only the pivot returns the
underlying realization unscaled. Passing higher frequencies scales by the
modified blackbody.

## `GaussianCMB`

Frequency-independent component backed by `CMBCl`.

| Field        | Default                          | Meaning                                  |
|--------------|----------------------------------|------------------------------------------|
| `ps`         | `CMBCl()` (bundled, `r=0`)       | CMB `C_ell` model.                       |
| `name`       | `"cmb"`                          | Component label.                         |

Differences from SED-backed components:

- `freqs_ghz` is **required** — there is no pivot frequency to default to.
- The same T/Q/U realization is repeated across the frequency axis. The
  underlying signal is identical at every frequency.
- The `metadata` carries `a_lens`, `r_tensor`, and the template directory.

```python
from gaussky.component import GaussianCMB
from gaussky.ps import CMBCl

cmb = GaussianCMB(ps=CMBCl(r_tensor=0.03))
sampled = cmb.sample_map(
    nside=64,
    freqs_ghz=[30.0, 90.0, 150.0, 220.0],
    fields=("T", "Q", "U"),
)
sampled.metadata['r_tensor']   # 0.03
```

## What `sample_map` actually does

Every component delegates to one unified helper in
`gaussky/component/component_utils.py`:

```python
sample_component_map(
    *,
    ps: AngularPowerSpectrum,
    sed: SpectralEnergyDistribution | None,
    component_name: str,
    metadata: Mapping[str, object],
    nside: int,
    fields: tuple[SignalField, ...],
    freqs_ghz: ArrayLike | None = None,
    beam_fwhm_rad: BeamFwhm = None,
    ordering: HealpixOrdering = "RING",
    coord: str | None = None,
    seed: int | None = None,
    lmax: int | None = None,
) -> MultiFreqCompMap
```

Behaviour:

1. validate `nside` (positive power of two), `lmax` (non-negative int when
   given; `None` resolves to `3 * nside - 1`);
2. normalize `freqs_ghz` (positive 1-D array of finite floats; when `sed` is
   given and `freqs_ghz is None`, fall back to `sed.nu0_ghz`; when `sed is
   None`, `freqs_ghz` is required);
3. normalize `fields` (preserve order, reject anything outside `T, Q, U`);
4. normalize `beam_fwhm_rad` (scalar or shape `(nfreq,)`);
5. translate `ps.unit` to a signal unit via
   `gaussky.units.signal_unit_for(...)` (default registry maps
   `uK_CMB^2 → uK_CMB`);
6. call `ps.to_healpy_cls(lmax)` and run `validate_healpy_cls`;
7. call `hp.synfast(..., pol=True, new=True)` to produce a `(3, npix)`
   pivot T/Q/U map; when `seed` is given, save/restore the NumPy global
   RNG around the call;
8. apply beam smoothing with `hp.smoothing(pol=True)`:
   - **scalar beam:** smooth once, broadcast across all frequencies;
   - **per-channel beam:** loop and smooth one channel at a time;
   - **`None`:** skip smoothing;
9. when `sed` is given, apply `sed.scale_maps(maps, freqs)` (staged RJ ⇄ CMB
   pipeline); when `sed is None`, broadcast the pivot map across the
   frequency axis instead;
10. reorder RING → NESTED if `ordering="NESTED"`;
11. wrap in a `MultiFreqCompMap` with `component_name`, `metadata` (the
    helper stitches `seed` in), `auxiliary_maps={}`, and the requested
    `freqs_ghz`, `fields`, `beam`, `coord`.

A parallel helper `sample_component_alm(*, ps, sed, lmax, ...)` returns a
[`MultiFreqCompAlm`](map.md#multifreqcompalm) container — T/E/B alm
coefficients without going through `hp.alm2map`. SED-backed alms use
`sed.scale(freq)` per channel; CMB-style alms broadcast the same pivot
across the frequency axis. No beam smoothing is applied in alm space —
combine with `healpy.gauss_beam` (or a custom `bl`) downstream.

### `validate_component_name(name)`

Tiny shared validator that rejects empty strings and non-strings. Called
from the `__post_init__` of every bundled component (via the
`BaseSEDBackedComponent` mixin for SED-backed ones, directly for CMB).

## Writing your own component

For an SED-backed component, subclass `BaseSEDBackedComponent` and only
implement `_build_sed` and `_metadata`. The mixin owns name validation, the
SED cache, the `sed` property, and the `sample_map` delegation.

```python
from collections.abc import Mapping
from dataclasses import dataclass

from gaussky.component import BaseSEDBackedComponent
from gaussky.ps import PowerLawCl
from gaussky.sed import PowerLawSED


@dataclass(frozen=True, kw_only=True)
class GaussianAME(BaseSEDBackedComponent):
    """Anomalous microwave emission with a fixed-slope power-law SED."""

    ps: PowerLawCl
    nu0_ghz: float
    name: str = "ame"

    def _build_sed(self) -> PowerLawSED:
        return PowerLawSED(beta=-3.5, nu0_ghz=self.nu0_ghz)

    def _metadata(self) -> Mapping[str, object]:
        return {"nu0_ghz": self.nu0_ghz}
```

For a frequency-independent component (CMB-style), implement the protocol
directly and call `sample_component_map(sed=None, ...)`:

```python
from dataclasses import dataclass

from gaussky.component import GaussianComponent
from gaussky.component.component_utils import (
    sample_component_map,
    validate_component_name,
)
from gaussky.ps import PowerLawCl


@dataclass(frozen=True, kw_only=True)
class GaussianFrequencyFlat(GaussianComponent):
    ps: PowerLawCl
    name: str = "flat"

    def __post_init__(self) -> None:
        validate_component_name(self.name)

    def sample_map(
        self,
        *,
        nside,
        fields,
        freqs_ghz=None,
        beam_fwhm_rad=None,
        ordering="RING",
        coord=None,
        seed=None,
        lmax=None,
    ):
        return sample_component_map(
            ps=self.ps,
            sed=None,
            component_name=self.name,
            metadata={},
            nside=nside,
            fields=fields,
            freqs_ghz=freqs_ghz,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
            seed=seed,
            lmax=lmax,
        )
```

Tips:

- Reuse `sample_component_map`. It centralizes every validation, seeding,
  and ordering step.
- Use `@dataclass(frozen=True, kw_only=True)` so users get the same
  immutable, keyword-only construction as the bundled components.
- Put your component file under `gaussky/component/<category>/` (existing
  categories are `cmb/`, `dust/`, `synchrotron/`). Add the variant to the
  category's `__init__.py` and to the top-level
  `gaussky/component/__init__.py`'s `__all__` + `__getattr__` switch if you
  want it discoverable via `from gaussky.component import …`.

See [Extending gaussky](../examples/extending.md) for a runnable walkthrough.
