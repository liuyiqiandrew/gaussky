# `gaussky.component` — Gaussian sky components

A component binds an `AngularPowerSpectrum` and (optionally) an
`SpectralEnergyDistribution` and exposes the sampling entry point that
`Sampler` calls. The package ships three concrete components and one shared
sampling pipeline.

```python
from gaussky.component import (
    GaussianCMB,                    # frequency-independent CMB
    GaussianComponent,              # protocol
    SimpleModifiedBlackbodyDust,    # dust foreground
    SimplePowerLawSynchrotron,      # synchrotron foreground
)
```

The package lazy-imports the concrete classes through `__getattr__`, so
registering a new bundled component means adding it to the lookup in
`gaussky/component/__init__.py`.

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
    ) -> MultiFreqCompMap: ...
```

Key facts:

- `name` is a non-empty string used in the returned `MultiFreqCompMap` and as
  the uniqueness key when several components are summed into a
  `MultiFreqTotalMap`.
- All arguments to `sample_map` are keyword-only.
- `freqs_ghz` defaults to `None`. SED-backed components fall back to their
  pivot frequency `nu0_ghz`; CMB requires a frequency grid (no pivot exists).

## `SimplePowerLawSynchrotron`

Frozen dataclass: `PowerLawCl` × `PowerLawSED`.

| Field        | Default          | Meaning                                                  |
|--------------|------------------|----------------------------------------------------------|
| `ps`         | (required)       | `PowerLawCl` instance at `nu0_ghz`.                      |
| `beta_s`     | (required)       | Synchrotron spectral index (RJ units).                   |
| `nu0_ghz`    | (required)       | Pivot frequency in GHz.                                  |
| `name`       | `"synchrotron"`  | Component label.                                         |

The internal SED `_sed = PowerLawSED(beta=beta_s, nu0_ghz=nu0_ghz)` is cached
in `__post_init__`. Access via `component.sed`.

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

Frozen dataclass: `PowerLawCl` × `ModifiedBlackbodySED`.

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

All three components delegate to one of two helpers in
`gaussky/component/component_utils.py`:

| Helper                                                       | Used by                              |
|--------------------------------------------------------------|--------------------------------------|
| `sample_gaussian_component_map`                              | synchrotron, dust (SED-backed)       |
| `sample_frequency_independent_gaussian_component_map`        | CMB (no SED)                         |

Both helpers:

1. validate `nside` (positive power of two);
2. normalize `freqs_ghz` (positive 1-D array of finite floats; SED-backed
   helpers fall back to `sed.nu0_ghz` if `None`);
3. normalize `fields` (preserve order, reject anything outside `T, Q, U`);
4. normalize `beam_fwhm_rad` (scalar or shape `(nfreq,)`);
5. translate `ps.unit` to a signal unit (currently `uK_CMB^2` → `uK_CMB`);
6. compute `lmax = 3 * nside - 1`;
7. call `ps.to_healpy_cls(lmax)` and run `validate_healpy_cls`;
8. call `hp.synfast(..., pol=True, new=True)` to produce a `(3, npix)` pivot
   T/Q/U map;
9. apply beam smoothing with `hp.smoothing(pol=True)`:
   - **scalar beam:** smooth once, broadcast across all frequencies;
   - **per-channel beam:** loop and smooth one channel at a time;
   - **`None`:** skip smoothing;
10. for SED-backed components, multiply by `sed.scale(freqs)[:, None, None]`;
    for CMB, skip;
11. reorder RING→NESTED if `ordering="NESTED"`;
12. wrap in a `MultiFreqCompMap` with `component_name`, `metadata`,
    `auxiliary_maps={}`, and the requested `freqs_ghz`, `fields`, `beam`,
    `coord`.

### `validate_component_name(name)`

Tiny shared validator that rejects empty strings and non-strings. Called from
the `__post_init__` of every bundled component.

## Writing your own component

The minimum looks like this:

```python
from dataclasses import dataclass
from gaussky.component import GaussianComponent
from gaussky.component.component_utils import (
    sample_gaussian_component_map,
    validate_component_name,
)
from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import PowerLawCl
from gaussky.sed import PowerLawSED

@dataclass(frozen=True, kw_only=True)
class MyAme(GaussianComponent):
    ps: PowerLawCl
    nu0_ghz: float
    name: str = "ame"

    def __post_init__(self) -> None:
        validate_component_name(self.name)

    @property
    def sed(self) -> PowerLawSED:
        # Replace with the SED that fits your model.
        return PowerLawSED(beta=-3.5, nu0_ghz=self.nu0_ghz)

    def sample_map(self, *, nside, fields, freqs_ghz=None,
                   beam_fwhm_rad=None, ordering="RING", coord=None):
        return sample_gaussian_component_map(
            ps=self.ps,
            sed=self.sed,
            component_name=self.name,
            metadata={"nu0_ghz": self.nu0_ghz},
            nside=nside, fields=fields, freqs_ghz=freqs_ghz,
            beam_fwhm_rad=beam_fwhm_rad, ordering=ordering, coord=coord,
        )
```

Tips:

- Reuse `sample_gaussian_component_map` or
  `sample_frequency_independent_gaussian_component_map`. They centralize every
  validation and ordering step.
- Use `@dataclass(frozen=True, kw_only=True)` so users get the same
  immutable, keyword-only construction as the bundled components.
- Cache derived objects (like an internal SED) using
  `object.__setattr__(self, "_field_name", value)` inside `__post_init__`, the
  same pattern used by `SimplePowerLawSynchrotron`.
- Add the class to `gaussky/component/__init__.py`'s `__all__` and the
  `__getattr__` lazy-import switch if you want it to be discoverable via
  `from gaussky.component import MyAme`.

See [Extending gaussky](../examples/extending.md) for a runnable walkthrough.
