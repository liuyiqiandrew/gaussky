# `gaussky.sampler` — the `Sampler` facade

`Sampler` is the user-facing entry point. It captures per-scene defaults
(HEALPix resolution, fields, frequencies, beam, ordering, coord, seed, lmax)
at construction time and dispatches each component to its `sample_map`
method, summing the results when called with a list. Per-call overrides
beat the stored defaults.

```python
class Sampler:
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
    ) -> None: ...

    @overload
    def sample(self, comp: GaussianComponent, **overrides) -> MultiFreqCompMap: ...

    @overload
    def sample(self, comp: Sequence[GaussianComponent], **overrides) -> MultiFreqTotalMap: ...

    def with_(self, **changes) -> Sampler: ...
```

The signature is overloaded so the return type follows from the input:

- a single component → `MultiFreqCompMap`;
- a sequence of components → `MultiFreqTotalMap`.

## Constructor

```python
Sampler(
    nside,
    *,                          # everything after this is keyword-only
    fields=("T", "Q", "U"),
    freqs_ghz=None,             # ArrayLike or None
    beam_fwhm_rad=0.0,          # float, NDArray, or None
    coord=None,                 # str or None
    ordering="RING",            # "RING" or "NESTED"
    seed=None,                  # int or None
    lmax=None,                  # int or None; None → 3*nside - 1 in the helper
)
```

`nside` must be a positive power of two. Validation runs the first time the
sampler hands the value to a component's `sample_map`.

Each constructor argument becomes the **default** for the matching `sample`
keyword. Pass an argument to `sample` to override the default for that call
only; pass nothing to inherit.

## `sample(...)`

```python
sample(
    comp: GaussianComponent | Sequence[GaussianComponent],
    *,
    fields=<UNSET>,
    freqs_ghz=<UNSET>,
    beam_fwhm_rad=<UNSET>,
    coord=<UNSET>,
    ordering=<UNSET>,
    seed=<UNSET>,
    lmax=<UNSET>,
) -> MultiFreqCompMap | MultiFreqTotalMap
```

Every override defaults to a private `_Unset` sentinel, **not** `None` —
that way you can explicitly pass `None` to override a non-`None` stored
default (e.g. `sampler.sample(comp, freqs_ghz=None)` falls back to the
component's pivot frequency even if the sampler carries a `freqs_ghz`
default).

### Behavior

- **Single component:** dispatches to
  `comp.sample_map(nside=self.nside, freqs_ghz=resolved_freqs,
  fields=resolved_fields, ordering=resolved_ordering,
  beam_fwhm_rad=resolved_beam, coord=resolved_coord, seed=normalized_seed,
  lmax=resolved_lmax)` and returns the result.
- **Sequence:** samples each component with the same resolved options,
  derives a child seed for each from the (optional) root seed, then calls
  `MultiFreqTotalMap.from_components(...)`. Compatibility checks are what
  `from_components` enforces (frequency grid, fields, units, beams,
  ordering, coord, nside). The total map's `metadata` records the root
  seed and the per-component child seeds.

### Validation

- `comp` must be either a `GaussianComponent` instance or a non-empty
  sequence of them. Anything else raises `TypeError`. Empty sequences raise
  `ValueError("components must contain at least one component")`.
- Each element of a sequence is rechecked with `isinstance(component,
  GaussianComponent)`; non-components raise `TypeError`.
- Component names must be unique within a sequence; duplicates raise
  `ValueError("component names must be unique")`.
- `seed` must be `None` or an `int` in `[0, 2**32 - 1]`. `bool` is rejected
  (it's almost never intended).

## `with_(**changes)`

Returns a **new** `Sampler` whose stored defaults are updated. The original
sampler is not mutated.

Recognized keys: `nside`, `fields`, `freqs_ghz`, `beam_fwhm_rad`, `coord`,
`ordering`, `seed`, `lmax`. Unknown keys raise `TypeError`.

```python
fast = Sampler(nside=64, fields=("T",), seed=42)
hi_res = fast.with_(nside=256, seed=7)
```

## Examples

### One CMB realization, ergonomic setup

```python
import numpy as np
from gaussky import GaussianCMB, Sampler

sampler = Sampler(
    nside=128,
    fields=("T", "Q", "U"),
    freqs_ghz=np.array([90.0, 150.0, 220.0]),
    beam_fwhm_rad=np.deg2rad([0.5, 0.4, 0.3]),
    coord="G",
    seed=42,
)
cmb_map = sampler.sample(GaussianCMB())
assert cmb_map.maps.shape == (3, 3, 12 * 128**2)
```

### Three-component sky → total map

```python
from gaussky import (
    GaussianCMB, Sampler,
    SimpleModifiedBlackbodyDust, SimplePowerLawSynchrotron,
)
from gaussky.ps import CMBCl, PowerLawCl

sampler = Sampler(
    nside=128,
    fields=("T", "Q", "U"),
    freqs_ghz=[30.0, 90.0, 150.0, 220.0, 353.0],
    seed=2025,
)

cmb = GaussianCMB(ps=CMBCl(r_tensor=0.0))
sync = SimplePowerLawSynchrotron(
    ps=PowerLawCl(amp_ee=20.0, alpha_ee=-3.0, amp_bb=4.0, alpha_bb=-3.0),
    beta_s=-3.1, nu0_ghz=23.0,
)
dust = SimpleModifiedBlackbodyDust(
    ps=PowerLawCl(amp_ee=70.0, alpha_ee=-2.4, amp_bb=10.0, alpha_bb=-2.4),
    beta_d=1.6, temp_d=19.6, nu0_ghz=353.0,
)

total = sampler.sample([cmb, sync, dust])
total.component_names                       # ('cmb', 'synchrotron', 'dust')
total.component('dust').maps.shape          # (5, 3, 12 * 128**2)
total.metadata                              # {'seed': 2025, 'component_seeds': {...}}
```

### Reproducible realization batches

```python
sampler = Sampler(nside=64, freqs_ghz=[150.0], seed=1)

# Re-sampling with the same seed reproduces the same map.
a = sampler.sample(GaussianCMB())
b = sampler.sample(GaussianCMB())
assert (a.maps == b.maps).all()

# A different seed → different draw.
c = sampler.with_(seed=2).sample(GaussianCMB())
assert (a.maps != c.maps).any()

# Per-call seed override without touching the default sampler.
d = sampler.sample(GaussianCMB(), seed=99)
assert (d.maps != a.maps).any() and sampler.seed == 1
```

### Band-limited realization (`lmax` override)

```python
hi = Sampler(nside=64, freqs_ghz=[150.0], fields=("T",), seed=1)
band_limited = hi.sample(GaussianCMB(), lmax=32)
full_lmax = hi.sample(GaussianCMB())  # lmax = 3*nside - 1 = 191
assert band_limited.maps.shape == full_lmax.maps.shape  # only the spectrum changed
```

## What `Sampler` deliberately does *not* do

- It does not validate `nside` itself — the helpers in `component_utils` do.
- It does not perform any sampling. All real work happens in
  `component.sample_map` and `sample_component_map(...)` in
  `gaussky/component/component_utils.py`.
- It does not memoize, cache, or batch across calls.

This keeps `Sampler` thin enough that you can replace it with your own
driver (e.g. one that scatters realizations across processes) without
touching the component pipeline.
