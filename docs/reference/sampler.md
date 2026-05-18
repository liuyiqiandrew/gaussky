# `gaussky.sampler` — the `Sampler` facade

`Sampler` is a thin entry point that lets a caller share one HEALPix
resolution and one set of map options across one or more components. It owns
`nside`; everything else (fields, frequencies, beam, ordering, coordinate
frame) is supplied per call.

```python
class Sampler:
    def __init__(self, nside: int) -> None: ...

    @overload
    def sample(self, comp: GaussianComponent, ...) -> MultiFreqCompMap: ...

    @overload
    def sample(self, comp: Sequence[GaussianComponent], ...) -> MultiFreqTotalMap: ...
```

The signature is overloaded so that mypy can prove the return type from the
input:

- a single component → `MultiFreqCompMap`;
- a sequence of components → `MultiFreqTotalMap`.

## Constructor

```python
Sampler(nside: int)
```

`nside` must be a positive power of two — the same constraint enforced by the
helpers in `component_utils`. The constructor itself does not validate it;
validation happens when `sample(...)` calls `component.sample_map(nside=...)`.

## `sample(...)`

```python
sample(
    comp: GaussianComponent | Sequence[GaussianComponent],
    fields: tuple[SignalField, ...] = ("T", "Q", "U"),
    ordering: HealpixOrdering = "RING",
    beam_fwhm_rad: BeamFwhm = 0.0,
    coord: str | None = None,
    freqs_ghz: ArrayLike | None = None,
) -> MultiFreqCompMap | MultiFreqTotalMap
```

### Behavior

- **Single component:** dispatches to `comp.sample_map(nside=self.nside,
  freqs_ghz=freqs_ghz, fields=fields, ordering=ordering,
  beam_fwhm_rad=beam_fwhm_rad, coord=coord)` and returns the result.
- **Sequence:** samples each component with the same arguments, then calls
  `MultiFreqTotalMap.from_components(...)`. Components must produce signal
  maps with compatible frequency grids, fields, units, beams, ordering, and
  coordinate frames (this is what `from_components` checks).

### Validation

- `comp` must be either a `GaussianComponent` instance or a sequence of them.
  Anything else raises `TypeError`.
- An empty sequence raises `ValueError("components must contain at least one component")`.
- Each element of a sequence is rechecked with `isinstance(component, GaussianComponent)`;
  a non-component raises `TypeError`.

## Examples

### One CMB realization

```python
import numpy as np
from gaussky.component import GaussianCMB
from gaussky.sampler import Sampler

sampler = Sampler(nside=128)
cmb_map = sampler.sample(
    GaussianCMB(),
    freqs_ghz=np.array([90.0, 150.0, 220.0]),
    fields=("T", "Q", "U"),
    beam_fwhm_rad=np.deg2rad([0.5, 0.4, 0.3]),
    coord="G",
)
isinstance(cmb_map.maps, np.ndarray)
```

### Three-component sky → total map

```python
from gaussky.component import (
    GaussianCMB, SimpleModifiedBlackbodyDust, SimplePowerLawSynchrotron,
)
from gaussky.ps import CMBCl, PowerLawCl
from gaussky.sampler import Sampler

cmb = GaussianCMB(ps=CMBCl(r_tensor=0.0))
sync = SimplePowerLawSynchrotron(
    ps=PowerLawCl(amp_ee=20.0, alpha_ee=-3.0, amp_bb=4.0, alpha_bb=-3.0),
    beta_s=-3.1, nu0_ghz=23.0,
)
dust = SimpleModifiedBlackbodyDust(
    ps=PowerLawCl(amp_ee=70.0, alpha_ee=-2.4, amp_bb=10.0, alpha_bb=-2.4),
    beta_d=1.6, temp_d=19.6, nu0_ghz=353.0,
)

sampler = Sampler(nside=128)
total = sampler.sample(
    [cmb, sync, dust],
    freqs_ghz=[30.0, 90.0, 150.0, 220.0, 353.0],
    fields=("T", "Q", "U"),
)
total.component_names         # ('cmb', 'synchrotron', 'dust')
total.component('dust').maps.shape   # (5, 3, 12 * 128**2)
```

### Reusing the same Sampler across realizations

```python
import numpy as np

sampler = Sampler(nside=64)
realizations = [
    sampler.sample(cmb, freqs_ghz=np.array([90.0]), fields=("T",))
    for _ in range(8)
]
# Each call produces an independent Gaussian draw via hp.synfast.
```

If you want repeatable draws, seed NumPy's global RNG before calling
`sample`; `hp.synfast` uses NumPy's default generator.

## What `Sampler` deliberately does *not* do

- It does not validate `nside` itself — the helpers in `component_utils` do.
- It does not perform any sampling. All real work happens in
  `component.sample_map` and the helpers in `gaussky/component/component_utils.py`.
- It does not memoize, cache, or batch across calls.

This keeps `Sampler` thin enough that you can replace it with your own driver
(e.g. one that scatters realizations across processes) without touching the
component pipeline.
