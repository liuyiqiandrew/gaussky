# Example: multi-component total sky map

This walkthrough sums CMB, synchrotron, and dust into a single
`MultiFreqTotalMap` and shows how to navigate it.

## Code

```python
import numpy as np

from gaussky.component import (
    GaussianCMB,
    SimpleModifiedBlackbodyDust,
    SimplePowerLawSynchrotron,
)
from gaussky.ps import CMBCl, PowerLawCl
from gaussky.sampler import Sampler

freqs_ghz = np.array([30.0, 90.0, 150.0, 220.0, 353.0])
beams_rad = np.deg2rad([0.9, 0.3, 0.2, 0.15, 0.1])

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
    fields=("T", "Q", "U"),
    freqs_ghz=freqs_ghz,
    beam_fwhm_rad=beams_rad,
    coord="G",
)
```

`Sampler.sample` calls each component once, then
`MultiFreqTotalMap.from_components(...)` runs compatibility checks and adds
the maps.

## Inspecting the total

```python
type(total).__name__         # 'MultiFreqTotalMap'
total.maps.shape             # (5, 3, 12 * 128**2)  = (nfreq, nfield, npix)
total.unit                   # 'uK_CMB'
total.component_names        # ('cmb', 'synchrotron', 'dust')
```

Retrieve a single component:

```python
total.component('cmb').maps.shape   # (5, 3, 12 * 128**2)
total.component('dust').metadata['beta_d']    # 1.6
```

## Compatibility checks

Components must share:

- `nside`, `ordering`, `coord`;
- `freqs_ghz` (exact-equal under `np.allclose(rtol=0, atol=0)` by default);
- `fields` tuple (same order!);
- `unit`;
- `beam_fwhm_rad` (same shape and values).

`Sampler` enforces this implicitly because it passes the same options to every
component. If you construct components separately and call
`MultiFreqTotalMap.from_components` yourself, mismatches raise
`ValueError` with a precise message such as
`"frequency grids are incompatible"`,
`"map fields are incompatible"`,
`"beam FWHM values are incompatible"`,
or `"component names must be unique"`.

## Slicing the total

`select_field` and `select_freq` recurse into the stored components and return
a smaller total map whose components are sliced consistently:

```python
total_Q = total.select_field("Q")
total_Q.maps.shape                            # (5, 1, 196608)
total_Q.components[0].fields                  # ('Q',)

total_150 = total.select_freq(150.0)
total_150.maps.shape                          # (1, 3, 196608)
total_150.components[1].freqs_ghz             # [150.0]
```

## Recomputing the sum on copy

`copy_with(components=...)` without `maps=...` recomputes the sum from the
new component list:

```python
without_dust = total.copy_with(components=(total.component('cmb'),
                                          total.component('synchrotron')))
without_dust.component_names    # ('cmb', 'synchrotron')
```

## Why use a total map?

- Single shape `(nfreq, nfield, npix)` makes downstream code uniform.
- Component breakdown is preserved, so you can write component-separation /
  ILC code that compares `total.maps` to `total.component('cmb').maps`.
- All metadata (`unit`, `beam_fwhm_rad`, `coord`, `metadata`) is consistent
  across the components by construction.

If you only need the sum and not the per-component breakdown, you can also
construct a bare `MultiFreqTotalMap` with `components=(...)` and a precomputed
`maps`, but `from_components` is almost always what you want.
