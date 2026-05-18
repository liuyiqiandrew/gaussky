# Example: synchrotron and dust foregrounds

This walkthrough samples synchrotron and dust independently, inspects their
SED scaling, and confirms the per-frequency map structure.

## Code

```python
import numpy as np

from gaussky.component import (
    SimpleModifiedBlackbodyDust,
    SimplePowerLawSynchrotron,
)
from gaussky.ps import PowerLawCl
from gaussky.sampler import Sampler

freqs_ghz = np.array([23.0, 30.0, 90.0, 150.0, 220.0, 353.0])

# Synchrotron: power-law C_ell × power-law SED. Pivot at 23 GHz.
sync = SimplePowerLawSynchrotron(
    ps=PowerLawCl(
        amp_ee=20.0, alpha_ee=-3.0,
        amp_bb=4.0,  alpha_bb=-3.0,
    ),
    beta_s=-3.1,
    nu0_ghz=23.0,
)

# Dust: power-law C_ell × modified blackbody SED. Pivot at 353 GHz.
dust = SimpleModifiedBlackbodyDust(
    ps=PowerLawCl(
        amp_ee=70.0, alpha_ee=-2.4,
        amp_bb=10.0, alpha_bb=-2.4,
    ),
    beta_d=1.6, temp_d=19.6, nu0_ghz=353.0,
)

sampler = Sampler(nside=64)

sync_map = sampler.sample(
    sync, fields=("Q", "U"), freqs_ghz=freqs_ghz,
)
dust_map = sampler.sample(
    dust, fields=("Q", "U"), freqs_ghz=freqs_ghz,
)
```

## Inspecting the SED scaling

Each component caches an SED instance derived from its parameters:

```python
sync.sed.scale(freqs_ghz)
# array([1.0, ~0.50, ~0.020, ~0.0073, ~0.0033, ~0.00128])

dust.sed.scale(freqs_ghz)
# array([~5e-6, ~9e-6, ~2e-3, ~0.018, ~0.07, 1.0])
```

The synchrotron SED is normalized at 23 GHz and falls steeply with `beta_s = -3.1`.
The dust SED is normalized at 353 GHz; foreground amplitude drops toward lower
frequencies due to the modified blackbody factor.

## How the maps are constructed

Both components produce a single pivot-frequency T/Q/U realization with
`hp.synfast` (using the spectrum given to the component) and then multiply by
their SED:

```
maps[i_freq, i_field, :] = sed.scale(freq_i) * pivot_TQU[field_i]
```

This is why the *spatial pattern* is identical across frequencies — just
scaled. Different frequency channels are *not* statistically independent;
to get independent realizations per component, call `sample` separately for
each component (which we already do here).

## Beam smoothing

Pass `beam_fwhm_rad` either as a scalar (shared) or as a per-channel array:

```python
beams = np.deg2rad([0.9, 0.7, 0.3, 0.2, 0.15, 0.1])  # one per frequency
sync_map = sampler.sample(
    sync, fields=("Q", "U"), freqs_ghz=freqs_ghz, beam_fwhm_rad=beams,
)
```

Internally, `component_utils._smooth_tqu` calls `hp.smoothing(pol=True)` once
per requested channel, so each map sees its own FWHM.

## Inspecting per-frequency slices

```python
sync_q150 = sync_map.select_freq(150.0).select_field("Q")
dust_q150 = dust_map.select_freq(150.0).select_field("Q")

sync_q150.maps.shape   # (1, 1, 49152)
dust_q150.metadata     # MappingProxyType({'beta_d': 1.6, 'temp_d': 19.6, 'nu0_ghz': 353.0})
```

`select_freq` / `select_field` return revalidated copies. They preserve
`component_name`, `metadata`, `auxiliary_maps`, and `unit`.

## Next

- Combine these into one total sky → [Total map](total-map.md).
- Add a different foreground model (e.g. anomalous microwave emission) →
  [Extending gaussky](extending.md).
