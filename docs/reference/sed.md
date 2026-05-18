# `gaussky.sed` — spectral energy distributions

An SED in `gaussky` is a tiny three-method protocol: a per-frequency scalar
factor (`scale`) plus map and spectrum scaling entry points
(`scale_maps`, `scale_cls`) that match the operation order of the legacy
`pygsm` pipeline.

```python
class SpectralEnergyDistribution(Protocol):
    nu0_ghz: float
    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]: ...
    def scale_maps(self, maps: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]: ...
    def scale_cls(self, cls: ArrayLike, freq_ghz: ArrayLike) -> NDArray[np.float64]: ...
```

The sampling helper in `gaussky.component.component_utils` calls
`sed.scale_maps(maps, freqs)` (not `sed.scale`), so the staged
thermodynamic-CMB → RJ → SED → thermodynamic-CMB conversion is applied in
the same order as `pygsm.Sky`. For analytic per-frequency factors and the
alm path, `sed.scale(freq)` is the convenience entry.

Conventions:

- All frequencies are in GHz.
- All return values are in thermodynamic CMB temperature units, so a foreground
  component sampled at the pivot frequency keeps its `uK_CMB` amplitude.
- `scale(nu0_ghz) == 1.0` is a hard contract.

```python
from gaussky.sed import (
    ModifiedBlackbodySED,
    PowerLawSED,
    SpectralEnergyDistribution,
    planck_rj_spectrum,
    tcmb_to_trj,
    trj_to_tcmb,
)
from gaussky.sed.common import BaseSED  # shared base for staged-scaling SEDs
```

## `BaseSED`

`BaseSED` (in `gaussky.sed.common`) is the abstract base the bundled SEDs
subclass. It implements `scale`, `scale_maps`, and `scale_cls` in terms of
one abstract hook — `_rj_scaling(freq_ghz)` — that returns the dimensionless
SED factor in Rayleigh-Jeans temperature units.

```python
class BaseSED(ABC):
    nu0_ghz: float
    @abstractmethod
    def _rj_scaling(self, freq_ghz): ...
    # scale, scale_maps, scale_cls inherited
```

Subclasses declare their model parameters as dataclass fields and implement
`_rj_scaling`; they get the three Protocol methods for free.

## `PowerLawSED`

Power-law SED in Rayleigh-Jeans temperature units, wrapped with the RJ⇄CMB
conversion so that the output factor is dimensionless and equals 1 at `nu0`.

| Field      | Validation                            |
|------------|---------------------------------------|
| `beta`     | finite                                |
| `nu0_ghz`  | strictly positive                     |

Formula (callers do not need to invoke this directly):

```
scale(nu) = (trj_to_tcmb(nu) * tcmb_to_trj(nu0)) * (nu / nu0) ** beta
```

```python
from gaussky.sed import PowerLawSED

sync = PowerLawSED(beta=-3.0, nu0_ghz=30.0)
sync.scale(30.0)         # 1.0
sync.scale([23.0, 90.0]) # broadcasts; returns shape (2,)
```

## `ModifiedBlackbodySED`

Modified blackbody for dust. Wraps the spectral index, greybody factor, and
RJ⇄CMB conversion.

| Field          | Validation                          |
|----------------|-------------------------------------|
| `beta`         | finite                              |
| `temperature_k`| strictly positive                   |
| `nu0_ghz`      | strictly positive                   |

```python
from gaussky.sed import ModifiedBlackbodySED

dust = ModifiedBlackbodySED(beta=1.6, temperature_k=19.6, nu0_ghz=353.0)
dust.scale(353.0)        # 1.0
dust.scale([150.0, 220.0, 545.0])
```

The greybody is computed using `planck_rj_spectrum` so that low-GHz numerics
stay stable (uses `np.expm1`).

## Unit-conversion helpers

`gaussky.sed.sed_utils` provides three utilities used internally and exposed
because they are useful in user code:

```python
planck_rj_spectrum(temperature_k: float, freq_ghz: ArrayLike) -> NDArray[np.float64]
trj_to_tcmb(freq_ghz: ArrayLike) -> NDArray[np.float64]
tcmb_to_trj(freq_ghz: ArrayLike) -> NDArray[np.float64]
```

### `planck_rj_spectrum`

The frequency-dependent part of the Planck spectrum *without* the `nu**2` flux
prefactor — i.e. the factor needed for modified blackbody SED scaling in RJ
temperature units. Validates that the temperature is strictly positive and all
frequencies are strictly positive.

### `trj_to_tcmb` / `tcmb_to_trj`

Conversion factors between Rayleigh-Jeans brightness temperature and CMB
thermodynamic temperature. They are exact inverses of each other:

```python
import numpy as np
from gaussky.sed import trj_to_tcmb, tcmb_to_trj

freq = np.array([23.0, 93.0, 150.0, 353.0])
np.testing.assert_allclose(trj_to_tcmb(freq) * tcmb_to_trj(freq), 1.0)
```

Internally both use `np.expm1` for low-frequency stability and assume
`T_CMB = 2.725 K`.

## Writing your own SED

The recommended path is subclassing `BaseSED` and implementing one method —
`_rj_scaling` — that returns the dimensionless factor in RJ units. The base
class wraps that with the staged thermodynamic-CMB conversions so the
result is consistent with `pygsm` and compatible with
`sample_component_map`'s `sed.scale_maps(...)` call.

```python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gaussky.sed.common import BaseSED


@dataclass(frozen=True)
class FreeFreeSED(BaseSED):
    """Free-free emission SED with a fixed slope of -2.13 in RJ units."""

    nu0_ghz: float = 30.0

    def _rj_scaling(self, freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
        return (freq_ghz / self.nu0_ghz) ** -2.13
```

That's it — `scale`, `scale_maps`, and `scale_cls` are inherited and satisfy
the Protocol. Sanity check:

```python
import numpy as np

sed = FreeFreeSED(nu0_ghz=30.0)
np.testing.assert_allclose(sed.scale(30.0), 1.0)
```

If you cannot subclass `BaseSED` (e.g. you're wrapping an external SED), a
standalone class that implements `scale`, `scale_maps`, and `scale_cls`
directly still satisfies the Protocol — it's structural. Use
`gaussky.sed.tcmb_to_trj`/`trj_to_tcmb` to keep the RJ ⇄ CMB wrapping
consistent.

Drop it into a new `GaussianComponent` — see [Extending gaussky](../examples/extending.md).

## Why the RJ⇄CMB wrapping?

Foreground SEDs are most naturally expressed in Rayleigh-Jeans units (the
spectral index lives in RJ space). Sampled maps are stored in `uK_CMB`. The
helpers convert RJ → CMB at the target frequency and back at the reference
frequency, so that:

- `sed.scale(nu0) == 1` always.
- multiplying a `uK_CMB` pivot map by `sed.scale(nu)` gives a `uK_CMB` map at
  `nu`.

This is the same convention the bundled `PowerLawSED` and `ModifiedBlackbodySED`
use. Stick to it in custom SEDs so that they compose cleanly with the rest of
the pipeline.
