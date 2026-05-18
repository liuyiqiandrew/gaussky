# Example: extending gaussky

`gaussky` is built around three `typing.Protocol`s
(`AngularPowerSpectrum`, `SpectralEnergyDistribution`, `GaussianComponent`).
Anything that implements the right shape plugs straight into `Sampler` — no
core code changes required. This page walks through writing a custom SED, a
custom spectrum, and a complete custom component.

## 1. A custom SED — free-free emission

Subclass `BaseSED` and implement only `_rj_scaling`. The base class wraps
that with the staged thermodynamic-CMB conversions, so you inherit `scale`,
`scale_maps`, and `scale_cls` (the three methods the SED protocol requires)
for free.

```python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from gaussky.sed.common import BaseSED


@dataclass(frozen=True)
class FreeFreeSED(BaseSED):
    """Free-free SED, fixed slope -2.13 in RJ units, normalized at nu0."""

    nu0_ghz: float = 30.0

    def _rj_scaling(self, freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
        return (freq_ghz / self.nu0_ghz) ** -2.13
```

Check that `scale(nu0) == 1`:

```python
sed = FreeFreeSED(nu0_ghz=30.0)
np.testing.assert_allclose(sed.scale(30.0), 1.0)
```

If you can't subclass `BaseSED` (e.g. you're wrapping an external SED), a
standalone class that implements `scale`, `scale_maps`, and `scale_cls`
directly still satisfies the structural protocol — see the
[`gaussky.sed`](../reference/sed.md) reference.

## 2. A custom spectrum — Planck-like template loader

```python
import numpy as np
from numpy.typing import NDArray

from gaussky.conventions import HEALPY_POLARIZED_ORDER, U_K_CMB_SQUARED


class TabulatedCl:
    """Tabulated C_ell loaded from arrays."""

    def __init__(
        self,
        *,
        ells: np.ndarray,
        cls: dict[str, np.ndarray],
        unit: str = U_K_CMB_SQUARED,
    ) -> None:
        if not np.array_equal(ells, np.arange(ells.size)):
            raise ValueError("ells must cover 0..lmax with unit spacing")
        for pair in HEALPY_POLARIZED_ORDER:
            arr = cls.get(pair, np.zeros_like(ells, dtype=np.float64))
            if arr.shape != ells.shape:
                raise ValueError(f"{pair} must match ells.shape")
        self.unit: str = unit
        self._ells = np.asarray(ells, dtype=np.int64)
        self._cls = {pair: np.asarray(cls.get(pair, np.zeros_like(self._ells)),
                                       dtype=np.float64)
                     for pair in HEALPY_POLARIZED_ORDER}

    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]:
        if lmax > int(self._ells[-1]):
            raise ValueError(f"lmax={lmax} exceeds template range")
        return [self._cls[pair][:lmax + 1].copy() for pair in HEALPY_POLARIZED_ORDER]
```

The sampling helpers will run `validate_healpy_cls` on the output, so PSD and
shape checks happen automatically.

## 3. A custom component — Gaussian AME

For an SED-backed component, subclass `BaseSEDBackedComponent`. You only
declare your model fields and implement `_build_sed` and `_metadata`; the
mixin owns name validation, SED caching, the `sed` property, and the
default `sample_map` that calls `sample_component_map(...)` for you.

```python
from collections.abc import Mapping
from dataclasses import dataclass

from gaussky.component import BaseSEDBackedComponent
from gaussky.ps import PowerLawCl

# from the previous section
# from .my_seds import FreeFreeSED


@dataclass(frozen=True, kw_only=True)
class GaussianAME(BaseSEDBackedComponent):
    """Anomalous microwave emission with a fixed-slope free-free SED."""

    ps: PowerLawCl
    nu0_ghz: float
    name: str = "ame"

    def _build_sed(self) -> FreeFreeSED:
        return FreeFreeSED(nu0_ghz=self.nu0_ghz)

    def _metadata(self) -> Mapping[str, object]:
        return {"nu0_ghz": self.nu0_ghz}
```

Drop it straight into the sampler:

```python
import numpy as np
from gaussky import GaussianCMB, Sampler
from gaussky.ps import CMBCl, PowerLawCl

ame = GaussianAME(
    ps=PowerLawCl(amp_tt=200.0, alpha_tt=-3.5),
    nu0_ghz=30.0,
)
cmb = GaussianCMB(ps=CMBCl())

sampler = Sampler(
    nside=64,
    fields=("T",),
    freqs_ghz=np.array([20.0, 30.0, 40.0, 90.0, 150.0]),
    seed=2025,
)
total = sampler.sample([cmb, ame])
total.component_names   # ('cmb', 'ame')
```

## 4. A frequency-independent custom component

If your model has no SED (the way `GaussianCMB` works), implement the
protocol directly and pass `sed=None` to the unified helper:

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
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
            seed=seed,
            lmax=lmax,
        )
```

Unlike SED-backed components, `freqs_ghz` is required here — there is no
pivot to fall back to.

## Registering as a bundled component (optional)

If you want `from gaussky.component import GaussianAME` to work after writing
the class inside the `gaussky.component` package:

1. Pick a category sub-package (e.g. `gaussky/component/ame/`) or, if you're
   adding a variant to an existing category, drop the file into that
   category's directory (`gaussky/component/dust/spatially_varying_mbb.py`,
   etc.).
2. Add the class to that sub-package's `__init__.py`.
3. In `gaussky/component/__init__.py`, add the name to `__all__` and to the
   `__getattr__` lazy-import switch.

This keeps `gaussky` lazy — concrete components are only imported when the
caller asks for them.

## Testing tips

The component test suite shows the idiomatic test setup
(`tests/test_component.py`). Two patterns worth copying:

- **Fake `healpy` for deterministic draws.** The test file patches
  `component_utils.hp` with a `FakeHealpy` whose `synfast` returns a fixed
  array. That lets you assert exact pixel values without depending on RNG
  state.
- **Tiny `nside`.** All tests run at `nside=1` (12 pixels). Component logic
  is independent of resolution, so you don't need to allocate megapixel maps
  to test it.

```python
def _patch_healpy(monkeypatch):
    fake = FakeHealpy()                # see tests/test_component.py
    monkeypatch.setattr(component_utils, "hp", fake)
    return fake
```

## Where to look in the code

- `gaussky/component/base.py` — the `GaussianComponent` protocol.
- `gaussky/component/sed_backed.py` — the `BaseSEDBackedComponent` mixin.
- `gaussky/component/component_utils.py` — the unified `sample_component_map`
  and `sample_component_alm` helpers.
- `gaussky/component/{cmb,dust,synchrotron}/*.py` — minimal worked examples
  in the bundled style; each is a sub-package with a `__init__.py` re-export
  plus one variant file.
- `gaussky/templates/` — empty placeholder for cross-cutting auxiliary-template
  loaders (e.g. published β_d maps reused by several dust variants).
