# Example: extending gaussky

`gaussky` is built around three `typing.Protocol`s
(`AngularPowerSpectrum`, `SpectralEnergyDistribution`, `GaussianComponent`).
Anything that implements the right shape plugs straight into `Sampler` — no
core code changes required. This page walks through writing a custom SED, a
custom spectrum, and a complete custom component.

## 1. A custom SED — free-free emission

```python
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gaussky.sed import tcmb_to_trj, trj_to_tcmb


class FreeFreeSED:
    """Free-free SED, fixed slope -2.13 in RJ units, normalized at nu0."""

    def __init__(self, nu0_ghz: float) -> None:
        if not (nu0_ghz > 0):
            raise ValueError("nu0_ghz must be > 0")
        self.nu0_ghz: float = float(nu0_ghz)

    def scale(self, freq_ghz: ArrayLike) -> NDArray[np.float64]:
        freq = np.asarray(freq_ghz, dtype=np.float64)
        if np.any(~np.isfinite(freq)) or np.any(freq <= 0.0):
            raise ValueError("freq_ghz must contain finite positive values")
        rj = (freq / self.nu0_ghz) ** -2.13
        return trj_to_tcmb(freq) * tcmb_to_trj(self.nu0_ghz) * rj
```

Check that `scale(nu0) == 1`:

```python
sed = FreeFreeSED(nu0_ghz=30.0)
np.testing.assert_allclose(sed.scale(30.0), 1.0)
```

That's enough for the SED to be accepted anywhere a
`SpectralEnergyDistribution` is expected — the protocol is structural.

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

A component glues a spectrum and an SED behind the
`GaussianComponent` protocol. Use the helpers in
`gaussky.component.component_utils` to avoid reimplementing validation and
ordering.

```python
from dataclasses import dataclass, field

from gaussky.component import GaussianComponent
from gaussky.component.component_utils import (
    sample_gaussian_component_map,
    validate_component_name,
)
from gaussky.conventions import HealpixOrdering, SignalField
from gaussky.map import BeamFwhm, MultiFreqCompMap
from gaussky.ps import PowerLawCl


@dataclass(frozen=True, kw_only=True)
class GaussianAME(GaussianComponent):
    """Anomalous microwave emission with a fixed-slope free-free SED."""

    ps: PowerLawCl
    nu0_ghz: float
    name: str = "ame"
    _sed: FreeFreeSED = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_component_name(self.name)
        object.__setattr__(self, "_sed", FreeFreeSED(nu0_ghz=self.nu0_ghz))

    @property
    def sed(self) -> FreeFreeSED:
        return self._sed

    def sample_map(
        self,
        *,
        nside: int,
        fields: tuple[SignalField, ...],
        freqs_ghz=None,
        beam_fwhm_rad: BeamFwhm = None,
        ordering: HealpixOrdering = "RING",
        coord: str | None = None,
    ) -> MultiFreqCompMap:
        return sample_gaussian_component_map(
            ps=self.ps,
            sed=self.sed,
            component_name=self.name,
            metadata={"nu0_ghz": self.nu0_ghz},
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
        )
```

Drop it straight into the sampler:

```python
import numpy as np
from gaussky.component import GaussianCMB
from gaussky.ps import CMBCl, PowerLawCl
from gaussky.sampler import Sampler

ame = GaussianAME(
    ps=PowerLawCl(amp_tt=200.0, alpha_tt=-3.5),
    nu0_ghz=30.0,
)
cmb = GaussianCMB(ps=CMBCl())

sampler = Sampler(nside=64)
total = sampler.sample(
    [cmb, ame],
    fields=("T",),
    freqs_ghz=np.array([20.0, 30.0, 40.0, 90.0, 150.0]),
)
total.component_names   # ('cmb', 'ame')
```

## 4. A frequency-independent custom component

If your model has no SED (the way `GaussianCMB` works), use the other helper:

```python
from gaussky.component.component_utils import (
    sample_frequency_independent_gaussian_component_map,
)

@dataclass(frozen=True, kw_only=True)
class GaussianFrequencyFlat(GaussianComponent):
    ps: PowerLawCl
    name: str = "flat"

    def __post_init__(self) -> None:
        validate_component_name(self.name)

    def sample_map(self, *, nside, fields, freqs_ghz=None,
                   beam_fwhm_rad=None, ordering="RING", coord=None):
        return sample_frequency_independent_gaussian_component_map(
            ps=self.ps,
            component_name=self.name,
            metadata={},
            nside=nside,
            freqs_ghz=freqs_ghz,
            fields=fields,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
        )
```

Unlike SED-backed components, `freqs_ghz` is required here — there is no pivot
to fall back to.

## Registering as a bundled component (optional)

If you want `from gaussky.component import GaussianAME` to work after writing
the class inside the `gaussky.component` package:

1. Drop the file under `gaussky/component/ame.py`.
2. In `gaussky/component/__init__.py`, add the name to `__all__` and to the
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
- `gaussky/component/component_utils.py` — the two sampling helpers.
- `gaussky/component/dust.py`, `synchrotron.py`, `cmb.py` — minimal worked
  examples in the bundled style.
