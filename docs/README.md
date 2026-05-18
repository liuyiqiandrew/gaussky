# gaussky documentation

`gaussky` is a lightweight Gaussian CMB and foreground sky simulator. It samples
multi-frequency T/Q/U HEALPix maps from analytic angular power spectra (`C_ell`)
and frequency scaling models (SEDs), then packages the realization in validated
container objects.

The package is intentionally small: three pluggable model layers behind one
high-level `Sampler` facade. New spectra, SEDs, or components are added by
implementing a `typing.Protocol` — no core code changes required.

## Where to look

### Onboarding

- [Getting started](getting-started.md) — installation, first multi-frequency
  map, and what to read next.
- [Architecture overview](architecture.md) — how `ps`, `sed`, `component`,
  `map`, and `Sampler` fit together; what happens during one `sample()` call.
- [Shared conventions](conventions.md) — the `SignalField`, `SpectrumPair`,
  `HealpixOrdering` literals, the polarized spectrum ordering, and the unit
  strings used everywhere in the API.

### Module reference

- [`gaussky.ps`](reference/ps.md) — the `AngularPowerSpectrum` protocol,
  `validate_healpy_cls`, and the bundled `PowerLawCl` / `CMBCl` models.
- [`gaussky.sed`](reference/sed.md) — the `SpectralEnergyDistribution`
  protocol, `PowerLawSED`, `ModifiedBlackbodySED`, and the
  `trj_to_tcmb` / `tcmb_to_trj` / `planck_rj_spectrum` helpers.
- [`gaussky.component`](reference/component.md) — the `GaussianComponent`
  protocol, the shared sampling helpers in `component_utils.py`, and the
  bundled `GaussianCMB`, `SimplePowerLawSynchrotron`, and
  `SimpleModifiedBlackbodyDust` components.
- [`gaussky.map`](reference/map.md) — `BaseHealpixMap`, `BaseSignalMap`,
  `AuxiliaryHealpixMap`, `MultiFreqCompMap`, and `MultiFreqTotalMap`. The
  shape and metadata contracts that all components must respect.
- [`gaussky.sampler`](reference/sampler.md) — the `Sampler` entry point
  consumers normally use.

### Examples

- [Basic CMB realization](examples/basic-cmb.md)
- [Synchrotron and dust foregrounds](examples/foregrounds.md)
- [Multi-component total sky map](examples/total-map.md)
- [Extending gaussky with a custom component](examples/extending.md)

## Quick orientation

A typical multi-frequency sky simulation reads end to end like this:

```python
import numpy as np

from gaussky import (
    GaussianCMB, Sampler,
    SimpleModifiedBlackbodyDust, SimplePowerLawSynchrotron,
)
from gaussky.ps import CMBCl, PowerLawCl

cmb = GaussianCMB(ps=CMBCl(a_lens=1.0, r_tensor=0.0))
sync = SimplePowerLawSynchrotron(
    ps=PowerLawCl(amp_ee=20.0, alpha_ee=-3.0, amp_bb=4.0, alpha_bb=-3.0),
    beta_s=-3.1,
    nu0_ghz=23.0,
)
dust = SimpleModifiedBlackbodyDust(
    ps=PowerLawCl(amp_ee=70.0, alpha_ee=-2.4, amp_bb=10.0, alpha_bb=-2.4),
    beta_d=1.6,
    temp_d=19.6,
    nu0_ghz=353.0,
)

sampler = Sampler(
    nside=128,
    fields=("T", "Q", "U"),
    freqs_ghz=np.array([30.0, 90.0, 150.0, 220.0, 353.0]),
    beam_fwhm_rad=np.deg2rad(np.array([0.5, 0.3, 0.2, 0.15, 0.1])),
    seed=2025,
)
total = sampler.sample([cmb, sync, dust])

# total.maps has shape (nfreq=5, nfield=3, npix=12*128**2) in uK_CMB
# total.metadata records the root seed and the per-component child seeds
```

The same scene with a different seed:

```python
total_v2 = sampler.with_(seed=7).sample([cmb, sync, dust])
```

Or band-limited to a custom `lmax`, or grab harmonic-space output:

```python
total_band = sampler.sample([cmb, sync, dust], lmax=64)

from gaussky.component.component_utils import sample_component_alm
alm = sample_component_alm(
    ps=cmb.ps, sed=None,
    component_name="cmb", metadata={"r_tensor": 0.0},
    lmax=128, fields=("T", "E", "B"),
    freqs_ghz=[150.0], seed=2025,
)
# alm.alms has shape (1, 3, hp.Alm.getsize(128))
```

If you just want to run the package, jump to
[Getting started](getting-started.md). If you intend to extend it, read
[Architecture](architecture.md) and the per-module reference, then
[Extending gaussky](examples/extending.md).
