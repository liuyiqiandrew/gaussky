# Getting started

This page covers installation, package layout at a glance, and the smallest
end-to-end example that produces a multi-frequency sky map.

## Installation

`gaussky` requires Python ≥ 3.11 and depends only on `numpy` and `healpy` at
runtime.

### Editable install

```bash
pip install -e ".[dev]"
```

`[dev]` adds the developer extras (`black`, `mypy`, `pytest`, `ruff`).

### Conda environment

A reference environment is checked in at the project root:

```bash
conda env create -f environment.yml
conda activate gaussky
```

### Verify the install

```bash
pytest
```

The full suite uses small `nside`/`lmax` values and runs in a few seconds. The
component tests stub out `healpy` with a deterministic fake, so they do not
require a working FFT backend.

## Package layout

```
gaussky/
├── conventions.py      ← shared types, units, and orderings
├── sampler.py          ← high-level entry point
├── ps/                 ← angular power spectra (C_ell models)
├── sed/                ← spectral energy distributions
├── component/          ← Gaussian sky components (ps + sed binders)
├── map/                ← validated map containers
└── data/cmb_spec/      ← bundled CAMB CMB templates (D_ell)
```

Three protocols define what plugs into the sampler:

| Layer        | Protocol                          | Job                                    |
|--------------|-----------------------------------|----------------------------------------|
| `ps`         | `AngularPowerSpectrum`            | Provide `C_ell` in Healpy ordering.    |
| `sed`        | `SpectralEnergyDistribution`      | Frequency scaling, normalized at `nu0`.|
| `component`  | `GaussianComponent`               | Bind a `ps` + (optional) `sed`, sample.|

The `sampler.Sampler` is just a thin facade that calls `component.sample_map`.

## Hello, sky!

The smallest useful program: sample one CMB realization with the bundled
template at three frequencies.

```python
import numpy as np

from gaussky.component import GaussianCMB
from gaussky.sampler import Sampler

cmb = GaussianCMB()  # default a_lens=1.0, r_tensor=0.0, bundled templates
sampler = Sampler(nside=64)

sky = sampler.sample(
    cmb,
    fields=("T", "Q", "U"),
    freqs_ghz=np.array([30.0, 90.0, 150.0]),
)

print(sky.maps.shape)        # (3, 3, 49152)  -- (nfreq, nfield, npix)
print(sky.unit)              # 'uK_CMB'
print(sky.component_name)    # 'cmb'
print(sky.metadata)          # MappingProxyType({'a_lens': 1.0, 'r_tensor': 0.0, ...})
```

The CMB component samples the spectrum once and replicates the same realization
across the requested frequency channels (CMB has no SED). Per-frequency beam
smoothing is still supported — see [Components](reference/component.md).

## Next steps

- [Architecture overview](architecture.md) — what happens inside that
  `sampler.sample(...)` call.
- [Examples / Foregrounds](examples/foregrounds.md) — add power-law synchrotron
  and modified-blackbody dust to the same scene.
- [Examples / Total map](examples/total-map.md) — build a `MultiFreqTotalMap`
  from several components and slice it by field or frequency.
- [Examples / Extending gaussky](examples/extending.md) — write your own
  spectrum, SED, or component.
