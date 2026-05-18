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
├── __init__.py         ← top-level public API re-exports
├── conventions.py      ← shared types, units, and orderings
├── _validation.py      ← single source for normalize_* / validate_* helpers
├── units.py            ← extensible power-spectrum → signal-unit registry
├── sampler.py          ← high-level entry point (defaults + per-call overrides)
├── ps/                 ← angular power spectra (C_ell models)
├── sed/                ← spectral energy distributions
│   ├── base.py             ← SpectralEnergyDistribution Protocol
│   ├── common.py           ← BaseSED + PowerLawSED + ModifiedBlackbodySED
│   └── sed_utils.py        ← planck_rj_spectrum, trj_to_tcmb, tcmb_to_trj
├── component/          ← Gaussian sky components (ps + sed binders)
│   ├── base.py             ← GaussianComponent Protocol
│   ├── sed_backed.py       ← BaseSEDBackedComponent mixin
│   ├── component_utils.py  ← sample_component_map, sample_component_alm
│   ├── cmb/                ← GaussianCMB
│   ├── dust/               ← SimpleModifiedBlackbodyDust (+ future variants)
│   └── synchrotron/        ← SimplePowerLawSynchrotron (+ future variants)
├── map/                ← validated containers (signal map + alm)
├── templates/          ← cross-cutting auxiliary-template loaders
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

from gaussky import GaussianCMB, Sampler

sampler = Sampler(
    nside=64,
    fields=("T", "Q", "U"),
    freqs_ghz=np.array([30.0, 90.0, 150.0]),
    seed=42,
)
sky = sampler.sample(GaussianCMB())  # default a_lens=1.0, r_tensor=0.0

print(sky.maps.shape)        # (3, 3, 49152)  -- (nfreq, nfield, npix)
print(sky.unit)              # 'uK_CMB'
print(sky.component_name)    # 'cmb'
print(sky.metadata)          # MappingProxyType({'a_lens': 1.0, 'r_tensor': 0.0, ..., 'seed': 42})
```

The CMB component samples the spectrum once and replicates the same realization
across the requested frequency channels (CMB has no SED). Per-frequency beam
smoothing is still supported — see [Components](reference/component.md).

Same call with a different seed reproduces a different draw without touching
the sampler:

```python
other = sampler.sample(GaussianCMB(), seed=7)
assert not (sky.maps == other.maps).all()

# Or replace the sampler's default seed for the rest of the session:
sampler_v2 = sampler.with_(seed=2025)
```

## Next steps

- [Architecture overview](architecture.md) — what happens inside that
  `sampler.sample(...)` call.
- [Examples / Foregrounds](examples/foregrounds.md) — add power-law synchrotron
  and modified-blackbody dust to the same scene.
- [Examples / Total map](examples/total-map.md) — build a `MultiFreqTotalMap`
  from several components and slice it by field or frequency.
- [Examples / Extending gaussky](examples/extending.md) — write your own
  spectrum, SED, or component.
