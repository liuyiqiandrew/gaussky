# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`gaussky` is a Gaussian CMB/foreground sky simulator. It composes three independent models — an angular power spectrum (`ps`), a frequency-dependent SED (`sed`), and a `component` that ties them together — and samples them through HEALPix into multi-frequency T/Q/U map containers.

Python ≥ 3.11 is required; runtime dependencies are `numpy` and `healpy` only.

## Common commands

```bash
# Editable install (dev tools optional via [dev] extra)
pip install -e ".[dev]"
# or, set up a conda env from environment.yml
conda env create -f environment.yml

pytest                                 # run all tests
pytest tests/test_component.py         # single file
pytest tests/test_sampler.py::test_sampler_samples_single_component_map  # single test

black gaussky tests                    # format (line-length 88)
ruff check gaussky tests               # lint (E, F, I, B, UP, SIM; E501 ignored)
mypy gaussky                           # type-check (py311, disallow_untyped_defs=false)
```

`pyproject.toml` sets `pythonpath = ["."]` for pytest so tests import `gaussky` directly without install.

## Architecture

The codebase is organized as three pluggable model layers behind one `Sampler` facade:

```
ps/  (AngularPowerSpectrum)   →  C_ell, Healpy-ordered (TT, EE, BB, TE, EB, TB)
sed/ (SpectralEnergyDistribution) →  scalar frequency scaling, normalized at nu0_ghz
                                     (concrete SEDs subclass BaseSED)
component/ (GaussianComponent) →  binds ps + sed; produces a MultiFreqCompMap
                                  (SED-backed components subclass BaseSEDBackedComponent)
map/ (BaseSignalMap subclasses, MultiFreqCompAlm)
                                →  validated containers — signal maps shape
                                   (nfreq, nfield, npix); alms shape (nfreq, nharm, nalm)
sampler.Sampler                →  user entry point with per-scene defaults +
                                  per-call overrides; returns one component or a sum
templates/                     →  cross-cutting auxiliary-template loaders
                                  (β_d maps, etc.) reused by multiple components
units.POWER_TO_SIGNAL          →  extensible registry mapping spectrum unit
                                  → signal-map unit (currently uK_CMB^2 → uK_CMB)
```

Each layer is a `typing.Protocol` (`AngularPowerSpectrum`, `SpectralEnergyDistribution`, `GaussianComponent`) — new models implement the protocol and plug in without modifying core code. Concrete implementations are frozen `@dataclass`es that validate all parameters in `__post_init__`.

Top-level public API:

```python
from gaussky import (
    GaussianCMB, GaussianComponent,
    SimpleModifiedBlackbodyDust, SimplePowerLawSynchrotron,
    MultiFreqCompMap, MultiFreqTotalMap,
    Sampler,
)
```

### How a sample flows

`Sampler.sample(comp, ...)` (with options inherited from sampler-level defaults unless overridden)
→ dispatches to `component.sample_map(...)`
→ delegates to the unified helper in `gaussky/component/component_utils.py`:

- `sample_component_map(*, ps, sed, ...)` — single helper for both paths. When `sed` is given (synchrotron, dust), it draws one pivot T/Q/U realization via `hp.synfast` and applies `sed.scale_maps(maps, freqs)`. When `sed=None` (CMB), the same realization is broadcast across the requested frequency axis.

The helper computes `lmax = 3 * nside - 1` (overridable via the `lmax` kwarg), calls `ps.to_healpy_cls(lmax)`, runs `validate_healpy_cls` (shape + finiteness + positive-semidefinite T/E/B covariance), then `hp.synfast(..., pol=True, new=True)`. Beam smoothing via `hp.smoothing(pol=True)` happens before SED scaling; RING→NESTED reordering happens last. An optional `seed` is plumbed through; the helper saves/restores the NumPy global RNG around `hp.synfast`.

Calling `Sampler.sample` with a sequence of components returns a `MultiFreqTotalMap` built from `MultiFreqTotalMap.from_components`, which enforces compatible pixelization, frequency grid, fields, unit, beam, and coord across components and validates that `maps` equals the component sum.

There is also `sample_component_alm(*, ps, sed, lmax, ...)` in the same module, returning a `MultiFreqCompAlm` (T/E/B alm coefficients, shape `(nfreq, nharm, nalm)`) for callers who want to work in harmonic space or apply custom transfer functions before going to a map.

### Sampler ergonomics

`Sampler(nside, *, fields, freqs_ghz, beam_fwhm_rad, coord, ordering, seed, lmax)` captures per-scene defaults at construction time. `Sampler.sample(comp, **overrides)` resolves each argument to the explicit override when supplied (including `None` — distinguished from "not supplied" via an internal `_Unset` sentinel) or to the stored default otherwise. `Sampler.with_(**changes)` returns a sibling sampler with selected defaults replaced; the original is not mutated.

### Conventions (gaussky/conventions.py)

Single source of truth for cross-package literals. **Always import these rather than redefining**:
- `SignalField = Literal["T", "Q", "U"]`, `HarmonicField = Literal["T", "E", "B"]`
- `SpectrumPair = Literal["TT", "EE", "BB", "TE", "EB", "TB"]` — this **is** the Healpy polarized ordering (`HEALPY_POLARIZED_ORDER`). Any `to_healpy_cls` implementation must return spectra in this order.
- `HealpixOrdering = Literal["RING", "NESTED"]`
- Unit strings: `U_K_CMB`, `U_K_CMB_SQUARED`, `DIMENSIONLESS`.

Shared validators live in `gaussky/_validation.py` (private) and are re-exported via `gaussky/map/map_utils.py` and used directly from `gaussky/component/component_utils.py` and `gaussky/sed/base.py`. Don't duplicate them.

Unit mapping spectrum → signal lives in `gaussky/units.py`. Bundled mapping: `"uK_CMB^2" → "uK_CMB"`. New units register with `gaussky.units.register_signal_unit(power_unit, signal_unit)`.

### Map containers (gaussky/map/)

All map containers are frozen dataclasses with `eq=False`, read-only arrays, and revalidating `copy_with` / `select_field` / `select_freq` helpers. Signal arrays always have shape `(nfreq, nfield, npix)`. `BeamFwhm = float | ndarray | None` — scalar means shared across channels, 1-D means per-channel.

`MultiFreqCompMap` carries a `component_name`, `metadata` (read-only `MappingProxyType` of model parameters — sampler-level helpers stitch the `seed` in here), and `auxiliary_maps`. `MultiFreqTotalMap.from_components` is the canonical constructor — it sums and validates; the bare constructor also revalidates by recomputing the sum and accepts an optional `metadata` dict for sampler-level provenance.

`MultiFreqCompAlm` (in `gaussky/map/almmap.py`) is the harmonic-space sibling: `alms` shape `(nfreq, nharm, nalm)`, `lmax`/`mmax`, harmonic fields (`T`/`E`/`B`), and the usual `freqs_ghz`/`unit`/`coord`/`component_name`/`metadata`/`beam_fwhm_rad` (informational).

### Bundled CMB templates

`gaussky/data/cmb_spec/{camb_lens_nobb.dat, camb_lens_r1.dat}` are CAMB `D_ell` outputs with columns `ell, TT, EE, BB, TE`. `CMBCl` combines them as `a_lens * D_nobb + r_tensor * (D_r1 - D_nobb)`, converts `D_ell → C_ell`, and returns `TB = EB = 0`. Loading is cached via `@lru_cache` keyed on `(template_dir, no_tensor_mtime_ns, r1_mtime_ns)`, so editing a template file invalidates the cache automatically (no Python restart needed). The package-data entry in `pyproject.toml` ships these files with installs. `CMBCl.unit` is a `ClassVar` fixed at `"uK_CMB^2"` — not a constructor argument.

### Component package layout

Each component category is a sub-package under `gaussky/component/`:

```
gaussky/component/
├── base.py                # GaussianComponent Protocol
├── sed_backed.py          # BaseSEDBackedComponent mixin
├── component_utils.py     # sample_component_map, sample_component_alm, helpers
├── cmb/__init__.py        # re-exports
│   └── lensed.py          # GaussianCMB
├── dust/__init__.py
│   └── simple_mbb.py      # SimpleModifiedBlackbodyDust
└── synchrotron/__init__.py
    └── simple_powerlaw.py # SimplePowerLawSynchrotron
```

The flat public surface (`from gaussky.component import …` or `from gaussky import …`) is preserved by re-exports in each `__init__.py`. New variants of a category drop into the same sub-package without touching anything else.

## Notes when extending

- New components: prefer subclassing `BaseSEDBackedComponent` (declare your model fields + `_build_sed` + `_metadata`). For frequency-independent components, implement the `GaussianComponent` protocol directly and delegate to `sample_component_map(sed=None, ...)`. Either way, reuse the helpers in `component_utils.py` rather than calling `hp.synfast` directly so validation, seeding, and ordering stay consistent.
- New power spectra: implement `AngularPowerSpectrum.to_healpy_cls(lmax)` returning six `float64` arrays of length `lmax+1` in `HEALPY_POLARIZED_ORDER`. The covariance positive-semidefinite check in `validate_healpy_cls` runs on every sample, so cross-spectra must be self-consistent. If your spectrum has a non-default unit, register it with `gaussky.units.register_signal_unit(power_unit, signal_unit)` first.
- New SEDs: subclass `BaseSED` (`gaussky/sed/common.py`) and implement only `_rj_scaling(freq_ghz)`. The base class provides `scale`, `scale_maps`, and `scale_cls` in terms of staged RJ ⇄ CMB conversions matching the `pygsm` operation order. A standalone class implementing the three Protocol methods directly also works — the SED protocol is structural.
- New foreground/CMB variants live under `gaussky/component/<category>/<variant>.py` and are re-exported from `gaussky/component/<category>/__init__.py` and from `gaussky/component/__init__.py` (which still uses the `__getattr__` lazy switch).
- Cross-cutting auxiliary templates (e.g. β_d maps shared by several dust variants) live in `gaussky/templates/`. Bundled data files for them go under `gaussky/data/templates/`.
- Dev env: this repo uses the conda env `vis` (`/home/yl9946/.conda/envs/vis/bin/python`). It has `healpy`, `pytest`, and `black`. `mypy` and `ruff` are not installed there.
