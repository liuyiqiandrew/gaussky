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
component/ (GaussianComponent) →  binds ps + sed; produces a MultiFreqCompMap
map/ (BaseSignalMap subclasses) →  validated (nfreq, nfield, npix) containers
sampler.Sampler                →  user entry point; returns one component or a sum
```

Each layer is a `typing.Protocol` (`AngularPowerSpectrum`, `SpectralEnergyDistribution`, `GaussianComponent`) — new models implement the protocol and plug in without modifying core code. Concrete implementations are frozen `@dataclass`es that validate all parameters in `__post_init__`.

### How a sample flows

`Sampler.sample(comp, fields, freqs_ghz, ...)`
→ dispatches to `component.sample_map(...)`
→ delegates to one of two helpers in `gaussky/component/component_utils.py`:
- `sample_gaussian_component_map` — SED-backed (dust, synchrotron): one pivot T/Q/U realization via `hp.synfast`, then multiplied by `sed.scale(freqs)`.
- `sample_frequency_independent_gaussian_component_map` — CMB: same realization replicated across frequency channels, no SED.

Both helpers compute `lmax = 3 * nside - 1`, call `ps.to_healpy_cls(lmax)`, run `validate_healpy_cls` (shape + finiteness + positive-semidefinite T/E/B covariance), then `hp.synfast(..., pol=True, new=True)`. Beam smoothing via `hp.smoothing(pol=True)` happens before SED scaling; RING→NESTED reordering happens last.

Calling `Sampler.sample` with a sequence of components returns a `MultiFreqTotalMap` built from `MultiFreqCompMap.from_components`, which enforces compatible pixelization, frequency grid, fields, unit, beam, and coord across components and validates that `maps` equals the component sum.

### Conventions (gaussky/conventions.py)

Single source of truth for cross-package literals. **Always import these rather than redefining**:
- `SignalField = Literal["T", "Q", "U"]`, `HarmonicField = Literal["T", "E", "B"]`
- `SpectrumPair = Literal["TT", "EE", "BB", "TE", "EB", "TB"]` — this **is** the Healpy polarized ordering (`HEALPY_POLARIZED_ORDER`). Any `to_healpy_cls` implementation must return spectra in this order.
- `HealpixOrdering = Literal["RING", "NESTED"]`
- Unit strings: `U_K_CMB`, `U_K_CMB_SQUARED`, `DIMENSIONLESS`. The signal unit is currently derived strictly from the spectrum unit (`uK_CMB^2` → `uK_CMB`) in `_signal_unit`.

### Map containers (gaussky/map/)

All map containers are frozen dataclasses with `eq=False`, read-only float64 arrays, and revalidating `copy_with` / `select_field` / `select_freq` helpers. Signal arrays always have shape `(nfreq, nfield, npix)`. `BeamFwhm = float | ndarray | None` — scalar means shared across channels, 1-D means per-channel.

`MultiFreqCompMap` carries a `component_name`, `metadata` (read-only `MappingProxyType` of model parameters), and `auxiliary_maps`. `MultiFreqTotalMap.from_components` is the canonical constructor — it sums and validates; the bare constructor also revalidates by recomputing the sum.

### Bundled CMB templates

`gaussky/data/cmb_spec/{camb_lens_nobb.dat, camb_lens_r1.dat}` are CAMB `D_ell` outputs with columns `ell, TT, EE, BB, TE`. `CMBCl` combines them as `a_lens * D_nobb + r_tensor * (D_r1 - D_nobb)`, converts `D_ell → C_ell`, and returns `TB = EB = 0`. Loading is cached via `@lru_cache` keyed on the template directory string, so changes to template files require restarting the process. The package-data entry in `pyproject.toml` ships these files with installs.

## Notes when extending

- New components: implement the `GaussianComponent` protocol (have a `name: str` and a keyword-only `sample_map` matching the signature in `gaussky/component/base.py`). Reuse the helpers in `component_utils.py` rather than calling `hp.synfast` directly so validation and ordering stay consistent.
- New power spectra: implement `AngularPowerSpectrum.to_healpy_cls(lmax)` returning six `float64` arrays of length `lmax+1` in `HEALPY_POLARIZED_ORDER`. The covariance positive-semidefinite check in `validate_healpy_cls` runs on every sample, so cross-spectra must be self-consistent.
- New SEDs: implement `scale(freq_ghz)` returning dimensionless factors normalized to one at `self.nu0_ghz`. Follow `gaussky/sed/common.py` and use the `trj_to_tcmb` / `tcmb_to_trj` helpers so the output stays in thermodynamic CMB units.
- The component package lazy-loads concrete classes via `__getattr__` in `gaussky/component/__init__.py` — register new public components there.
