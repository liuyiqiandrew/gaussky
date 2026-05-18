# Repository Guidelines

## Project Structure & Module Organization

`gaussky/` contains the package source. Core conventions live in
`gaussky/conventions.py`; shared scalar/array validators are in
`gaussky/_validation.py`; and power-spectrum to signal-unit mappings are in
`gaussky/units.py`. Map containers live under `gaussky/map/`, including
pixel-space containers in `compmap.py` and harmonic-space containers in
`almmap.py`. Angular power spectra are under `gaussky/ps/`, SED models under
`gaussky/sed/`, and the high-level sampler is `gaussky/sampler.py`.

Sky components are organized by component family under `gaussky/component/`:
shared protocols and sampling helpers live in `base.py` and
`component_utils.py`, common SED-backed plumbing lives in `sed_backed.py`, and
concrete implementations live in subpackages such as
`component/cmb/lensed.py`, `component/dust/simple_mbb.py`, and
`component/synchrotron/simple_powerlaw.py`. Cross-cutting auxiliary template
loaders belong under `gaussky/templates/`. Bundled CMB CAMB templates are
stored in `gaussky/data/cmb_spec/` and are included as package data. User and
developer docs live in `docs/`. Tests live in `tests/` and are organized by
subsystem, for example `test_component.py`, `test_almmap.py`, and
`test_units.py`.

## Build, Test, and Development Commands

- `python -m pytest`: run the full test suite. Requires project dependencies,
  including `healpy`.
- `python -m pytest tests/test_power_spectrum.py`: run a focused subsystem test.
- `python -m black gaussky tests`: format Python source and tests.
- `python -m black --check gaussky tests`: verify formatting without rewriting.
- `python -m mypy gaussky`: run static type checks using `pyproject.toml`.
- `python -m ruff check gaussky tests`: run linting when `ruff` is installed.

Install development dependencies from `pyproject.toml` with the package manager
used in your environment, for example editable install plus the `dev` extra.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax, 4-space indentation, and Black formatting with an
88-character line length. Prefer explicit type annotations for public APIs and
NumPy-style docstrings for modules, classes, and nontrivial helpers. Keep public
class names descriptive, such as `PowerLawCl`, `CMBCl`, `GaussianCMB`,
`SimpleModifiedBlackbodyDust`, and `SimplePowerLawSynchrotron`. Component
pixel maps follow `(nfreq, nfield, npix)` throughout the package; component
alm containers follow `(nfreq, nharm, nalm)` with harmonic fields `T`, `E`,
and `B`. Prefer importing shared validation from `gaussky/_validation.py` and
unit translation from `gaussky/units.py` rather than duplicating local helpers.
New concrete components should live in the appropriate family subpackage and
reuse the shared component sampling helpers when possible.

## Testing Guidelines

Use `pytest`. Add tests beside related subsystem tests and name them
`test_<behavior>`. Prefer deterministic fake Healpy objects for component tests
that would otherwise depend on random map draws. Cover validation failures,
shape conventions, metadata, and compatibility with `MultiFreqCompMap` and
`MultiFreqTotalMap` when changing sampler or component behavior. Cover seeded
sampling metadata and NumPy RNG restoration when touching component sampling or
`Sampler`. Add `MultiFreqCompAlm` coverage in `test_almmap.py` when changing
harmonic-space behavior, and add unit-registry coverage in `test_units.py` when
changing power-spectrum to signal-unit handling.

## Commit & Pull Request Guidelines

Existing commits use short summary lines such as `refactor map implementation`
and `add simple dust implementation`. Follow that style: concise, imperative,
and focused on one logical change. Pull requests should describe the change,
list validation commands run, note any dependency or data-file changes, and link
related issues when applicable. Include screenshots only for visual artifacts,
which this package normally does not require.

## Agent-Specific Notes

Do not remove or rewrite bundled data in `gaussky/data/cmb_spec/` unless the
source and validation method are documented. Put shared auxiliary-template
loaders under `gaussky/templates/`; component-specific auxiliary-map semantics
belong with the component implementation that consumes them. Avoid committing
generated caches such as `__pycache__/`, `.pytest_cache/`, or `.mypy_cache/`.
