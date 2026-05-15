# Repository Guidelines

## Project Structure & Module Organization

`gaussky/` contains the package source. Core conventions live in
`gaussky/conventions.py`; map containers are under `gaussky/map/`; angular
power spectra are under `gaussky/ps/`; SED models are under `gaussky/sed/`;
sky components are under `gaussky/component/`; and the high-level sampler is
`gaussky/sampler.py`. Bundled CMB CAMB templates are stored in
`gaussky/data/cmb_spec/` and are included as package data. Tests live in
`tests/` and are organized by subsystem, for example `test_power_spectrum.py`.

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
class names descriptive, such as `PowerLawCl`, `CMBCl`, and `GaussianCMB`.
Component map arrays follow `(nfreq, nfield, npix)` throughout the package.

## Testing Guidelines

Use `pytest`. Add tests beside related subsystem tests and name them
`test_<behavior>`. Prefer deterministic fake Healpy objects for component tests
that would otherwise depend on random map draws. Cover validation failures,
shape conventions, metadata, and compatibility with `MultiFreqCompMap` and
`MultiFreqTotalMap` when changing sampler or component behavior.

## Commit & Pull Request Guidelines

Existing commits use short summary lines such as `refactor map implementation`
and `add simple dust implementation`. Follow that style: concise, imperative,
and focused on one logical change. Pull requests should describe the change,
list validation commands run, note any dependency or data-file changes, and link
related issues when applicable. Include screenshots only for visual artifacts,
which this package normally does not require.

## Agent-Specific Notes

Do not remove or rewrite bundled data in `gaussky/data/cmb_spec/` unless the
source and validation method are documented. Avoid committing generated caches
such as `__pycache__/`, `.pytest_cache/`, or `.mypy_cache/`.
