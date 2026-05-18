# Architecture

`gaussky` is intentionally small. There are three pluggable model layers
(`ps`, `sed`, `component`), one validated container layer (`map`), one
high-level facade (`sampler`), and a small shared-conventions module.

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Sampler.sample                            │
│                              (facade)                                │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────┐
            │      component.sample_map(...)       │
            │   (one of the GaussianComponent)     │
            └──────────────────────────────────────┘
                       │                │
                       ▼                ▼
       ┌─────────────────────┐   ┌─────────────────────┐
       │  AngularPowerSpec   │   │ SpectralEnergyDistr │
       │  to_healpy_cls(lmax)│   │     scale(freq)     │
       └─────────────────────┘   └─────────────────────┘
                       │
                       ▼
       ┌─────────────────────────────────────────────┐
       │ component_utils.sample_gaussian_component_  │
       │ map  /  ..._frequency_independent_...       │
       │   • validate_healpy_cls                     │
       │   • hp.synfast (pol=True, new=True)         │
       │   • hp.smoothing (pol=True)                 │
       │   • RING → NESTED if requested              │
       │   • SED scaling (skipped for CMB)           │
       └─────────────────────────────────────────────┘
                       │
                       ▼
       ┌─────────────────────────────────────────────┐
       │            MultiFreqCompMap                 │
       │      (nfreq, nfield, npix) signal map       │
       └─────────────────────────────────────────────┘
                       │
                       │  (only for sequence of components)
                       ▼
       ┌─────────────────────────────────────────────┐
       │    MultiFreqTotalMap.from_components(...)   │
       │       compatibility checks + sum            │
       └─────────────────────────────────────────────┘
```

## How a single sample flows

1. The user constructs a `GaussianComponent` (e.g. `GaussianCMB`,
   `SimplePowerLawSynchrotron`, `SimpleModifiedBlackbodyDust`) with concrete
   `ps` and (optionally) `sed` parameters. Components are frozen dataclasses;
   parameter validation runs in `__post_init__`.

2. `Sampler(nside).sample(component, ...)` dispatches to
   `component.sample_map(nside=..., fields=..., freqs_ghz=..., ...)`. The
   `Sampler` only owns `nside`; everything else flows through as keyword
   arguments.

3. The component delegates to one of the two helpers in
   `gaussky/component/component_utils.py`:

   - `sample_gaussian_component_map` — used for SED-backed components
     (synchrotron, dust). Samples one pivot-frequency T/Q/U realization with
     `hp.synfast`, then multiplies it by `sed.scale(freqs)`.
   - `sample_frequency_independent_gaussian_component_map` — used for the CMB
     and any future component without an SED. Samples one realization and
     repeats it across the frequency axis.

4. Both helpers compute `lmax = 3 * nside - 1`, call
   `ps.to_healpy_cls(lmax)`, and then run `validate_healpy_cls` to check that
   the spectra:
   - have six entries in Healpy polarized order (`TT, EE, BB, TE, EB, TB`),
   - are one-dimensional with length `lmax + 1`,
   - contain only finite values,
   - and define a positive-semidefinite T/E/B covariance at every multipole.

5. `hp.synfast(..., pol=True, new=True)` produces the pivot T/Q/U map at the
   spectrum unit (`uK_CMB^2` → `uK_CMB`). Beam smoothing happens *before* SED
   scaling using `hp.smoothing(pol=True)`. A scalar beam is smoothed once; a
   per-channel beam smooths each frequency separately. RING→NESTED reordering
   is the last step before the result is wrapped in a `MultiFreqCompMap`.

6. When `Sampler.sample` is called with a sequence of components, every
   component is sampled independently, then `MultiFreqTotalMap.from_components`
   sums them. The total map enforces compatibility (same `nside`, ordering,
   coord, frequency grid, fields, unit, beam) and revalidates that
   `total.maps == sum(component.maps)`.

## Why the helpers exist

Putting `synfast`, `smoothing`, and reordering behind shared helpers keeps the
contract uniform across components and makes new components a thin wrapper
around `_sample_*_component_map`. The helpers also centralize all input
validation: `nside` (positive power of two), frequency arrays
(positive 1-D), field tuples, and beam shapes.

If you implement a custom component, prefer reusing these helpers rather than
calling `hp.synfast` directly — that way validation, ordering, and metadata
stays consistent with the bundled components.

## Map containers

Every sampled product flows through `MultiFreqCompMap`:

- Frozen dataclass with `eq=False` and read-only `float64` arrays.
- Signal arrays always have shape `(nfreq, nfield, npix)`.
- Carries `freqs_ghz`, `fields`, `unit`, `beam_fwhm_rad`, `nside`, `ordering`,
  `coord`, plus a component-level `component_name`, `metadata`, and
  `auxiliary_maps` map.
- `select_field`, `select_freq`, and `copy_with` return revalidated copies.

`MultiFreqTotalMap` is the same signal contract plus a tuple of components and
a name-uniqueness check. The bare constructor revalidates the sum; the
`from_components` classmethod is the canonical way to build one.

A separate `AuxiliaryHealpixMap` is provided for products that are not signal
maps (e.g. spectral-index maps). It only carries the generic HEALPix contract
(`maps` array with trailing pixel axis, `nside`, `ordering`, `coord`).

## Where conventions live

`gaussky/conventions.py` is the single source of truth for the literal types
and ordering used across the package. `ps`, `sed`, `component`, and `map` all
re-export the same values. See [Conventions](conventions.md).

## Extension points

| You want to add…           | Implement…                                                | Where                  |
|----------------------------|-----------------------------------------------------------|------------------------|
| New `C_ell` model          | `AngularPowerSpectrum` (returns six Healpy-ordered cls)   | `gaussky/ps/`          |
| New SED                    | `SpectralEnergyDistribution` (`scale(freq_ghz)`, `nu0`)   | `gaussky/sed/`         |
| New component              | `GaussianComponent` (`name`, `sample_map(...)`)           | `gaussky/component/`   |
| Reuse sampling pipeline    | Call the helpers in `component/component_utils.py`        | (no change)            |

For walkthroughs, see the [Extending example](examples/extending.md).
