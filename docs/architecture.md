# Architecture

`gaussky` is intentionally small. There are three pluggable model layers
(`ps`, `sed`, `component`), one validated container layer (`map`), one
high-level facade (`sampler`), and a small shared-conventions module.

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Sampler.sample                            │
│   (carries per-scene defaults; per-call kwargs override one-by-one)  │
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
       │  to_healpy_cls(lmax)│   │ scale_maps(maps,...)│
       └─────────────────────┘   └─────────────────────┘
                       │
                       ▼
       ┌─────────────────────────────────────────────┐
       │ component_utils.sample_component_map(...)   │
       │   • validate_healpy_cls                     │
       │   • hp.synfast (pol=True, new=True)         │
       │     (NumPy global RNG saved/restored        │
       │      around the call when seed is given)    │
       │   • hp.smoothing (pol=True)                 │
       │   • RING → NESTED if requested              │
       │   • sed.scale_maps(maps, freqs)             │
       │     (skipped when sed is None — CMB path)   │
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

A parallel harmonic-space path
(`component_utils.sample_component_alm(...)` → `MultiFreqCompAlm`) returns
T/E/B alm coefficients without going to a pixelated map; see the
[`gaussky.map`](reference/map.md) and
[`gaussky.component`](reference/component.md) references.

## How a single sample flows

1. The user constructs a `GaussianComponent` (e.g. `GaussianCMB`,
   `SimplePowerLawSynchrotron`, `SimpleModifiedBlackbodyDust`) with concrete
   `ps` and (optionally) `sed` parameters. Components are frozen dataclasses;
   parameter validation runs in `__post_init__`. SED-backed components
   typically subclass `BaseSEDBackedComponent` so the SED-cache + `sample_map`
   plumbing is shared.

2. `Sampler(nside, **defaults).sample(component, **overrides)` dispatches to
   `component.sample_map(nside=..., fields=..., freqs_ghz=..., lmax=...,
   seed=..., ...)`. Each per-call kwarg overrides the matching default
   captured at sampler construction time; the rest are inherited.

3. The component delegates to the unified helper in
   `gaussky/component/component_utils.py`:

   - `sample_component_map(*, ps, sed, ...)` — the one helper for both paths.
     When `sed` is given (synchrotron, dust), it samples one pivot-frequency
     T/Q/U realization with `hp.synfast` and applies
     `sed.scale_maps(maps, freqs)`. When `sed=None` (CMB), the same realization
     is broadcast across every requested frequency channel.

4. The helper computes `lmax = 3 * nside - 1` (overridable via the `lmax`
   kwarg), calls `ps.to_healpy_cls(lmax)`, and then runs `validate_healpy_cls`
   to check that the spectra:
   - have six entries in Healpy polarized order (`TT, EE, BB, TE, EB, TB`),
   - are one-dimensional with length `lmax + 1`,
   - contain only finite values,
   - and define a positive-semidefinite T/E/B covariance at every multipole.

5. `hp.synfast(..., pol=True, new=True)` produces the pivot T/Q/U map at the
   spectrum unit (translated via `gaussky.units.signal_unit_for(ps.unit)` —
   `uK_CMB^2 → uK_CMB` by default). When `seed` is given, the helper
   save/restores the NumPy global RNG around the `synfast` call so the draw
   is deterministic without leaking the seeded state to the rest of the
   program. Beam smoothing happens *before* SED scaling using
   `hp.smoothing(pol=True)`. A scalar beam is smoothed once; a per-channel
   beam smooths each frequency separately. RING→NESTED reordering is the
   last step before the result is wrapped in a `MultiFreqCompMap`.

6. When `Sampler.sample` is called with a sequence of components, every
   component is sampled independently with its own derived child seed (from
   `np.random.RandomState(root_seed)`), then
   `MultiFreqTotalMap.from_components` sums them. The total map enforces
   compatibility (same `nside`, ordering, coord, frequency grid, fields, unit,
   beam) and revalidates that `total.maps == sum(component.maps)`. The
   sampler-level seed and per-component seeds are stored on the total map's
   `metadata`.

## Why the helper exists

Putting `synfast`, `smoothing`, reordering, and SED scaling behind one shared
helper keeps the contract uniform across components and makes new components a
thin wrapper around `sample_component_map`. The helper also centralizes all
input validation: `nside` (positive power of two), `lmax` (non-negative int),
frequency arrays (positive 1-D), field tuples, beam shapes, and seed bounds.

If you implement a custom component, prefer reusing this helper rather than
calling `hp.synfast` directly — that way validation, ordering, seeding, and
metadata stay consistent with the bundled components.

## Map containers

Every sampled signal product flows through `MultiFreqCompMap`:

- Frozen dataclass with `eq=False` and read-only `float64` arrays.
- Signal arrays always have shape `(nfreq, nfield, npix)`.
- Carries `freqs_ghz`, `fields`, `unit`, `beam_fwhm_rad`, `nside`, `ordering`,
  `coord`, plus a component-level `component_name`, `metadata`, and
  `auxiliary_maps` map.
- `select_field`, `select_freq`, and `copy_with` return revalidated copies.

`MultiFreqTotalMap` is the same signal contract plus a tuple of components, a
name-uniqueness check, and its own `metadata` slot (the `Sampler` uses it to
record the root seed and the per-component child seeds). The bare constructor
revalidates the sum; the `from_components` classmethod is the canonical way
to build one and accepts an optional `metadata` dict.

A separate `AuxiliaryHealpixMap` is provided for products that are not signal
maps (e.g. spectral-index maps). It only carries the generic HEALPix contract
(`maps` array with trailing pixel axis, `nside`, `ordering`, `coord`).

A sibling `MultiFreqCompAlm` container (in `gaussky/map/almmap.py`) holds
spherical-harmonic coefficients with shape `(nfreq, nharm, nalm)` and the
matching harmonic-field tuple (`T`/`E`/`B`). It is the output of
`sample_component_alm(...)` and lets downstream code apply custom transfer
functions, pixel windows, or `hp.alm2map` at any chosen `nside`.

## Auxiliary maps live in three places

Auxiliary HEALPix products (a `beta_d(θ,φ)` template, a dust-temperature
map, a hits map, an apodization mask, etc.) split into three orthogonal
concerns, each with its own home:

1. **Output container** — the `auxiliary_maps` slot on
   `MultiFreqCompMap` holds whatever supporting products a component
   chose to publish alongside its sampled signal. The container itself is
   `AuxiliaryHealpixMap` in `gaussky/map/`. Same-pixelization validation
   runs on every entry.
2. **Model code that consumes an auxiliary map** lives **with the
   component that uses it**. A spatially-varying-β dust component, for
   example, declares its `beta_d_map` field inside
   `gaussky/component/dust/<variant>.py` (or its corresponding SED in
   `gaussky/sed/`), and republishes the accepted map through its own
   `auxiliary_maps` on the way out. Per-component auxiliary maps stay
   with the component that owns the semantics.
3. **Cross-cutting auxiliary-template loaders** live in
   `gaussky/templates/`, sibling to `gaussky/data/`. This is the right
   place for "load the standard Planck β_d template at this nside" —
   the loader returns an `AuxiliaryHealpixMap` that any dust component
   can consume. Bundled template data files go under
   `gaussky/data/templates/`.

This keeps the *production* of an auxiliary map (in `templates/` or in
a component), its *consumption* (in a component), and its *transport*
(in `map/`) decoupled, so adding a new variant of the dust or noise
zoo never needs to touch the storage layer.

## Where conventions live

`gaussky/conventions.py` is the single source of truth for the literal types
and ordering used across the package. `ps`, `sed`, `component`, and `map` all
re-export the same values. See [Conventions](conventions.md).

## Extension points

| You want to add…           | Implement…                                                | Where                  |
|----------------------------|-----------------------------------------------------------|------------------------|
| New `C_ell` model          | `AngularPowerSpectrum` (returns six Healpy-ordered cls)   | `gaussky/ps/`          |
| New SED                    | `SpectralEnergyDistribution` (`scale(freq_ghz)`, `nu0`)   | `gaussky/sed/`         |
| New SED-backed component   | Subclass `BaseSEDBackedComponent`                          | `gaussky/component/<category>/` |
| New frequency-independent component | `GaussianComponent` (`name`, `sample_map(...)`)  | `gaussky/component/<category>/` |
| Cross-cutting auxiliary template loader | Module under `gaussky.templates`          | `gaussky/templates/`   |
| Reuse sampling pipeline    | Call `sample_component_map(...)` in `component/component_utils.py` | (no change) |

For walkthroughs, see the [Extending example](examples/extending.md).
