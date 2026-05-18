# Shared conventions

`gaussky/conventions.py` is the single source of truth for the literal types,
spectrum orderings, and unit strings used across the package. **Always import
these names rather than redefining them.** Other subpackages re-export the same
values for convenience (for example `gaussky.ps.HEALPY_POLARIZED_ORDER`,
`gaussky.map.SUPPORTED_SIGNAL_FIELDS`).

## Signal fields

```python
from gaussky.conventions import SIGNAL_FIELDS, SignalField

SignalField     # Literal["T", "Q", "U"]
SIGNAL_FIELDS   # ("T", "Q", "U")
```

Signal fields appear along axis 1 of every signal map. Components let the
caller request any subset in any order; the helper preserves the requested
order.

## Harmonic fields

```python
from gaussky.conventions import HARMONIC_FIELDS, HarmonicField

HarmonicField     # Literal["T", "E", "B"]
HARMONIC_FIELDS   # ("T", "E", "B")
```

Used for cross-power covariance matrices in
`PowerLawCl.alm_covariance` and inside `validate_healpy_cls`.

## Spectrum pairs and the Healpy polarized order

```python
from gaussky.conventions import HEALPY_POLARIZED_ORDER, SpectrumPair

SpectrumPair             # Literal["TT", "EE", "BB", "TE", "EB", "TB"]
HEALPY_POLARIZED_ORDER   # ("TT", "EE", "BB", "TE", "EB", "TB")
```

`HEALPY_POLARIZED_ORDER` **is** the convention that `healpy.synfast(..., pol=True, new=True)`
expects. Any `AngularPowerSpectrum.to_healpy_cls(lmax)` implementation must
return six `float64` arrays of length `lmax + 1` in exactly this order.

## HEALPix ordering

```python
from gaussky.conventions import HEALPIX_ORDERINGS, HealpixOrdering

HealpixOrdering       # Literal["RING", "NESTED"]
HEALPIX_ORDERINGS     # ("RING", "NESTED")
```

The sampling pipeline always works in `RING` internally (because
`hp.synfast` and `hp.smoothing` are RING-ordered). RING→NESTED conversion
happens as the last step before wrapping the result in a `MultiFreqCompMap`.

Map containers accept either case when the user supplies the ordering — both
`"ring"` and `"RING"` work and are canonicalized.

## Units

The unit *strings* live in `conventions.py`:

```python
from gaussky.conventions import U_K_CMB, U_K_CMB_SQUARED, DIMENSIONLESS

U_K_CMB           # "uK_CMB"
U_K_CMB_SQUARED   # "uK_CMB^2"
DIMENSIONLESS     # "dimensionless"
```

The mapping from a *power-spectrum unit* to the corresponding *signal-map
unit* lives in the extensible registry `gaussky.units`:

```python
from gaussky.units import POWER_TO_SIGNAL, register_signal_unit, signal_unit_for

signal_unit_for("uK_CMB^2")        # "uK_CMB" (the bundled default)
register_signal_unit("K_RJ^2", "K_RJ")
```

The sampling pipeline calls `signal_unit_for(ps.unit)` rather than
hard-coding the mapping, so noise components or alternative spectra (e.g.
`K_RJ^2`, `Jy^2/sr`) plug in by registering their pair before sampling.
Registering the same pair twice is a no-op; conflicting remaps raise.

## Putting it together

A component that respects every convention looks like this:

```python
from gaussky.conventions import (
    HEALPY_POLARIZED_ORDER,
    HealpixOrdering,
    SignalField,
    U_K_CMB_SQUARED,
)

class MyPowerSpectrum:
    unit: str = U_K_CMB_SQUARED

    def to_healpy_cls(self, lmax: int):
        # Must return six float64 arrays in HEALPY_POLARIZED_ORDER,
        # each of length lmax + 1.
        ...

def my_component_sample(
    nside: int,
    fields: tuple[SignalField, ...],
    ordering: HealpixOrdering = "RING",
):
    ...
```

Tests verify the re-exports stay aligned with `conventions.py`. See
`tests/test_power_spectrum.py::test_power_spectrum_exports_shared_pair_ordering`
and `tests/test_map.py::test_map_exports_shared_conventions`.
