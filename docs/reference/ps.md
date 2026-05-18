# `gaussky.ps` — angular power spectra

The `ps` subpackage defines what every Gaussian component samples from. It
ships:

- An `AngularPowerSpectrum` `Protocol`.
- `validate_healpy_cls`, the shared validator the sampling helpers call before
  every `hp.synfast`.
- Two concrete models: `PowerLawCl` (analytic) and `CMBCl` (template-backed).

```python
from gaussky.ps import (
    AngularPowerSpectrum,     # protocol
    CMBCl,                    # CAMB-template CMB model
    HEALPY_POLARIZED_ORDER,   # ("TT","EE","BB","TE","EB","TB")
    PowerLawCl,               # power-law model
    SpectrumPair,             # Literal type
    validate_healpy_cls,      # shared validator
)
```

## The protocol

```python
class AngularPowerSpectrum(Protocol):
    unit: str
    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]: ...
```

Implementations must return six `float64` arrays of length `lmax + 1` ordered
as `TT, EE, BB, TE, EB, TB` — the polarized ordering expected by
`healpy.synfast(..., pol=True, new=True)`.

`unit` is the unit of the returned `C_ell` values. The sampling pipeline
translates the power-spectrum unit to a signal-map unit through the
registry in [`gaussky.units`](../conventions.md#units); the bundled mapping
is `"uK_CMB^2" → "uK_CMB"`. Register additional pairs with
`gaussky.units.register_signal_unit(power_unit, signal_unit)` before
sampling with a custom spectrum unit.

## `validate_healpy_cls(healpy_cls, lmax, *, atol=0.0)`

Used inside `component_utils.sample_*` to harden every realization.

Checks performed:

- exactly six spectra;
- each spectrum is 1-D and has shape `(lmax + 1,)`;
- all values are finite;
- the T/E/B covariance built from `(TT, EE, BB, TE, EB, TB)` is
  positive semidefinite at every multipole.

Returns a list of read-only `float64` arrays in the same order.

```python
import numpy as np
from gaussky.ps import PowerLawCl, validate_healpy_cls

cls = PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_bb=1.0).to_healpy_cls(lmax=80)
validated = validate_healpy_cls(cls, lmax=80)
assert all(not v.flags.writeable for v in validated)
```

The `atol` keyword controls how negative an eigenvalue may be before the
covariance is rejected — useful for templates where floating-point roundoff
produces tiny negative eigenvalues.

## `PowerLawCl`

Analytic per-spectrum power law of the form
`C_ell = amp * (ell / ell0) ** alpha`, with one `(amp, alpha)` pair per
spectrum component. Stored as a frozen dataclass; all parameters are validated
in `__post_init__`.

### Parameters

| Field                        | Default     | Meaning                                                            |
|------------------------------|-------------|--------------------------------------------------------------------|
| `amp_tt … amp_tb`            | `0.0`       | Spectrum amplitudes at `ell0` (in `unit`).                         |
| `alpha_tt … alpha_tb`        | `0.0`       | Power-law slopes per spectrum component.                           |
| `ell0`                       | `80.0`      | Reference multipole. Must be strictly positive.                    |
| `ell_min`                    | `2`         | Multipoles below this value are forced to zero.                    |
| `unit`                       | `"uK_CMB^2"`| Unit of the returned values.                                       |
| `is_cell`                    | `True`      | If `False`, amplitudes are interpreted as `D_ell` and converted.   |

Auto-spectrum amplitudes (`TT, EE, BB`) must be non-negative; cross spectra may
be either sign, but they must keep the T/E/B covariance PSD or
`validate_healpy_cls` will reject the realization.

### Methods

```python
spectrum.cl(pair, ell)        # C_ell at one or more multipoles
spectrum.to_healpy_cls(lmax)  # six arrays in Healpy polarized order
spectrum.alm_covariance(ell)  # ell.shape + (3, 3) T/E/B covariance
spectrum.validate_positive_semidefinite(ell, *, atol=0.0)
```

### Examples

A flat EE-only spectrum normalized at `ell0 = 80`:

```python
from gaussky.ps import PowerLawCl

spec = PowerLawCl(amp_ee=2.0, alpha_ee=-2.4, ell0=80.0)
spec.cl("EE", 80.0)  # array(2.0)
```

Six-spectrum container suitable for `synfast`:

```python
spec = PowerLawCl(
    amp_tt=1.0, alpha_tt=-2.0,
    amp_ee=0.5, alpha_ee=-2.5,
    amp_bb=0.2, alpha_bb=-2.5,
)
cls = spec.to_healpy_cls(lmax=3)
assert len(cls) == 6
```

`D_ell` amplitudes (e.g. taken from a paper that quotes `ell(ell+1)C_ell/2π`):

```python
spec = PowerLawCl(
    amp_ee=56.0, alpha_ee=-0.32,
    ell0=80.0, is_cell=False,    # amplitudes are D_ell
)
```

PSD self-check (handy when you're tweaking cross-spectrum amplitudes):

```python
import numpy as np

spec = PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_te=0.5)
spec.validate_positive_semidefinite(np.arange(2, 100))   # ok
spec2 = PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_te=2.0)
spec2.validate_positive_semidefinite(np.arange(2, 100))  # raises ValueError
```

## `CMBCl`

Frozen dataclass that combines two bundled CAMB `D_ell` templates and exposes
them as Healpy-ordered `C_ell`.

The bundled files live under `gaussky/data/cmb_spec/`:

- `camb_lens_nobb.dat` — lensed CMB with no tensor (the lensing template).
- `camb_lens_r1.dat`   — CMB at `r = 1`.

Both files have columns `ell, TT, EE, BB, TE`. `CMBCl` combines them as

```
D_ell = a_lens * D_ell^(lens-nobb) + r_tensor * (D_ell^(r1) - D_ell^(lens-nobb))
```

then converts to `C_ell = 2π / (ell(ell+1)) * D_ell`. `TB` and `EB` are set
exactly to zero.

### Parameters

| Field          | Default                 | Meaning                                                       |
|----------------|-------------------------|---------------------------------------------------------------|
| `a_lens`       | `1.0`                   | Lensing-template amplitude. `1` returns the template.         |
| `r_tensor`     | `0.0`                   | Tensor-to-scalar ratio.                                       |
| `template_dir` | `DEFAULT_CMB_SPEC_DIR`  | Directory containing the two CAMB files.                      |

`CMBCl.unit` is a class-level `ClassVar` fixed at `"uK_CMB^2"` — not a
constructor argument. Passing `unit=...` to the constructor raises
`TypeError`.

### Caching

`_load_cmb_templates(template_dir)` is `@lru_cache`-d on `(template_dir,
no_tensor_mtime_ns, r1_mtime_ns)`. Editing a template file changes its mtime
and therefore the cache key, so the next load picks up the new contents
automatically — no Python restart required.

### Examples

Default bundled templates, `r=0`, full lensing:

```python
from gaussky.ps import CMBCl

cmb = CMBCl()
cls = cmb.to_healpy_cls(lmax=200)
# cls[0] is TT, cls[1] is EE, cls[2] is BB, cls[3] is TE, cls[4]=cls[5]=0
```

Tensor-to-scalar ratio `r = 0.03`:

```python
cmb = CMBCl(a_lens=1.0, r_tensor=0.03)
```

Custom template directory (e.g. higher `lmax`):

```python
cmb = CMBCl(template_dir="/path/to/my/cmb_templates")
cls = cmb.to_healpy_cls(lmax=1500)
```

If you request `lmax` beyond what the bundled templates cover, `to_healpy_cls`
raises `ValueError: lmax=… exceeds the CMB template range (lmax=…)`.

## Writing your own spectrum

Anything with the right shape works:

```python
import numpy as np
from numpy.typing import NDArray

class WhiteNoiseCl:
    unit = "uK_CMB^2"

    def __init__(self, amplitude: float) -> None:
        self.amplitude = float(amplitude)

    def to_healpy_cls(self, lmax: int) -> list[NDArray[np.float64]]:
        flat = np.full(lmax + 1, self.amplitude, dtype=np.float64)
        zeros = np.zeros(lmax + 1, dtype=np.float64)
        return [flat, flat, flat, zeros, zeros, zeros]
```

Plug it into any SED-backed component or `GaussianCMB`. The sampling helpers
will run `validate_healpy_cls` on the output, so shape and PSD checks happen
automatically.
