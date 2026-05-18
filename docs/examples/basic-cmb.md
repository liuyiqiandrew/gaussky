# Example: basic CMB realization

This walkthrough produces a multi-frequency Gaussian CMB realization using
the bundled CAMB templates and inspects the resulting container.

## Goal

- Sample one CMB realization at `nside=64`.
- Get T, Q, U maps at three frequency channels.
- Inspect metadata, units, and array shapes.

## Code

```python
import numpy as np

from gaussky.component import GaussianCMB
from gaussky.ps import CMBCl
from gaussky.sampler import Sampler

# Default templates ship with the package; a_lens=1.0 and r_tensor=0.0.
cmb = GaussianCMB(ps=CMBCl())

sampler = Sampler(nside=64)
sky = sampler.sample(
    cmb,
    fields=("T", "Q", "U"),
    freqs_ghz=np.array([90.0, 150.0, 220.0]),
    beam_fwhm_rad=np.deg2rad([0.5, 0.3, 0.2]),
    coord="G",
)

print("type:           ", type(sky).__name__)        # MultiFreqCompMap
print("component_name: ", sky.component_name)        # 'cmb'
print("unit:           ", sky.unit)                  # 'uK_CMB'
print("maps.shape:     ", sky.maps.shape)            # (3, 3, 49152)
print("freqs_ghz:      ", sky.freqs_ghz)             # [ 90. 150. 220.]
print("fields:         ", sky.fields)                # ('T', 'Q', 'U')
print("metadata:       ", dict(sky.metadata))
```

## Why the same realization appears at every frequency

CMB is intrinsically frequency-independent in thermodynamic units, so the
sampler draws **one** T/Q/U map and repeats it along the frequency axis.
Per-channel beam smoothing still differentiates the channels:

```python
T_90 = sky.select_freq(90.0).select_field("T").maps[0, 0]
T_150 = sky.select_freq(150.0).select_field("T").maps[0, 0]

# Without beam smoothing T_90 and T_150 would be identical.
np.allclose(T_90, T_150)  # False, because each channel uses a different beam
```

## Tensor-to-scalar ratio

`CMBCl(a_lens=1.0, r_tensor=0.0)` gives a lensed ΛCDM CMB. Increase
`r_tensor` to add tensor B-modes:

```python
cmb_r03 = GaussianCMB(ps=CMBCl(a_lens=1.0, r_tensor=0.03))
sky_r03 = sampler.sample(cmb_r03, freqs_ghz=[150.0], fields=("Q", "U"))
sky_r03.metadata["r_tensor"]  # 0.03
```

The `D_ell` combination used is
`a_lens * D_lens-nobb + r_tensor * (D_r1 - D_lens-nobb)`, which is then
converted to `C_ell` before passing to `synfast`.

## Slicing the result

```python
Q_150 = sky.select_freq(150.0).select_field("Q")
Q_150.maps.shape                    # (1, 1, 49152)
Q_150.beam_fwhm_rad                 # only the 150 GHz beam survives
```

All `select_*` and `copy_with` calls revalidate, so the resulting container is
still a fully-formed `MultiFreqCompMap` you can pass anywhere a signal map is
accepted.

## Where to go next

- Add foregrounds to the same scene → [Foregrounds](foregrounds.md).
- Sum components into a total map → [Total map](total-map.md).
- Custom CMB templates: pass `template_dir=...` to `CMBCl`. See
  [`gaussky.ps`](../reference/ps.md).
