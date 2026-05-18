"""Cross-cutting auxiliary-template loaders.

This package holds loaders for auxiliary HEALPix templates that are reused
across multiple component implementations -- for example, a published
spatial template of the dust spectral index ``beta_d`` that the simple,
spatially-varying, and decorrelated dust components all want to consume.

The package is intentionally separate from :mod:`gaussky.component`:

- a :class:`~gaussky.map.AuxiliaryHealpixMap` is the **output** container
  for an auxiliary product attached to a sampled component map and lives in
  :mod:`gaussky.map`;
- model code that *consumes* an auxiliary template lives **with the
  component** that needs it (e.g. a varying-beta dust component owns its
  ``beta_d_map`` field and republishes it through
  :attr:`~gaussky.map.MultiFreqCompMap.auxiliary_maps`);
- loaders for **shared input templates** (this package) decouple "I want
  the standard Planck beta_d template" from "I want a dust component
  driven by it." Bundled data files belong in ``gaussky/data/templates/``.

Loaders are not yet implemented. This package exists so the
spatially-varying foreground components introduced in later refactor steps
have a clear home for their template I/O.
"""

from __future__ import annotations

__all__: list[str] = []
