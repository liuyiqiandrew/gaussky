"""End-to-end integration tests against the real Sampler + components.

These tests run the full pipeline with no monkey-patching at a small
``nside`` to keep the suite fast. They cover:

- multi-component sampling (CMB + synchrotron + dust + white noise)
- per-component seed determinism and root-seed reproducibility
- the noise-vs-beam invariant (G2)
"""

from __future__ import annotations

import healpy as hp
import numpy as np

from gaussky import (
    GaussianCMB,
    Sampler,
    SimpleModifiedBlackbodyDust,
    SimplePowerLawSynchrotron,
    WhiteNoise,
)
from gaussky.map import MultiFreqTotalMap
from gaussky.ps import PowerLawCl


_FREQS_GHZ = (30.0, 90.0, 150.0, 220.0, 353.0)
_BEAMS_RAD = np.deg2rad([0.5, 0.3, 0.2, 0.15, 0.1])


def _build_scene() -> tuple[Sampler, list[object]]:
    """Build a 5-band sky scene with CMB, foregrounds, and noise."""
    sampler = Sampler(
        nside=32,
        fields=("T", "Q", "U"),
        freqs_ghz=np.asarray(_FREQS_GHZ, dtype=np.float64),
        beam_fwhm_rad=_BEAMS_RAD,
        seed=2026,
    )
    components: list[object] = [
        GaussianCMB(),
        SimplePowerLawSynchrotron(
            ps=PowerLawCl(amp_ee=20.0, alpha_ee=-3.0, amp_bb=4.0, alpha_bb=-3.0),
            beta_s=-3.1,
            nu0_ghz=23.0,
        ),
        SimpleModifiedBlackbodyDust(
            ps=PowerLawCl(amp_ee=70.0, alpha_ee=-2.4, amp_bb=10.0, alpha_bb=-2.4),
            beta_d=1.6,
            temp_d=19.6,
            nu0_ghz=353.0,
        ),
        WhiteNoise(sigma_uK_arcmin=np.array([4.0, 2.0, 1.5, 2.0, 3.0])),
    ]
    return sampler, components


def test_end_to_end_total_map_shape_and_metadata() -> None:
    """Full pipeline returns a 5-band T/Q/U total map with provenance."""
    sampler, components = _build_scene()
    total = sampler.sample(components)

    assert isinstance(total, MultiFreqTotalMap)
    assert total.maps.shape == (5, 3, hp.nside2npix(32))
    assert total.unit == "uK_CMB"
    assert total.metadata["seed"] == 2026
    assert set(total.metadata["component_seeds"].keys()) == {
        "cmb",
        "synchrotron",
        "dust",
        "white_noise",
    }
    assert total.component_names == (
        "cmb",
        "synchrotron",
        "dust",
        "white_noise",
    )


def test_end_to_end_total_equals_component_sum() -> None:
    """The reported total equals the sum of the per-component maps."""
    sampler, components = _build_scene()
    total = sampler.sample(components)

    expected = sum(component.maps for component in total.components)
    np.testing.assert_array_equal(total.maps, expected)


def test_end_to_end_root_seed_reproducibility() -> None:
    """Two sample calls with the same root seed produce identical maps."""
    sampler_a, components_a = _build_scene()
    sampler_b, components_b = _build_scene()
    a = sampler_a.sample(components_a)
    b = sampler_b.sample(components_b)
    np.testing.assert_array_equal(a.maps, b.maps)


def test_end_to_end_different_root_seed_yields_different_realization() -> None:
    """``Sampler.with_(seed=...)`` produces a fresh realization."""
    sampler, components = _build_scene()
    a = sampler.sample(components)
    b = sampler.with_(seed=99).sample(components)
    assert not np.array_equal(a.maps, b.maps)


# --- G2: noise-vs-beam regression -------------------------------------------


def test_signal_path_is_still_beam_smoothed() -> None:
    """``GaussianCMB.sample_map`` differs with vs without a beam at the same seed.

    Complement of :func:`test_white_noise_is_not_beam_smoothed` in
    ``test_noise.py``: confirms the signal path *is* beam-smoothed (we did
    not accidentally lobotomize it while structurally separating the noise
    path in B1).
    """
    cmb = GaussianCMB()
    unsmoothed = cmb.sample_map(
        nside=8,
        fields=("T", "Q", "U"),
        freqs_ghz=[150.0],
        beam_fwhm_rad=None,
        seed=2026,
    )
    with_beam = cmb.sample_map(
        nside=8,
        fields=("T", "Q", "U"),
        freqs_ghz=[150.0],
        beam_fwhm_rad=np.deg2rad(0.5),
        seed=2026,
    )
    assert not np.array_equal(unsmoothed.maps, with_beam.maps)
