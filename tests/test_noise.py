"""Tests for the ``NoiseComponent`` protocol and ``WhiteNoise``."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from gaussky import Sampler, WhiteNoise
from gaussky.component.base import GaussianComponent, NoiseComponent
from gaussky.map import MultiFreqCompMap, MultiFreqTotalMap


def _pixel_area_arcmin2(nside: int) -> float:
    """Return the HEALPix pixel area in arcmin^2."""
    return float(hp.nside2pixarea(nside, degrees=True)) * 3600.0


def test_white_noise_satisfies_noise_protocol_only() -> None:
    """``WhiteNoise`` is a noise component, never a signal component."""
    noise = WhiteNoise(sigma_uK_arcmin=1.0)
    assert isinstance(noise, NoiseComponent)
    assert not isinstance(noise, GaussianComponent)


def test_white_noise_rejects_invalid_sigma() -> None:
    """Non-finite, non-positive, or multi-dim sigmas raise at construction."""
    with pytest.raises(ValueError, match="finite positive"):
        WhiteNoise(sigma_uK_arcmin=0.0)
    with pytest.raises(ValueError, match="finite positive"):
        WhiteNoise(sigma_uK_arcmin=-1.0)
    with pytest.raises(ValueError, match="finite positive"):
        WhiteNoise(sigma_uK_arcmin=np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        WhiteNoise(sigma_uK_arcmin=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_white_noise_scalar_sigma_broadcasts_across_channels() -> None:
    """A scalar sigma yields independent draws with matching variance per channel."""
    noise = WhiteNoise(sigma_uK_arcmin=10.0)
    sampled = noise.sample_noise_map(
        nside=16,
        fields=("T", "Q", "U"),
        freqs_ghz=[30.0, 90.0, 150.0],
        seed=42,
    )

    assert isinstance(sampled, MultiFreqCompMap)
    assert sampled.maps.shape == (3, 3, hp.nside2npix(16))
    assert sampled.unit == "uK_CMB"
    # The same realization is *not* shared across channels.
    assert not np.array_equal(sampled.maps[0], sampled.maps[1])
    assert not np.array_equal(sampled.maps[1], sampled.maps[2])


def test_white_noise_array_sigma_must_match_freq_count() -> None:
    """A 1-D sigma whose length differs from ``freqs_ghz`` raises."""
    noise = WhiteNoise(sigma_uK_arcmin=np.array([4.0, 2.0]))
    with pytest.raises(ValueError, match="sigma_uK_arcmin length must match"):
        noise.sample_noise_map(
            nside=4,
            fields=("T",),
            freqs_ghz=[30.0, 90.0, 150.0],
        )


def test_white_noise_pixel_variance_matches_sigma() -> None:
    """Sampled variance ≈ sigma_uK_arcmin² / pixel_area_arcmin² across pixels."""
    nside = 32
    sigma_uK_arcmin = 5.0
    expected_var = sigma_uK_arcmin**2 / _pixel_area_arcmin2(nside)

    noise = WhiteNoise(sigma_uK_arcmin=sigma_uK_arcmin)
    sampled = noise.sample_noise_map(
        nside=nside,
        fields=("T",),
        freqs_ghz=[150.0],
        seed=2025,
    )
    observed_var = float(np.var(sampled.maps[0, 0, :]))
    # 12288 pixels — std-dev of variance estimate is ~ var * sqrt(2/N) ≈ 1.3%.
    assert observed_var == pytest.approx(expected_var, rel=0.05)


def test_white_noise_per_channel_sigma_applies_per_channel() -> None:
    """A per-channel sigma array produces matching per-channel variance."""
    nside = 32
    sigmas = np.array([2.0, 4.0])
    noise = WhiteNoise(sigma_uK_arcmin=sigmas)
    sampled = noise.sample_noise_map(
        nside=nside,
        fields=("T",),
        freqs_ghz=[30.0, 90.0],
        seed=11,
    )

    pixel_area = _pixel_area_arcmin2(nside)
    expected = sigmas**2 / pixel_area
    observed = np.var(sampled.maps[:, 0, :], axis=-1)
    np.testing.assert_allclose(observed, expected, rtol=0.05)


def test_white_noise_is_not_beam_smoothed() -> None:
    """``beam_fwhm_rad`` is stored only — same seed ⇒ bitwise-equal draws.

    This is the structural enforcement that ``sample_noise_map`` never
    calls ``hp.smoothing``. Lands the G2 invariant alongside B1.
    """
    noise = WhiteNoise(sigma_uK_arcmin=3.0)
    unsmoothed = noise.sample_noise_map(
        nside=8,
        fields=("T", "Q", "U"),
        freqs_ghz=[30.0, 90.0],
        beam_fwhm_rad=None,
        seed=7,
    )
    with_beam = noise.sample_noise_map(
        nside=8,
        fields=("T", "Q", "U"),
        freqs_ghz=[30.0, 90.0],
        beam_fwhm_rad=0.05,
        seed=7,
    )

    np.testing.assert_array_equal(unsmoothed.maps, with_beam.maps)
    assert unsmoothed.beam_fwhm_rad is None
    assert with_beam.beam_fwhm_rad == 0.05


def test_white_noise_reorder_ring_to_nested_preserves_variance() -> None:
    """RING→NESTED is a permutation of pixels, so variance is preserved."""
    noise = WhiteNoise(sigma_uK_arcmin=2.0)
    ring = noise.sample_noise_map(nside=8, fields=("T",), freqs_ghz=[150.0], seed=99)
    nested = noise.sample_noise_map(
        nside=8,
        fields=("T",),
        freqs_ghz=[150.0],
        seed=99,
        ordering="NESTED",
    )

    assert nested.ordering == "NESTED"
    np.testing.assert_allclose(
        np.var(ring.maps[0, 0, :]),
        np.var(nested.maps[0, 0, :]),
    )


def test_white_noise_metadata_records_sigma_and_seed() -> None:
    """``MultiFreqCompMap.metadata`` carries the sigma and resolved seed."""
    noise_scalar = WhiteNoise(sigma_uK_arcmin=4.0)
    sampled = noise_scalar.sample_noise_map(
        nside=4, fields=("T",), freqs_ghz=[90.0], seed=33
    )
    assert sampled.metadata["sigma_uK_arcmin"] == 4.0
    assert sampled.metadata["seed"] == 33

    noise_per_ch = WhiteNoise(sigma_uK_arcmin=np.array([4.0, 2.0]))
    sampled_per_ch = noise_per_ch.sample_noise_map(
        nside=4, fields=("T",), freqs_ghz=[30.0, 90.0]
    )
    assert sampled_per_ch.metadata["sigma_uK_arcmin"] == (4.0, 2.0)


def test_sampler_dispatches_noise_through_sample_noise_map() -> None:
    """``Sampler.sample(WhiteNoise(...))`` returns a noise-component map."""
    sampler = Sampler(nside=4, fields=("T",), freqs_ghz=[150.0])
    noise = WhiteNoise(sigma_uK_arcmin=1.5)
    sampled = sampler.sample(noise, seed=2026)

    assert isinstance(sampled, MultiFreqCompMap)
    assert sampled.component_name == "white_noise"
    assert sampled.unit == "uK_CMB"


def test_sampler_mixed_signal_and_noise_list_returns_total_map() -> None:
    """Mixed lists route by protocol kind and produce a total map."""
    from gaussky import GaussianCMB

    sampler = Sampler(
        nside=4,
        fields=("T", "Q", "U"),
        freqs_ghz=[30.0, 90.0],
        seed=2026,
    )
    cmb = GaussianCMB()
    noise = WhiteNoise(sigma_uK_arcmin=np.array([4.0, 2.0]))

    total = sampler.sample([cmb, noise])
    assert isinstance(total, MultiFreqTotalMap)
    assert total.component_names == ("cmb", "white_noise")
    assert total.unit == "uK_CMB"

    expected = total.component("cmb").maps + total.component("white_noise").maps
    np.testing.assert_array_equal(total.maps, expected)
