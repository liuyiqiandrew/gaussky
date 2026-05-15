import numpy as np
import pytest

from gaussky.map import MultiFreqCompMap, MultiFreqTotalMap
from gaussky.sampler import Sampler


class FakeComponent:
    def __init__(self, name, value, default_freq=42.0):
        self.name = name
        self.value = value
        self.default_freq = default_freq
        self.sample_calls = []

    def sample_map(
        self,
        *,
        nside,
        fields,
        freqs_ghz=None,
        beam_fwhm_rad=None,
        ordering="RING",
        coord=None,
    ):
        freqs = np.asarray(
            [self.default_freq] if freqs_ghz is None else freqs_ghz,
            dtype=np.float64,
        )
        maps = np.full(
            (freqs.size, len(fields), 12 * nside**2),
            self.value,
            dtype=np.float64,
        )
        self.sample_calls.append(
            {
                "nside": nside,
                "freqs_ghz": freqs,
                "raw_freqs_ghz": freqs_ghz,
                "fields": fields,
                "beam_fwhm_rad": beam_fwhm_rad,
                "ordering": ordering,
                "coord": coord,
            }
        )
        return MultiFreqCompMap(
            component_name=self.name,
            freqs_ghz=freqs,
            fields=fields,
            unit="uK_CMB",
            maps=maps,
            nside=nside,
            beam_fwhm_rad=beam_fwhm_rad,
            ordering=ordering,
            coord=coord,
        )


def test_sampler_samples_single_component_map():
    component = FakeComponent("sync", 2.0)

    sampled = Sampler(nside=1).sample(
        component,
        freqs_ghz=[30.0, 90.0],
        fields=("Q", "U"),
        beam_fwhm_rad=0.01,
        ordering="NESTED",
        coord="G",
    )

    assert isinstance(sampled, MultiFreqCompMap)
    assert sampled.component_name == "sync"
    np.testing.assert_allclose(sampled.maps, 2.0)
    assert component.sample_calls[0]["nside"] == 1
    assert component.sample_calls[0]["fields"] == ("Q", "U")
    assert component.sample_calls[0]["beam_fwhm_rad"] == 0.01
    assert component.sample_calls[0]["ordering"] == "NESTED"
    assert component.sample_calls[0]["coord"] == "G"
    np.testing.assert_allclose(component.sample_calls[0]["freqs_ghz"], [30.0, 90.0])


def test_sampler_samples_total_map_from_component_list():
    sync = FakeComponent("sync", 2.0)
    dust = FakeComponent("dust", 3.0)

    sampled = Sampler(nside=1).sample(
        [sync, dust],
        freqs_ghz=[30.0, 90.0],
        fields=("T", "Q"),
        beam_fwhm_rad=[0.01, 0.02],
        coord="G",
    )

    assert isinstance(sampled, MultiFreqTotalMap)
    assert sampled.component_names == ("sync", "dust")
    np.testing.assert_allclose(sampled.maps, 5.0)
    assert sampled.components[0] is not sampled.components[1]
    assert len(sync.sample_calls) == 1
    assert len(dust.sample_calls) == 1


def test_sampler_allows_component_default_frequencies():
    component = FakeComponent("sync", 1.0)

    sampled = Sampler(nside=1).sample(component, fields=("T",))

    np.testing.assert_allclose(sampled.freqs_ghz, [42.0])
    assert component.sample_calls[0]["raw_freqs_ghz"] is None


def test_sampler_rejects_empty_or_invalid_component_list():
    sampler = Sampler(nside=1)

    with pytest.raises(ValueError, match="at least one component"):
        sampler.sample([], freqs_ghz=[30.0])

    with pytest.raises(TypeError, match="GaussianComponent"):
        sampler.sample([FakeComponent("sync", 1.0), object()], freqs_ghz=[30.0])
