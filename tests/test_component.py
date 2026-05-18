import numpy as np
import pytest

import gaussky.component as component
from gaussky.component import cmb, component_utils, dust, synchrotron
from gaussky.component.cmb import GaussianCMB
from gaussky.component.dust import SimpleModifiedBlackbodyDust
from gaussky.component.synchrotron import SimplePowerLawSynchrotron
from gaussky.ps import PowerLawCl


class FakeHealpy:
    def __init__(self):
        self.synfast_calls = []
        self.smoothing_calls = []
        self.reorder_calls = []

    def nside2npix(self, nside):
        return 12 * nside**2

    def synfast(self, cls, nside, alm=False, pol=True, new=True):
        self.synfast_calls.append(
            {"cls": cls, "nside": nside, "alm": alm, "pol": pol, "new": new}
        )
        npix = self.nside2npix(nside)
        pixels = np.arange(npix, dtype=np.float64)
        return np.vstack(
            [
                10.0 + pixels,
                20.0 + pixels,
                30.0 + pixels,
            ]
        )

    def smoothing(self, map_in, fwhm=0.0, pol=True):
        self.smoothing_calls.append({"fwhm": fwhm, "pol": pol})
        return np.asarray(map_in, dtype=np.float64) + 1000.0 * fwhm

    def reorder(self, maps, r2n=False):
        self.reorder_calls.append({"r2n": r2n})
        return np.asarray(maps, dtype=np.float64)[..., ::-1]


class RandomHealpy(FakeHealpy):
    def synfast(self, cls, nside, alm=False, pol=True, new=True):
        self.synfast_calls.append(
            {"cls": cls, "nside": nside, "alm": alm, "pol": pol, "new": new}
        )
        return np.random.standard_normal((3, self.nside2npix(nside)))


class FakeCMBPowerSpectrum:
    unit = "uK_CMB^2"
    a_lens = 0.75
    r_tensor = 0.05
    template_dir = "/fake/cmb_spec"

    def to_healpy_cls(self, lmax):
        auto = np.ones(lmax + 1, dtype=np.float64)
        cross = np.zeros(lmax + 1, dtype=np.float64)
        return [auto, auto, auto, cross, cross, cross]


def _power_spectrum(unit="uK_CMB^2"):
    return PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_bb=1.0, unit=unit)


def _synchrotron_component(ps=None, *, beta_s=-3.0, nu0_ghz=30.0):
    return SimplePowerLawSynchrotron(
        ps=_power_spectrum() if ps is None else ps,
        beta_s=beta_s,
        nu0_ghz=nu0_ghz,
    )


def _dust_component(ps=None, *, beta_d=1.6, temp_d=19.6, nu0_ghz=353.0):
    return SimpleModifiedBlackbodyDust(
        ps=_power_spectrum() if ps is None else ps,
        beta_d=beta_d,
        temp_d=temp_d,
        nu0_ghz=nu0_ghz,
    )


def _cmb_component(ps=None):
    return GaussianCMB(ps=FakeCMBPowerSpectrum() if ps is None else ps)


def _patch_healpy(monkeypatch):
    fake = FakeHealpy()
    monkeypatch.setattr(component_utils, "hp", fake)
    return fake


def test_component_package_exports_flat_public_api():
    assert component.GaussianCMB is GaussianCMB
    assert component.GaussianCMB is cmb.GaussianCMB
    assert component.GaussianComponent is synchrotron.GaussianComponent
    assert component.SimplePowerLawSynchrotron is SimplePowerLawSynchrotron
    assert component.SimpleModifiedBlackbodyDust is SimpleModifiedBlackbodyDust
    assert component.SimpleModifiedBlackbodyDust is dust.SimpleModifiedBlackbodyDust
    assert component.__all__ == [
        "GaussianCMB",
        "GaussianComponent",
        "SimpleModifiedBlackbodyDust",
        "SimplePowerLawSynchrotron",
    ]


def test_simple_power_law_synchrotron_samples_scaled_component_map(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _synchrotron_component()
    freqs = np.array([30.0, 90.0])

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=freqs,
        fields=("T", "Q", "U"),
        beam_fwhm_rad=0.1,
        ordering="RING",
        coord="G",
    )

    pixels = np.arange(12, dtype=np.float64)
    base_tqu = np.vstack([110.0 + pixels, 120.0 + pixels, 130.0 + pixels])
    expected = sampler.sed.scale(freqs)[:, None, None] * base_tqu[None, :, :]
    np.testing.assert_allclose(sampled.maps, expected)
    assert sampled.component_name == "synchrotron"
    assert sampled.unit == "uK_CMB"
    assert sampled.ordering == "RING"
    assert sampled.coord == "G"
    assert sampled.metadata == {"beta_s": -3.0, "nu0_ghz": 30.0, "seed": None}
    assert len(fake.synfast_calls) == 1
    synfast_call = fake.synfast_calls[0]
    assert synfast_call["nside"] == 1
    assert synfast_call["alm"] is False
    assert synfast_call["pol"] is True
    assert synfast_call["new"] is True
    for actual_cls, expected_cls in zip(
        synfast_call["cls"], sampler.ps.to_healpy_cls(2), strict=True
    ):
        np.testing.assert_allclose(actual_cls, expected_cls)
    assert fake.smoothing_calls == [{"fwhm": 0.1, "pol": True}]


def test_simple_power_law_synchrotron_preserves_requested_field_order(monkeypatch):
    _patch_healpy(monkeypatch)
    sampler = _synchrotron_component()

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[30.0],
        fields=("U", "Q"),
        beam_fwhm_rad=None,
    )

    np.testing.assert_allclose(sampled.maps[0, :, 0], [30.0, 20.0])
    assert sampled.fields == ("U", "Q")


def test_simple_power_law_synchrotron_defaults_to_pivot_frequency(monkeypatch):
    _patch_healpy(monkeypatch)
    sampler = _synchrotron_component()

    sampled = sampler.sample_map(nside=1, fields=("T",))

    pixels = np.arange(12, dtype=np.float64)
    np.testing.assert_allclose(sampled.freqs_ghz, [sampler.nu0_ghz])
    np.testing.assert_allclose(sampled.maps[0, 0], 10.0 + pixels)


def test_simple_power_law_synchrotron_supports_per_frequency_beams(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _synchrotron_component()
    beams = np.array([0.01, 0.02])
    freqs = np.array([30.0, 40.0])

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=freqs,
        fields=("T",),
        beam_fwhm_rad=beams,
    )

    pixels = np.arange(12, dtype=np.float64)
    expected_t = np.vstack([20.0 + pixels, 30.0 + pixels])
    expected = sampler.sed.scale(freqs)[:, None] * expected_t
    np.testing.assert_allclose(sampled.maps[:, 0, :], expected)
    assert fake.smoothing_calls == [
        {"fwhm": 0.01, "pol": True},
        {"fwhm": 0.02, "pol": True},
    ]


def test_simple_power_law_synchrotron_reorders_nested_output(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _synchrotron_component()

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[30.0],
        fields=("T",),
        ordering="NESTED",
    )

    np.testing.assert_allclose(sampled.maps[0, 0], 10.0 + np.arange(11, -1, -1))
    assert sampled.ordering == "NESTED"
    assert fake.reorder_calls == [{"r2n": True}]


def test_simple_power_law_synchrotron_rejects_unsupported_ps_unit():
    sampler = _synchrotron_component(ps=_power_spectrum(unit="K_RJ^2"))

    with pytest.raises(ValueError, match="Unsupported power-spectrum unit"):
        sampler.sample_map(nside=1, freqs_ghz=[30.0], fields=("T",))


def test_simple_power_law_synchrotron_rejects_non_psd_power_spectrum():
    sampler = _synchrotron_component(ps=PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_te=2.0))

    with pytest.raises(ValueError, match="positive semidefinite"):
        sampler.sample_map(nside=1, freqs_ghz=[30.0], fields=("T",))


def test_gaussian_cmb_samples_frequency_independent_map(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _cmb_component()

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[90.0, 150.0],
        fields=("T", "Q"),
        coord="G",
    )

    pixels = np.arange(12, dtype=np.float64)
    expected = np.vstack([10.0 + pixels, 20.0 + pixels])
    np.testing.assert_allclose(sampled.maps[0], expected)
    np.testing.assert_allclose(sampled.maps[1], expected)
    assert sampled.component_name == "cmb"
    assert sampled.unit == "uK_CMB"
    assert sampled.coord == "G"
    assert sampled.metadata == {
        "a_lens": 0.75,
        "r_tensor": 0.05,
        "template_dir": "/fake/cmb_spec",
        "seed": None,
    }
    assert len(fake.synfast_calls) == 1
    assert fake.smoothing_calls == []


def test_gaussian_cmb_smooths_scalar_beam_once_and_repeats(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _cmb_component()

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[90.0, 150.0],
        fields=("T",),
        beam_fwhm_rad=0.02,
    )

    pixels = np.arange(12, dtype=np.float64)
    expected = 30.0 + pixels
    np.testing.assert_allclose(sampled.maps[:, 0, :], np.vstack([expected, expected]))
    assert fake.smoothing_calls == [{"fwhm": 0.02, "pol": True}]


def test_gaussian_cmb_supports_per_frequency_beams(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _cmb_component()
    beams = np.array([0.01, 0.02])

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[90.0, 150.0],
        fields=("U", "Q"),
        beam_fwhm_rad=beams,
    )

    pixels = np.arange(12, dtype=np.float64)
    expected = np.array(
        [
            [40.0 + pixels, 30.0 + pixels],
            [50.0 + pixels, 40.0 + pixels],
        ]
    )
    np.testing.assert_allclose(sampled.maps, expected)
    assert fake.smoothing_calls == [
        {"fwhm": 0.01, "pol": True},
        {"fwhm": 0.02, "pol": True},
    ]


def test_gaussian_cmb_reorders_nested_output(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _cmb_component()

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=[90.0],
        fields=("T",),
        ordering="NESTED",
    )

    np.testing.assert_allclose(sampled.maps[0, 0], 10.0 + np.arange(11, -1, -1))
    assert sampled.ordering == "NESTED"
    assert fake.reorder_calls == [{"r2n": True}]


def test_simple_modified_blackbody_dust_samples_scaled_component_map(monkeypatch):
    fake = _patch_healpy(monkeypatch)
    sampler = _dust_component()
    freqs = np.array([353.0, 150.0])

    sampled = sampler.sample_map(
        nside=1,
        freqs_ghz=freqs,
        fields=("T", "Q"),
        beam_fwhm_rad=0.05,
        coord="G",
    )

    pixels = np.arange(12, dtype=np.float64)
    base_tq = np.vstack([60.0 + pixels, 70.0 + pixels])
    expected = sampler.sed.scale(freqs)[:, None, None] * base_tq[None, :, :]
    np.testing.assert_allclose(sampled.maps, expected)
    assert sampled.component_name == "dust"
    assert sampled.unit == "uK_CMB"
    assert sampled.coord == "G"
    assert sampled.metadata == {
        "beta_d": 1.6,
        "temp_d": 19.6,
        "nu0_ghz": 353.0,
        "seed": None,
    }
    assert fake.smoothing_calls == [{"fwhm": 0.05, "pol": True}]


def test_simple_modified_blackbody_dust_defaults_to_pivot_frequency(monkeypatch):
    _patch_healpy(monkeypatch)
    sampler = _dust_component()

    sampled = sampler.sample_map(nside=1, fields=("Q",))

    pixels = np.arange(12, dtype=np.float64)
    np.testing.assert_allclose(sampled.freqs_ghz, [sampler.nu0_ghz])
    np.testing.assert_allclose(sampled.maps[0, 0], 20.0 + pixels)


def test_simple_modified_blackbody_dust_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="temperature_k"):
        SimpleModifiedBlackbodyDust(
            ps=_power_spectrum(),
            beta_d=1.6,
            temp_d=0.0,
            nu0_ghz=353.0,
        )


def test_component_seed_reproducibility_and_metadata(monkeypatch):
    fake = RandomHealpy()
    monkeypatch.setattr(component_utils, "hp", fake)
    sampler = _cmb_component()

    maps1 = sampler.sample_map(nside=1, freqs_ghz=[90.0], fields=("T", "Q"), seed=42)
    maps2 = sampler.sample_map(nside=1, freqs_ghz=[90.0], fields=("T", "Q"), seed=42)
    maps3 = sampler.sample_map(nside=1, freqs_ghz=[90.0], fields=("T", "Q"), seed=99)

    np.testing.assert_array_equal(maps1.maps, maps2.maps)
    assert not np.array_equal(maps1.maps, maps3.maps)
    assert maps1.metadata["seed"] == 42


def test_component_seed_does_not_modify_global_numpy_state(monkeypatch):
    fake = RandomHealpy()
    monkeypatch.setattr(component_utils, "hp", fake)

    np.random.seed(123)
    expected = np.random.standard_normal(5)

    np.random.seed(123)
    _cmb_component().sample_map(nside=1, freqs_ghz=[90.0], fields=("T",), seed=42)
    actual = np.random.standard_normal(5)

    np.testing.assert_array_equal(actual, expected)


def test_component_rejects_invalid_seed(monkeypatch):
    _patch_healpy(monkeypatch)
    sampler = _dust_component()

    with pytest.raises(TypeError, match="seed"):
        sampler.sample_map(nside=1, fields=("T",), seed=True)
    with pytest.raises(ValueError, match="seed"):
        sampler.sample_map(nside=1, fields=("T",), seed=-1)
    with pytest.raises(ValueError, match="seed"):
        sampler.sample_map(nside=1, fields=("T",), seed=2**32)
