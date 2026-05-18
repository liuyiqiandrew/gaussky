import numpy as np
import pytest

from gaussky.sed import (
    ModifiedBlackbodySED,
    PowerLawSED,
    planck_rj_spectrum,
    tcmb_to_trj,
    trj_to_tcmb,
)


def test_power_law_sed_is_normalized_at_reference_frequency():
    sed = PowerLawSED(beta=-3.0, nu0_ghz=30.0)

    np.testing.assert_allclose(sed.scale(30.0), 1.0)


def test_modified_blackbody_sed_is_normalized_at_reference_frequency():
    sed = ModifiedBlackbodySED(beta=1.6, temperature_k=19.6, nu0_ghz=353.0)

    np.testing.assert_allclose(sed.scale(353.0), 1.0)


def test_temperature_conversion_factors_are_inverses():
    freq_ghz = np.array([23.0, 93.0, 150.0, 353.0])

    np.testing.assert_allclose(
        trj_to_tcmb(freq_ghz) * tcmb_to_trj(freq_ghz),
        np.ones_like(freq_ghz),
    )


def test_temperature_conversion_uses_legacy_pygsm_tcmb():
    freq_ghz = np.array([23.0, 93.0, 150.0, 353.0])
    h_planck = 6.62607015e-34
    k_boltzmann = 1.380649e-23
    t_cmb = 2.725
    x = h_planck * freq_ghz * 1e9 / t_cmb / k_boltzmann

    expected_trj_to_tcmb = (np.exp(x) - 1.0) ** 2 / x**2 / np.exp(x)

    np.testing.assert_allclose(trj_to_tcmb(freq_ghz), expected_trj_to_tcmb)


def test_temperature_conversions_use_expm1_for_low_frequency_stability():
    # At low GHz, `exp(x) - 1.0` loses precision via catastrophic cancellation
    # while `expm1(x)` retains it. Pin the implementation to the expm1 form by
    # asserting bitwise agreement with an expm1-based reference at frequencies
    # where the difference from the naive form is well above machine epsilon.
    freq_ghz = np.array([1e-4, 1e-2, 1.0])
    h_planck = 6.62607015e-34
    k_boltzmann = 1.380649e-23
    t_cmb = 2.725
    x = h_planck * freq_ghz * 1e9 / (k_boltzmann * t_cmb)

    expected_trj = freq_ghz / np.expm1(x)
    expected_trj_to_tcmb = np.expm1(x) ** 2 / (x**2 * np.exp(x))
    expected_tcmb_to_trj = x**2 * np.exp(x) / np.expm1(x) ** 2

    np.testing.assert_array_equal(planck_rj_spectrum(t_cmb, freq_ghz), expected_trj)
    np.testing.assert_array_equal(trj_to_tcmb(freq_ghz), expected_trj_to_tcmb)
    np.testing.assert_array_equal(tcmb_to_trj(freq_ghz), expected_tcmb_to_trj)


def test_power_law_sed_scales_maps_with_staged_unit_conversions():
    sed = PowerLawSED(beta=-3.0, nu0_ghz=23.0)
    freq_ghz = np.array([30.0, 93.0])
    maps = np.arange(12, dtype=np.float64).reshape(2, 2, 3)

    expected = maps.copy()
    expected *= tcmb_to_trj(sed.nu0_ghz)
    expected *= ((freq_ghz / sed.nu0_ghz) ** sed.beta)[:, None, None]
    expected *= trj_to_tcmb(freq_ghz)[:, None, None]

    np.testing.assert_array_equal(sed.scale_maps(maps, freq_ghz), expected)


def test_modified_blackbody_sed_scales_cls_with_staged_unit_conversions():
    sed = ModifiedBlackbodySED(beta=1.54, temperature_k=19.6, nu0_ghz=353.0)
    freq_ghz = np.array([93.0, 150.0])
    cls = np.arange(8, dtype=np.float64).reshape(2, 4)

    rj_scaling = (freq_ghz / sed.nu0_ghz) ** sed.beta * (
        planck_rj_spectrum(sed.temperature_k, freq_ghz)
        / planck_rj_spectrum(sed.temperature_k, sed.nu0_ghz)
    )
    expected = np.tile(cls * tcmb_to_trj(sed.nu0_ghz) ** 2, (freq_ghz.size, 1, 1))
    expected *= rj_scaling[:, None, None] ** 2
    expected *= trj_to_tcmb(freq_ghz)[:, None, None] ** 2

    np.testing.assert_array_equal(sed.scale_cls(cls, freq_ghz), expected)


def test_sed_rejects_non_positive_frequency():
    sed = PowerLawSED(beta=-3.0, nu0_ghz=30.0)

    with pytest.raises(ValueError, match="freq_ghz"):
        sed.scale(0.0)


def test_sed_rejects_non_positive_reference_frequency():
    with pytest.raises(ValueError, match="nu0_ghz"):
        PowerLawSED(beta=-3.0, nu0_ghz=0.0)


def test_public_sed_helpers_use_shared_frequency_validation():
    with pytest.raises(ValueError, match="freq_ghz"):
        trj_to_tcmb([-1.0, 30.0])

    with pytest.raises(ValueError, match="freq_ghz"):
        planck_rj_spectrum(20.0, 0.0)

    with pytest.raises(ValueError, match="freq_ghz"):
        PowerLawSED(beta=-3.0, nu0_ghz=30.0).scale(np.nan)


def test_base_sed_cannot_be_instantiated_without_rj_scaling():
    """Subclasses of ``BaseSED`` must implement ``_rj_scaling`` (it's @abstractmethod)."""
    from dataclasses import dataclass

    from gaussky.sed.common import BaseSED

    @dataclass(frozen=True)
    class IncompleteSED(BaseSED):
        nu0_ghz: float = 30.0

    with pytest.raises(TypeError, match="abstract"):
        IncompleteSED()


def test_base_sed_subclass_with_rj_scaling_works_end_to_end():
    """A minimal subclass implementing ``_rj_scaling`` satisfies the protocol."""
    from dataclasses import dataclass

    from numpy.typing import NDArray

    from gaussky.sed.common import BaseSED

    @dataclass(frozen=True)
    class FlatSED(BaseSED):
        """Identity SED — useful for spec-conformance tests."""

        nu0_ghz: float = 30.0

        def _rj_scaling(self, freq_ghz: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.ones_like(freq_ghz)

    sed = FlatSED()
    # The Protocol surface (scale, scale_maps, scale_cls) all work.
    np.testing.assert_allclose(sed.scale(30.0), 1.0)
    maps = np.ones((2, 1, 4), dtype=np.float64)
    out = sed.scale_maps(maps, np.array([30.0, 60.0]))
    assert out.shape == maps.shape

    cls = np.ones((1, 4), dtype=np.float64)
    out_cls = sed.scale_cls(cls, np.array([30.0, 60.0]))
    assert out_cls.shape == (2, 1, 4)
