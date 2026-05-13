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
