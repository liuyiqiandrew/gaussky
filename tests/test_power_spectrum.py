import numpy as np
import pytest

from gaussky.conventions import HEALPY_POLARIZED_ORDER as SHARED_HEALPY_ORDER
from gaussky.ps import (
    HEALPY_POLARIZED_ORDER,
    ClSpectra,
    PowerLawCl,
    cl_spectra_from_model,
)


def test_power_spectrum_exports_shared_pair_ordering():
    assert HEALPY_POLARIZED_ORDER == SHARED_HEALPY_ORDER


def test_power_law_cl_is_normalized_at_reference_multipole():
    spectrum = PowerLawCl(amp_ee=2.0, alpha_ee=-2.4, ell0=80.0)

    np.testing.assert_allclose(spectrum.cl("EE", 80.0), 2.0)


def test_power_law_cl_zeros_low_multipoles():
    spectrum = PowerLawCl(amp_tt=1.0, alpha_tt=-2.0, ell0=80.0)

    np.testing.assert_allclose(
        spectrum.cl("TT", np.array([0, 1, 2])),
        [0.0, 0.0, 1600.0],
    )


def test_to_healpy_cls_uses_polarized_ordering():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=2.0, amp_bb=3.0, ell0=2.0)
    cls = spectrum.to_healpy_cls(lmax=3)

    assert len(cls) == len(HEALPY_POLARIZED_ORDER)
    np.testing.assert_allclose(cls[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(cls[1], [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(cls[2], [0.0, 0.0, 3.0, 3.0])


def test_cl_spectra_names_spectra_and_adapts_to_healpy_ordering():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=2.0, amp_bb=3.0, ell0=2.0)
    spectra = spectrum.to_cl_spectra(lmax=3)

    assert spectra.pairs == HEALPY_POLARIZED_ORDER
    assert spectra.unit == spectrum.unit
    np.testing.assert_allclose(spectra.ells, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(spectra.cls_for("EE"), [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(spectra.to_healpy_cls()[2], [0.0, 0.0, 3.0, 3.0])
    assert not spectra.ells.flags.writeable
    assert not spectra.values.flags.writeable


def test_cl_spectra_from_model_can_select_pair_subset():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=2.0, ell0=2.0)
    spectra = cl_spectra_from_model(spectrum, lmax=2, pairs=("TT", "EE"))

    assert spectra.pairs == ("TT", "EE")
    np.testing.assert_allclose(spectra.values[:, 2], [1.0, 2.0])


def test_cl_spectra_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="values"):
        ClSpectra(
            ells=np.arange(3),
            pairs=("TT", "EE"),
            values=np.zeros((1, 3)),
            unit="uK_CMB^2",
        )


def test_invalid_pair_raises_clear_error():
    spectrum = PowerLawCl()

    with pytest.raises(ValueError, match="Unknown spectrum pair"):
        spectrum.cl("VV", 80.0)


def test_negative_auto_spectrum_amplitude_is_rejected():
    with pytest.raises(ValueError, match="amp_tt"):
        PowerLawCl(amp_tt=-1.0)


def test_negative_multipole_is_rejected():
    spectrum = PowerLawCl()

    with pytest.raises(ValueError, match="ell"):
        spectrum.cl("TT", -1)


def test_positive_semidefinite_validation_rejects_invalid_cross_spectrum():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=1.0, amp_te=2.0)

    with pytest.raises(ValueError, match="positive semidefinite"):
        spectrum.validate_positive_semidefinite([80.0])
