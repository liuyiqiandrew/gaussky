import numpy as np
import pytest

from gaussky.conventions import HEALPY_POLARIZED_ORDER as SHARED_HEALPY_ORDER
from gaussky.ps import (
    CMBCl,
    HEALPY_POLARIZED_ORDER,
    PowerLawCl,
    validate_healpy_cls,
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


def test_power_law_cl_converts_dl_amplitudes_to_cl():
    spectrum = PowerLawCl(amp_ee=56.0, alpha_ee=-0.32, ell0=80.0, is_cell=False)

    ells = np.array([0.0, 1.0, 2.0, 80.0])
    expected = np.array(
        [
            0.0,
            0.0,
            56.0 * (2.0 / 80.0) ** -0.32 * 2.0 * np.pi / (2.0 * 3.0),
            56.0 * 2.0 * np.pi / (80.0 * 81.0),
        ]
    )

    np.testing.assert_allclose(spectrum.cl("EE", ells), expected)


def test_to_healpy_cls_uses_polarized_ordering():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=2.0, amp_bb=3.0, ell0=2.0)
    cls = spectrum.to_healpy_cls(lmax=3)

    assert len(cls) == len(HEALPY_POLARIZED_ORDER)
    np.testing.assert_allclose(cls[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(cls[1], [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(cls[2], [0.0, 0.0, 3.0, 3.0])


def test_validate_healpy_cls_returns_readonly_normalized_spectra():
    spectrum = PowerLawCl(amp_tt=1.0, amp_ee=2.0, amp_bb=3.0, ell0=2.0)
    cls = validate_healpy_cls(spectrum.to_healpy_cls(lmax=3), lmax=3)

    assert len(cls) == len(HEALPY_POLARIZED_ORDER)
    np.testing.assert_allclose(cls[1], [0.0, 0.0, 2.0, 2.0])
    assert all(not values.flags.writeable for values in cls)


def test_validate_healpy_cls_rejects_incompatible_spectra():
    with pytest.raises(ValueError, match="six spectra"):
        validate_healpy_cls([np.zeros(3)], lmax=2)

    with pytest.raises(ValueError, match="length"):
        validate_healpy_cls([np.zeros(2) for _ in HEALPY_POLARIZED_ORDER], lmax=2)

    bad_cls = [np.zeros(3) for _ in HEALPY_POLARIZED_ORDER]
    bad_cls[0][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_healpy_cls(bad_cls, lmax=2)


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
        validate_healpy_cls(spectrum.to_healpy_cls(lmax=80), lmax=80)


def _write_cmb_template(path, rows):
    np.savetxt(path, np.asarray(rows, dtype=np.float64))


def test_cmb_cl_combines_templates_and_zeros_tb_eb(tmp_path):
    _write_cmb_template(
        tmp_path / "camb_lens_nobb.dat",
        [
            [1, 10.0, 20.0, 30.0, 40.0],
            [2, 12.0, 22.0, 32.0, 42.0],
            [3, 14.0, 24.0, 34.0, 44.0],
        ],
    )
    _write_cmb_template(
        tmp_path / "camb_lens_r1.dat",
        [
            [1, 100.0, 200.0, 300.0, 400.0],
            [2, 120.0, 220.0, 320.0, 420.0],
            [3, 140.0, 240.0, 340.0, 440.0],
        ],
    )

    cls = CMBCl(a_lens=0.5, r_tensor=0.25, template_dir=tmp_path).to_healpy_cls(3)

    assert len(cls) == len(HEALPY_POLARIZED_ORDER)
    scale = np.array([0.0, 0.0, 2.0 * np.pi / 6.0, 2.0 * np.pi / 12.0])
    expected_tt_dl = np.array([0.0, 0.5 * 10.0 + 0.25 * 90.0, 33.0, 38.5])
    expected_te_dl = np.array([0.0, 0.5 * 40.0 + 0.25 * 360.0, 115.5, 121.0])
    np.testing.assert_allclose(
        cls[HEALPY_POLARIZED_ORDER.index("TT")], expected_tt_dl * scale
    )
    np.testing.assert_allclose(
        cls[HEALPY_POLARIZED_ORDER.index("TE")], expected_te_dl * scale
    )
    np.testing.assert_allclose(cls[HEALPY_POLARIZED_ORDER.index("EB")], 0.0)
    np.testing.assert_allclose(cls[HEALPY_POLARIZED_ORDER.index("TB")], 0.0)


def test_cmb_cl_rejects_invalid_parameters_and_lmax(tmp_path):
    _write_cmb_template(tmp_path / "camb_lens_nobb.dat", [[1, 1.0, 1.0, 1.0, 0.0]])
    _write_cmb_template(tmp_path / "camb_lens_r1.dat", [[1, 1.0, 1.0, 1.0, 0.0]])

    with pytest.raises(ValueError, match="a_lens"):
        CMBCl(a_lens=-1.0, template_dir=tmp_path)
    with pytest.raises(ValueError, match="r_tensor"):
        CMBCl(r_tensor=-1.0, template_dir=tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        CMBCl(template_dir=tmp_path).to_healpy_cls(2)


def test_cmb_cl_loads_bundled_default_templates():
    cls = CMBCl().to_healpy_cls(10)

    assert len(cls) == len(HEALPY_POLARIZED_ORDER)
    assert all(values.shape == (11,) for values in cls)
    np.testing.assert_allclose(cls[HEALPY_POLARIZED_ORDER.index("EB")], 0.0)
    np.testing.assert_allclose(cls[HEALPY_POLARIZED_ORDER.index("TB")], 0.0)


def test_cmb_cl_template_cache_invalidates_on_file_mtime(tmp_path):
    """Editing a template file invalidates the cached load.

    A7 keys the lru_cache on ``(path, no_tensor_mtime_ns, r1_mtime_ns)``,
    so writes to a template file produce a fresh load instead of a stale
    cached one.
    """
    import os
    import time

    a = tmp_path / "camb_lens_nobb.dat"
    b = tmp_path / "camb_lens_r1.dat"

    _write_cmb_template(a, [[i, 1.0, 1.0, 1.0, 0.0] for i in range(1, 6)])
    _write_cmb_template(b, [[i, 1.0, 1.0, 1.0, 0.0] for i in range(1, 6)])
    cls_before = CMBCl(template_dir=tmp_path).to_healpy_cls(4)[0]

    # Bump file content and mtime so the cache key changes.
    time.sleep(0.01)
    _write_cmb_template(a, [[i, 10.0, 10.0, 10.0, 0.0] for i in range(1, 6)])
    _write_cmb_template(b, [[i, 10.0, 10.0, 10.0, 0.0] for i in range(1, 6)])
    new_time = time.time()
    os.utime(a, (new_time, new_time))
    os.utime(b, (new_time, new_time))

    cls_after = CMBCl(template_dir=tmp_path).to_healpy_cls(4)[0]
    assert not np.allclose(cls_before, cls_after)


def test_cmb_cl_unit_is_a_classvar_not_a_field():
    """A7 demoted ``unit`` to ``ClassVar``; passing it as a kwarg now errors."""
    with pytest.raises(TypeError, match="unit"):
        CMBCl(unit="K_CMB^2")  # type: ignore[call-arg]
    # but the class-level constant still resolves
    assert CMBCl.unit == "uK_CMB^2"
