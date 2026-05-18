"""Tests for the harmonic-space container :class:`gaussky.map.MultiFreqCompAlm`."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest

from gaussky.map import MultiFreqCompAlm


def _alms(nfreq: int, nfields: int, lmax: int) -> np.ndarray:
    """Return a deterministic complex alm array of the expected shape."""
    nalm = int(hp.Alm.getsize(lmax))
    real = np.arange(nfreq * nfields * nalm, dtype=np.float64)
    return real.reshape(nfreq, nfields, nalm) + 1j * real.reshape(nfreq, nfields, nalm)


def _alm_container(
    *,
    lmax: int = 4,
    fields: tuple = ("T", "E", "B"),
    freqs_ghz: tuple = (30.0, 90.0),
    unit: str = "uK_CMB",
    component_name: str = "cmb",
    beam_fwhm_rad: object = None,
    coord: str | None = None,
    metadata: dict | None = None,
) -> MultiFreqCompAlm:
    alms = _alms(len(freqs_ghz), len(fields), lmax)
    return MultiFreqCompAlm(
        alms=alms,
        lmax=lmax,
        freqs_ghz=np.asarray(freqs_ghz, dtype=np.float64),
        fields=fields,
        unit=unit,
        component_name=component_name,
        beam_fwhm_rad=beam_fwhm_rad,
        coord=coord,
        metadata={} if metadata is None else metadata,
    )


def test_alm_container_stores_validated_readonly_complex_array() -> None:
    """The stored alms are float64-complex and read-only."""
    alm = _alm_container()
    assert alm.alms.dtype == np.complex128
    assert not alm.alms.flags.writeable
    assert alm.nfreq == 2
    assert alm.nfields == 3
    assert alm.nalm == int(hp.Alm.getsize(alm.lmax))


def test_alm_container_normalizes_mmax_to_lmax_when_unset() -> None:
    """``mmax=None`` is canonicalized to ``mmax == lmax``."""
    alm = _alm_container(lmax=6)
    assert alm.mmax == alm.lmax


def test_alm_container_rejects_mismatched_alm_shape() -> None:
    """An alm array whose trailing axis disagrees with ``lmax`` is rejected."""
    with pytest.raises(ValueError, match="shape"):
        MultiFreqCompAlm(
            alms=np.zeros((1, 1, 3), dtype=np.complex128),
            lmax=10,
            freqs_ghz=np.array([90.0]),
            fields=("T",),
            unit="uK_CMB",
            component_name="cmb",
        )


def test_alm_container_rejects_unknown_harmonic_field() -> None:
    """Only ``T``, ``E``, ``B`` are valid harmonic-field labels."""
    with pytest.raises(ValueError, match="Unknown harmonic field"):
        _alm_container(fields=("T", "Q"))  # type: ignore[arg-type]


def test_alm_container_rejects_empty_unit_or_component_name() -> None:
    """Empty strings on ``unit`` or ``component_name`` are rejected."""
    with pytest.raises(ValueError, match="unit"):
        _alm_container(unit="")
    with pytest.raises(ValueError, match="component_name"):
        _alm_container(component_name="")


def test_alm_container_rejects_non_finite_alms() -> None:
    """NaN/inf entries are rejected so downstream alm2map cannot silently blow up."""
    alms = _alms(1, 1, 4)
    alms[0, 0, 0] = complex(np.nan, 0.0)
    with pytest.raises(ValueError, match="finite"):
        MultiFreqCompAlm(
            alms=alms,
            lmax=4,
            freqs_ghz=np.array([90.0]),
            fields=("T",),
            unit="uK_CMB",
            component_name="cmb",
        )


def test_alm_container_metadata_is_readonly_mappingproxy() -> None:
    """The metadata view is read-only, matching the signal-map convention."""
    alm = _alm_container(metadata={"a_lens": 1.0})
    assert alm.metadata["a_lens"] == 1.0
    with pytest.raises(TypeError):
        alm.metadata["new"] = "value"  # type: ignore[index]


def test_alm_container_copy_with_revalidates_replaced_fields() -> None:
    """``copy_with`` propagates validation, including unknown-field rejection."""
    alm = _alm_container(component_name="cmb")
    relabelled = alm.copy_with(component_name="cmb_lensed")
    assert relabelled.component_name == "cmb_lensed"
    np.testing.assert_array_equal(relabelled.alms, alm.alms)

    with pytest.raises(ValueError, match="Unknown alm-container field"):
        alm.copy_with(bogus=1)


def test_alm_container_supports_per_frequency_beam_provenance() -> None:
    """``beam_fwhm_rad`` accepts a per-channel array (informational only)."""
    alm = _alm_container(beam_fwhm_rad=np.array([0.01, 0.02]))
    assert alm.beam_fwhm_rad is not None
    np.testing.assert_allclose(alm.beam_fwhm_rad, [0.01, 0.02])
