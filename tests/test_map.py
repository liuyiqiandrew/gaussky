import numpy as np
import pytest

from gaussky.conventions import HEALPIX_ORDERINGS, SIGNAL_FIELDS
from gaussky.map import (
    SUPPORTED_HEALPIX_ORDERINGS,
    SUPPORTED_SIGNAL_FIELDS,
    AuxiliaryHealpixMap,
    MultiFreqCompMap,
    MultiFreqTotalMap,
)


def _npix(nside=1):
    return 12 * nside**2


def _auxiliary(
    *,
    maps=None,
    nside=1,
    ordering="ring",
    coord="G",
    unit=None,
    metadata=None,
):
    if maps is None:
        maps = np.arange(_npix(nside), dtype=np.float64)

    return AuxiliaryHealpixMap(
        maps=maps,
        nside=nside,
        ordering=ordering,
        coord=coord,
        unit=unit,
        metadata={} if metadata is None else metadata,
    )


def _component(
    *,
    name="sync",
    freqs=(30.0, 90.0),
    fields=("Q", "U"),
    maps=None,
    unit="uK_CMB",
    nside=1,
    beam_fwhm_rad=(0.01, 0.02),
    ordering="ring",
    coord="G",
    auxiliary_maps=None,
    metadata=None,
):
    if maps is None:
        maps = np.arange(
            len(freqs) * len(fields) * _npix(nside),
            dtype=np.float64,
        ).reshape(len(freqs), len(fields), _npix(nside))

    return MultiFreqCompMap(
        component_name=name,
        freqs_ghz=np.asarray(freqs, dtype=np.float64),
        fields=fields,
        unit=unit,
        maps=maps,
        nside=nside,
        beam_fwhm_rad=beam_fwhm_rad,
        ordering=ordering,
        coord=coord,
        auxiliary_maps={} if auxiliary_maps is None else auxiliary_maps,
        metadata={} if metadata is None else metadata,
    )


def test_map_exports_shared_conventions():
    assert SUPPORTED_SIGNAL_FIELDS == SIGNAL_FIELDS
    assert SUPPORTED_HEALPIX_ORDERINGS == HEALPIX_ORDERINGS


def test_auxiliary_healpix_map_has_only_generic_healpix_contract():
    auxiliary = _auxiliary(
        maps=np.arange(2 * _npix(), dtype=np.float64).reshape(2, _npix()),
        unit="dimensionless",
        metadata={"field": "beta_p"},
    )

    assert auxiliary.npix == _npix()
    assert auxiliary.npix_expected == _npix()
    assert auxiliary.ordering == "RING"
    assert auxiliary.coord == "G"
    assert auxiliary.unit == "dimensionless"
    assert not hasattr(auxiliary, "freqs_ghz")
    assert not hasattr(auxiliary, "fields")
    assert auxiliary.maps.dtype == np.float64
    assert not auxiliary.maps.flags.writeable

    with pytest.raises(TypeError):
        auxiliary.metadata["new"] = "value"


def test_auxiliary_healpix_map_validates_pixel_axis_and_metadata():
    with pytest.raises(ValueError, match="trailing HEALPix pixel axis"):
        _auxiliary(maps=1.0)

    with pytest.raises(ValueError, match="trailing axis"):
        _auxiliary(maps=np.zeros(_npix() - 1))

    with pytest.raises(ValueError, match="power of two"):
        _auxiliary(nside=3, maps=np.zeros(_npix(3)))

    with pytest.raises(ValueError, match="unit"):
        _auxiliary(unit="")


def test_auxiliary_pixelization_compatibility_checks_healpix_metadata():
    reference = _auxiliary()
    reference.assert_same_pixelization(_auxiliary(maps=np.zeros((2, _npix()))))

    with pytest.raises(ValueError, match="nside"):
        reference.assert_same_pixelization(_auxiliary(nside=2))

    with pytest.raises(ValueError, match="orderings"):
        reference.assert_same_pixelization(_auxiliary(ordering="nested"))

    with pytest.raises(ValueError, match="coordinate"):
        reference.assert_same_pixelization(_auxiliary(coord="C"))


def test_component_map_normalizes_metadata_and_requires_auxiliary_containers():
    auxiliary = _auxiliary(metadata={"field": "beta_p"})
    component = _component(
        auxiliary_maps={"beta": auxiliary},
        metadata={"model": "power-law"},
    )

    assert component.nfreq == 2
    assert component.nfields == 2
    assert component.npix == _npix()
    assert component.npix_expected == _npix()
    assert component.fields == ("Q", "U")
    assert component.ordering == "RING"
    assert component.maps.dtype == np.float64
    assert not component.maps.flags.writeable
    assert not component.freqs_ghz.flags.writeable
    assert not component.beam_fwhm_rad.flags.writeable
    assert component.auxiliary_maps["beta"] is auxiliary

    with pytest.raises(TypeError):
        component.metadata["new"] = "value"
    with pytest.raises(TypeError):
        component.auxiliary_maps["new"] = auxiliary


def test_component_map_rejects_invalid_signal_shape_and_auxiliary_maps():
    with pytest.raises(ValueError, match="trailing axis"):
        _component(maps=np.zeros((2, 2, _npix() - 1)))

    with pytest.raises(ValueError, match="maps must have shape"):
        _component(maps=np.zeros((2, _npix())))

    with pytest.raises(ValueError, match="Unknown signal field"):
        _component(fields=("Q", "V"))

    with pytest.raises(ValueError, match="power of two"):
        _component(nside=3, maps=np.zeros((2, 2, _npix(3))))

    with pytest.raises(ValueError, match="beam_fwhm_rad"):
        _component(beam_fwhm_rad=(0.01,))

    with pytest.raises(TypeError, match="HealpixMapContainer"):
        _component(auxiliary_maps={"beta": np.zeros(_npix())})

    with pytest.raises(ValueError, match="coordinate"):
        _component(auxiliary_maps={"beta": _auxiliary(coord="C")})


def test_indexing_selection_and_copy_helpers_preserve_signal_contract():
    maps = np.arange(2 * 2 * _npix(), dtype=np.float64).reshape(2, 2, _npix())
    auxiliary = _auxiliary()
    component = _component(maps=maps, auxiliary_maps={"beta": auxiliary})

    assert component.field_index("U") == 1
    assert component.freq_index(90.0) == 1

    selected_field = component.select_field("U")
    assert selected_field.component_name == component.component_name
    assert selected_field.fields == ("U",)
    assert selected_field.auxiliary_maps["beta"] is auxiliary
    np.testing.assert_allclose(selected_field.maps, maps[:, 1:2, :])
    assert not selected_field.maps.flags.writeable

    selected_freq = component.select_freq(90.0)
    np.testing.assert_allclose(selected_freq.freqs_ghz, [90.0])
    np.testing.assert_allclose(selected_freq.maps, maps[1:2, :, :])
    assert selected_freq.beam_fwhm_rad == 0.02
    assert selected_freq.auxiliary_maps["beta"] is auxiliary

    copied = component.copy_with(unit="K_CMB")
    assert copied.unit == "K_CMB"
    np.testing.assert_allclose(copied.maps, component.maps)

    with pytest.raises(ValueError, match="Unknown map field replacement"):
        component.copy_with(unknown=1)


def test_signal_compatibility_checks_map_grid_and_metadata():
    reference = _component()
    reference.assert_compatible(_component(name="dust"))

    reference.assert_compatible(
        _component(name="dust", beam_fwhm_rad=np.array([0.01, 0.02]))
    )

    with pytest.raises(ValueError, match="frequency grids"):
        reference.assert_compatible(_component(name="dust", freqs=(30.0, 91.0)))

    with pytest.raises(ValueError, match="map units"):
        reference.assert_compatible(_component(name="dust", unit="K_CMB"))

    with pytest.raises(ValueError, match="beam FWHM"):
        reference.assert_compatible(_component(name="dust", beam_fwhm_rad=None))


def test_total_map_is_built_from_compatible_components():
    sync = _component(name="sync", maps=np.ones((2, 2, _npix())))
    dust = _component(name="dust", maps=2.0 * np.ones((2, 2, _npix())))

    total = MultiFreqTotalMap.from_components((sync, dust))

    np.testing.assert_allclose(total.maps, 3.0)
    assert not total.maps.flags.writeable
    assert total.components == (sync, dust)
    assert total.component_names == ("sync", "dust")
    assert total.component("dust") is dust

    selected = total.select_freq(90.0).select_field("Q")
    assert selected.freqs_ghz.tolist() == [90.0]
    assert selected.fields == ("Q",)
    np.testing.assert_allclose(selected.maps, 3.0)
    assert all(component.fields == ("Q",) for component in selected.components)

    with pytest.raises(KeyError):
        total.component("cmb")


def test_total_map_rejects_duplicate_incompatible_or_inconsistent_components():
    sync = _component(name="sync", maps=np.ones((2, 2, _npix())))
    duplicate_sync = _component(name="sync", maps=np.ones((2, 2, _npix())))
    dust = _component(name="dust", maps=2.0 * np.ones((2, 2, _npix())))

    with pytest.raises(ValueError, match="unique"):
        MultiFreqTotalMap.from_components((sync, duplicate_sync))

    with pytest.raises(ValueError, match="map fields"):
        MultiFreqTotalMap.from_components(
            (sync, _component(name="dust", fields=("Q",)))
        )

    with pytest.raises(ValueError, match="sum of component maps"):
        MultiFreqTotalMap(
            freqs_ghz=sync.freqs_ghz,
            fields=sync.fields,
            unit=sync.unit,
            maps=np.zeros_like(sync.maps),
            nside=sync.nside,
            beam_fwhm_rad=sync.beam_fwhm_rad,
            ordering=sync.ordering,
            coord=sync.coord,
            components=(sync, dust),
        )
