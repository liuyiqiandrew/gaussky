"""Tests for the power → signal unit registry in :mod:`gaussky.units`."""

from __future__ import annotations

import pytest

import gaussky.units as units
from gaussky.conventions import U_K_CMB, U_K_CMB_SQUARED
from gaussky.units import POWER_TO_SIGNAL, register_signal_unit, signal_unit_for


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    """Snapshot and restore the registry around every test to keep state clean."""
    saved = dict(POWER_TO_SIGNAL)
    yield
    POWER_TO_SIGNAL.clear()
    POWER_TO_SIGNAL.update(saved)


def test_default_mapping_covers_bundled_unit() -> None:
    """The bundled ``uK_CMB^2 → uK_CMB`` mapping is registered out of the box."""
    assert signal_unit_for(U_K_CMB_SQUARED) == U_K_CMB


def test_signal_unit_for_unknown_unit_raises() -> None:
    """Unregistered units raise a helpful ``ValueError``."""
    with pytest.raises(ValueError, match="Unsupported"):
        signal_unit_for("MJy^2/sr^2")


def test_register_signal_unit_then_lookup() -> None:
    """A newly registered mapping resolves through :func:`signal_unit_for`."""
    register_signal_unit("K_RJ^2", "K_RJ")
    assert signal_unit_for("K_RJ^2") == "K_RJ"


def test_register_signal_unit_is_idempotent_for_same_mapping() -> None:
    """Registering the same mapping twice is a no-op."""
    register_signal_unit("K_RJ^2", "K_RJ")
    register_signal_unit("K_RJ^2", "K_RJ")
    assert POWER_TO_SIGNAL["K_RJ^2"] == "K_RJ"


def test_register_signal_unit_rejects_conflicting_remap() -> None:
    """Registering a different signal unit for an existing key raises."""
    register_signal_unit("K_RJ^2", "K_RJ")
    with pytest.raises(ValueError, match="already registered"):
        register_signal_unit("K_RJ^2", "uK_RJ")


def test_register_signal_unit_rejects_empty_strings() -> None:
    """Empty unit strings on either side are rejected."""
    with pytest.raises(ValueError, match="must not be empty"):
        register_signal_unit("", "K_RJ")
    with pytest.raises(ValueError, match="must not be empty"):
        register_signal_unit("K_RJ^2", "")


def test_register_signal_unit_rejects_non_string_arguments() -> None:
    """Non-string arguments are rejected with ``TypeError``."""
    with pytest.raises(TypeError, match="must be strings"):
        register_signal_unit(1, "K_RJ")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be strings"):
        register_signal_unit("K_RJ^2", 2)  # type: ignore[arg-type]


def test_registry_is_module_level_mutable_state() -> None:
    """``POWER_TO_SIGNAL`` is the canonical registry; the fixture restores it."""
    assert isinstance(units.POWER_TO_SIGNAL, dict)
    assert U_K_CMB_SQUARED in units.POWER_TO_SIGNAL
