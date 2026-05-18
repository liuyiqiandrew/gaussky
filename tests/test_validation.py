"""Tests for the shared validation helpers in :mod:`gaussky._validation`."""

from __future__ import annotations

import pytest

from gaussky._validation import validate_lmax


def test_validate_lmax_accepts_non_negative_integers() -> None:
    """Zero and positive integers are valid band-limits."""
    for lmax in (0, 1, 100, 10_000):
        validate_lmax(lmax)


def test_validate_lmax_rejects_negative() -> None:
    """Negative ``lmax`` is rejected with a clear message."""
    with pytest.raises(ValueError, match="non-negative"):
        validate_lmax(-1)


def test_validate_lmax_rejects_non_integer() -> None:
    """Non-integer ``lmax`` is rejected with ``TypeError``."""
    with pytest.raises(TypeError, match="integer"):
        validate_lmax(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        validate_lmax("32")  # type: ignore[arg-type]


def test_validate_lmax_rejects_bool() -> None:
    """``bool`` (a subclass of int) is rejected — it's almost never intended."""
    with pytest.raises(TypeError, match="integer"):
        validate_lmax(True)  # type: ignore[arg-type]
