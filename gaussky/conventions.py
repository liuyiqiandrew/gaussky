"""Shared domain conventions used across gaussky subpackages."""

from __future__ import annotations

from typing import Literal, TypeAlias

SignalField: TypeAlias = Literal["T", "Q", "U"]
HarmonicField: TypeAlias = Literal["T", "E", "B"]
SpectrumPair: TypeAlias = Literal["TT", "EE", "BB", "TE", "EB", "TB"]
HealpixOrdering: TypeAlias = Literal["RING", "NESTED"]

SIGNAL_FIELDS: tuple[SignalField, ...] = ("T", "Q", "U")
HARMONIC_FIELDS: tuple[HarmonicField, ...] = ("T", "E", "B")
HEALPY_POLARIZED_ORDER: tuple[SpectrumPair, ...] = (
    "TT",
    "EE",
    "BB",
    "TE",
    "EB",
    "TB",
)
HEALPIX_ORDERINGS: tuple[HealpixOrdering, ...] = ("RING", "NESTED")

U_K_CMB: str = "uK_CMB"
U_K_CMB_SQUARED: str = "uK_CMB^2"
DIMENSIONLESS: str = "dimensionless"
