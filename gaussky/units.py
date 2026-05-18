"""Unit registry for power-spectrum → signal-map unit translation.

The Gaussian sampling helpers in :mod:`gaussky.component.component_utils`
need to know which signal-map unit corresponds to a given power-spectrum
unit (e.g. ``uK_CMB^2 → uK_CMB``). Rather than hard-code this in the helper,
the registry below maps power-spectrum units to signal units. New spectrum
or noise units register their own mapping with :func:`register_signal_unit`.

The default mapping covers the only unit pair the bundled spectra use; the
registry is extensible so noise components (which may carry, for example,
``uK_CMB^2`` variance or ``Jy^2/sr``) and future spectrum models can plug
in without patching the sampling helpers.
"""

from __future__ import annotations

from gaussky.conventions import U_K_CMB, U_K_CMB_SQUARED

__all__ = [
    "POWER_TO_SIGNAL",
    "register_signal_unit",
    "signal_unit_for",
]


POWER_TO_SIGNAL: dict[str, str] = {
    U_K_CMB_SQUARED: U_K_CMB,
}


def signal_unit_for(power_unit: str) -> str:
    """Return the signal-map unit implied by a power-spectrum unit.

    Parameters
    ----------
    power_unit : str
        Unit string carried by an :class:`~gaussky.ps.AngularPowerSpectrum`
        (typically ``"uK_CMB^2"`` for the bundled models).

    Returns
    -------
    str
        Signal-map unit string for the sampled realization.

    Raises
    ------
    ValueError
        If ``power_unit`` has no registered mapping. Register it first with
        :func:`register_signal_unit` if the unit is supported by a custom
        spectrum or noise model.
    """
    try:
        return POWER_TO_SIGNAL[power_unit]
    except KeyError as error:
        raise ValueError(
            f"Unsupported power-spectrum unit for Gaussian component: "
            f"{power_unit!r}. Register it via "
            f"gaussky.units.register_signal_unit(...) first."
        ) from error


def register_signal_unit(power_unit: str, signal_unit: str) -> None:
    """Register a power-spectrum → signal-map unit mapping.

    Parameters
    ----------
    power_unit : str
        Unit string carried by spectra (e.g. ``"K_RJ^2"``, ``"Jy^2/sr"``).
    signal_unit : str
        Unit string for the sampled signal map (e.g. ``"K_RJ"``,
        ``"Jy/sr"``).

    Raises
    ------
    TypeError
        If either argument is not a string.
    ValueError
        If either argument is empty, or if ``power_unit`` is already
        registered with a different signal unit.
    """
    if not isinstance(power_unit, str) or not isinstance(signal_unit, str):
        raise TypeError("power_unit and signal_unit must be strings")
    if power_unit == "" or signal_unit == "":
        raise ValueError("power_unit and signal_unit must not be empty")

    existing = POWER_TO_SIGNAL.get(power_unit)
    if existing is not None and existing != signal_unit:
        raise ValueError(
            f"power_unit {power_unit!r} is already registered with signal "
            f"unit {existing!r}; refusing to remap to {signal_unit!r}"
        )
    POWER_TO_SIGNAL[power_unit] = signal_unit
