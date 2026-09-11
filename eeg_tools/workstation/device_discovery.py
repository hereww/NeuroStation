"""Cross-platform discovery helpers used by the BrainFlow workbench.

The workstation must not require users to guess a COM number.  BrainFlow
itself opens the serial device, so this module deliberately only discovers
candidate names; the acquisition worker remains responsible for opening and
validating a candidate with ``BoardShim.prepare_session``.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class SerialPortInfo:
    """A serial endpoint that can be passed to BrainFlow."""

    device: str
    description: str = ""
    hardware_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "device": self.device,
            "description": self.description,
            "hardware_id": self.hardware_id,
        }


def _normalise(items: Iterable[SerialPortInfo]) -> list[SerialPortInfo]:
    unique: dict[str, SerialPortInfo] = {}
    for item in items:
        device = item.device.strip()
        if not device:
            continue
        key = device.casefold()
        unique.setdefault(key, SerialPortInfo(device, item.description, item.hardware_id))

    def sort_key(item: SerialPortInfo) -> tuple[int, str]:
        match = re.fullmatch(r"COM(\d+)", item.device.upper())
        return (int(match.group(1)), item.device.upper()) if match else (10**9, item.device)

    return sorted(unique.values(), key=sort_key)


def _windows_registry_ports() -> list[SerialPortInfo]:
    if os.name != "nt":
        return []
    try:
        import winreg

        key_path = r"HARDWARE\DEVICEMAP\SERIALCOMM"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            result: list[SerialPortInfo] = []
            for index in range(winreg.QueryInfoKey(key)[1]):
                name, value, _kind = winreg.EnumValue(key, index)
                result.append(SerialPortInfo(str(value), str(name)))
            return result
    except (ImportError, OSError):
        return []


def _unix_device_ports() -> list[SerialPortInfo]:
    if os.name == "nt":
        return []
    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/ttyS*", "/dev/cu.*", "/dev/tty.*")
    return [SerialPortInfo(path) for pattern in patterns for path in glob.glob(pattern)]


def discover_serial_ports() -> tuple[SerialPortInfo, ...]:
    """Return available serial endpoints without opening any device.

    ``pyserial`` is used when present for richer descriptions, while the
    standard-library fallbacks keep discovery working in the packaged app
    where pyserial is intentionally optional.
    """

    discovered: list[SerialPortInfo] = []
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]

        discovered.extend(
            SerialPortInfo(
                str(port.device),
                str(port.description or ""),
                str(port.hwid or ""),
            )
            for port in list_ports.comports()
        )
    except (ImportError, OSError):
        pass
    discovered.extend(_windows_registry_ports())
    discovered.extend(_unix_device_ports())
    return tuple(_normalise(discovered))


def candidate_serial_ports(requested: str | None) -> tuple[SerialPortInfo, ...]:
    """Resolve an explicit port or the automatic scan order.

    ``AUTO`` (also accepted as an empty value) means all discovered ports in
    deterministic order.  Explicit values are retained even when discovery
    cannot see them, which lets BrainFlow report a useful driver error.
    """

    requested_value = (requested or "AUTO").strip()
    if requested_value and requested_value.upper() != "AUTO":
        return (SerialPortInfo(requested_value),)
    return discover_serial_ports()


__all__ = ["SerialPortInfo", "candidate_serial_ports", "discover_serial_ports"]
