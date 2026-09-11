from __future__ import annotations

import unittest
from unittest.mock import patch

from eeg_tools.workstation.device_discovery import (
    SerialPortInfo,
    candidate_serial_ports,
    discover_serial_ports,
)


class DeviceDiscoveryTests(unittest.TestCase):
    def test_auto_candidates_use_deterministic_deduplicated_order(self) -> None:
        with patch(
            "eeg_tools.workstation.device_discovery.discover_serial_ports",
            return_value=(
                SerialPortInfo("COM12", "second"),
                SerialPortInfo("COM5", "first"),
                SerialPortInfo("COM5", "duplicate"),
            ),
        ):
            self.assertEqual(["COM5", "COM12"], [item.device for item in candidate_serial_ports("AUTO")])

    def test_explicit_port_does_not_require_discovery(self) -> None:
        with patch(
            "eeg_tools.workstation.device_discovery.discover_serial_ports",
            side_effect=AssertionError("explicit port must not scan"),
        ):
            self.assertEqual(["COM99"], [item.device for item in candidate_serial_ports("COM99")])

    def test_discovery_is_safe_without_pyserial(self) -> None:
        ports = discover_serial_ports()
        self.assertIsInstance(ports, tuple)
        self.assertTrue(all(item.device for item in ports))


if __name__ == "__main__":
    unittest.main()
