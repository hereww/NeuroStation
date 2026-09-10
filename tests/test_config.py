from __future__ import annotations

import copy
import unittest
from pathlib import Path

from eeg_tools.config import ConfigError, load_and_validate_configs, validate_stimulus_config


ROOT = Path(__file__).resolve().parents[1]
STIMULUS_CONFIG = ROOT / "configs" / "ssvep_config_v1.json"
CHANNEL_CONFIG = ROOT / "configs" / "channel_config_v1_template.json"


class ConfigTests(unittest.TestCase):
    def test_repository_configs_are_structurally_valid(self) -> None:
        _, _, warnings = load_and_validate_configs(STIMULUS_CONFIG, CHANNEL_CONFIG)
        self.assertTrue(any("draft" in warning.lower() for warning in warnings))
        self.assertTrue(any("electrode positions" in warning.lower() for warning in warnings))

    def test_frequency_must_divide_refresh_rate(self) -> None:
        stimulus, _, _ = load_and_validate_configs(STIMULUS_CONFIG, CHANNEL_CONFIG)
        invalid = copy.deepcopy(stimulus)
        invalid["frequencies_hz"] = [11]
        with self.assertRaisesRegex(ConfigError, "does not divide"):
            validate_stimulus_config(invalid, STIMULUS_CONFIG)


if __name__ == "__main__":
    unittest.main()
