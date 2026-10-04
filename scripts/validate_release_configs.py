"""Reject non-formal bundled acquisition defaults before any desktop build."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.config import ConfigError, load_and_validate_configs
from eeg_tools.workstation.ssvep import SSVEPProtocol


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigError(f"Cannot read release configuration: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"Release configuration must be an object: {path}")
    return value


def validate_release_configs(root: Path = ROOT) -> dict:
    configs = root / "configs"
    protocols = sorted((configs / "protocols").rglob("*.json"))
    default_protocol = configs / "protocols" / "ssvep_four_target_v2.json"
    if default_protocol not in protocols:
        raise ConfigError(f"Default acquisition protocol is missing: {default_protocol}")

    # Include the legacy stimulus resource and all future protocol resources.
    stimulus_path = configs / "ssvep_config_v1.json"
    for path in [stimulus_path, *protocols]:
        value = _read_object(path)
        if value.get("status") != "formal":
            raise ConfigError(f"Release protocol status must be formal: {path}")

    for path in sorted(configs.rglob("*.json")):
        value = _read_object(path)
        for key in ("status", "config_version", "protocol_id"):
            label = str(value.get(key, "")).upper()
            if "DRAFT" in label or "MVP" in label:
                raise ConfigError(f"Release {key} must not contain MVP/DRAFT: {path}")

    for path in protocols:
        protocol = SSVEPProtocol.load(path)
        if (protocol.channel_count, protocol.sampling_rate_hz, protocol.refresh_rate_hz) != (8, 250, 60):
            raise ConfigError(f"Release protocol must match Cyton/display defaults: {path}")

    channel_path = configs / "channel_config_v1_auto.json"
    _, _, warnings = load_and_validate_configs(stimulus_path, channel_path)
    if warnings:
        raise ConfigError("Default acquisition configuration is incomplete: " + "; ".join(warnings))

    return {
        "status": "ready",
        "protocol_status": "formal",
        "protocol_count": len(protocols),
        "channel_config_path": str(channel_path.resolve()),
        "warnings": [],
    }


def main() -> int:
    try:
        report = validate_release_configs()
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
