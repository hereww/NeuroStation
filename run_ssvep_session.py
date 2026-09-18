"""Compatibility CLI for the workstation's unified Qt/BrainFlow SSVEP worker.

The former pygame runner used a different 160-trial protocol.  This entry now
routes to the same versioned protocol and acquisition implementation as the
desktop application, so CLI and workstation datasets have identical semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eeg_tools.config import ConfigError, validate_channel_config
from eeg_tools.workstation.acquisition_worker import main as worker_main
from eeg_tools.workstation.dataset import DatasetRepository
from eeg_tools.workstation.ssvep import SSVEPProtocol, SSVEPProtocolError


ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
DEFAULT_CHANNEL_CONFIG = ROOT / "configs" / "channel_config_v1_auto.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one NeuroStation SSVEP acquisition session."
    )
    parser.add_argument("--port", default="AUTO")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--channel-config", type=Path, default=DEFAULT_CHANNEL_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DatasetRepository.default_root())
    parser.add_argument("--participant", default="P001")
    parser.add_argument("--session-name", default="SSVEP")
    parser.add_argument("--user-id", default="")
    parser.add_argument("--user-name", default="")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--stimulus-seconds", type=float)
    parser.add_argument("--rest-seconds", type=float)
    parser.add_argument("--countdown-seconds", type=float)
    parser.add_argument("--eye-side", choices=("left", "right"), required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--acknowledge-flicker-risk", action="store_true")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--allow-draft-channel-config", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.validate_only:
        try:
            protocol = SSVEPProtocol.load(arguments.protocol)
            channel_value = json.loads(arguments.channel_config.read_text(encoding="utf-8"))
            warnings = validate_channel_config(channel_value, arguments.channel_config)
        except (ConfigError, OSError, json.JSONDecodeError, SSVEPProtocolError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "valid": True,
                    "protocol": str(arguments.protocol.resolve()),
                    "channel_config": str(arguments.channel_config.resolve()),
                    "trial_count": protocol.trial_count,
                    "recording_duration_s": protocol.recording_duration_s,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    forwarded = [
        "--protocol", str(arguments.protocol),
        "--channel-config", str(arguments.channel_config),
        "--output-root", str(arguments.output_root),
        "--participant", arguments.participant,
        "--session-name", arguments.session_name,
        "--user-id", arguments.user_id,
        "--user-name", arguments.user_name,
        "--board", "cyton",
        "--port", arguments.port,
        "--eye-side", arguments.eye_side,
    ]
    optional_values = (
        ("--repetitions", arguments.repetitions),
        ("--stimulus-seconds", arguments.stimulus_seconds),
        ("--rest-seconds", arguments.rest_seconds),
        ("--countdown-seconds", arguments.countdown_seconds),
    )
    for flag, value in optional_values:
        if value is not None:
            forwarded.extend([flag, str(value)])
    for flag, enabled in (
        ("--headless", arguments.headless),
        ("--acknowledge-flicker-risk", arguments.acknowledge_flicker_risk),
        ("--allow-draft-protocol", arguments.allow_draft_protocol),
        ("--allow-draft-channel-config", arguments.allow_draft_channel_config),
    ):
        if enabled:
            forwarded.append(flag)
    return worker_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
