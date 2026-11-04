"""Audit native personal SSVEP sessions without training or modifying data."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.personal_ssvep_session_contract import (  # noqa: E402
    PersonalSSVEPSessionContractError,
    audit_session,
)


def _write_new(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent run from overwriting a report.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-repetitions-per-target", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        sources = [path.expanduser().resolve() for path in args.session]
        output = args.output.expanduser().resolve()
        if len(set(sources)) != len(sources):
            raise ValueError("each session directory may appear only once")
        if any(output.is_relative_to(source) for source in sources):
            raise ValueError("output must be outside every source session")
        if output.exists():
            raise ValueError(f"output already exists: {output}")
        sessions = [
            audit_session(path, minimum_repetitions_per_target=args.minimum_repetitions_per_target)
            for path in sources
        ]
        report = {
            "schema_version": 1,
            "status": "formal",
            "completion_status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_stage": "personal_ssvep_session_label_and_validation_contract",
            "session_count": len(sessions),
            "sessions": sessions,
            "all_sources_unchanged": all(item["source_files_unchanged"] for item in sessions),
            "runtime_enabled": False,
            "production_validated": False,
            "default_deployment_authorized": False,
        }
        _write_new(output, report)
    except (OSError, ValueError, TypeError, PersonalSSVEPSessionContractError) as exc:
        print(
            json.dumps(
                {
                    "status": "formal",
                    "completion_status": "incomplete",
                    "error": str(exc),
                    "runtime_enabled": False,
                    "production_validated": False,
                    "default_deployment_authorized": False,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

