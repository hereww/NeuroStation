"""Subprocess boundary for BrainFlow-backed SSVEP acquisition.

The desktop process owns presentation and navigation.  A separate worker owns
the serial device, BrainFlow session, stimulus window, markers, and dataset
files.  This keeps a driver or display failure from taking down the workstation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import time
from typing import Any

from neurostation_contract import (
    CaptureConfig,
    CaptureMode,
    Dataset,
    DeviceInfo,
    Phase,
    TaskSnapshot,
)

from .dataset import DatasetRepository
from eeg_tools.session_files import write_json, write_manifest
from .ssvep import SSVEPProtocol
from eeg_tools.waveform import channel_labels_from_path


ROOT = Path(__file__).resolve().parents[2]


class AcquisitionProcessGateway:
    """Run exactly one BrainFlow worker and translate its status into UI state."""

    def __init__(self, *, protocol_path: Path, channel_config_path: Path) -> None:
        self.protocol_path = protocol_path.resolve()
        self.channel_config_path = channel_config_path.resolve()
        self.config = CaptureConfig()
        self.device = DeviceInfo()
        self._snapshot = TaskSnapshot()
        self._datasets: list[Dataset] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._runtime_dir: Path | None = None
        self._status_file: Path | None = None
        self._cancel_file: Path | None = None
        self._stderr_file: Path | None = None
        self._preflight_file: Path | None = None
        self._preflight_report: dict[str, Any] | None = None
        self._last_status_mtime_ns = -1
        self._last_status: dict[str, Any] = {}
        self._live_waveform_path: Path | None = None
        self._live_waveform_position = 0
        self._live_waveform_header: list[str] = []
        self._live_channel_names: tuple[str, ...] = ()

    @property
    def snapshot(self) -> TaskSnapshot:
        return self._snapshot

    @property
    def datasets(self) -> tuple[Dataset, ...]:
        return tuple(self._datasets)

    def discard_dataset(self, dataset_id: str) -> None:
        self._datasets = [
            dataset for dataset in self._datasets if dataset.id != dataset_id
        ]

    @staticmethod
    def _is_bundled_executable() -> bool:
        main_module = sys.modules.get("__main__")
        return bool(
            getattr(sys, "frozen", False)
            or (main_module is not None and hasattr(main_module, "__compiled__"))
        )

    @staticmethod
    def _bundled_executable() -> Path:
        """Return the real standalone executable used for worker relaunches.

        Nuitka embeds Python modules, so ``__file__`` inside this module may
        point at a source-like path that is not present on disk.  Conversely,
        a launcher can report a relative or stale ``sys.executable``.  Resolve
        the executable from a small set of concrete candidates and fail with a
        useful diagnostic before ``Popen`` gets a cryptic WinError 2.
        """

        candidates: list[Path] = []
        executable = str(getattr(sys, "executable", "") or "").strip()
        if executable:
            candidates.append(Path(executable).expanduser())
        try:
            module_root = Path(__file__).resolve().parents[2]
        except OSError:
            module_root = None
        if module_root is not None:
            candidates.append(module_root / "workstation.exe")
            candidates.append(module_root / "workstation.bin")
        candidates.append(Path.cwd() / "workstation.exe")
        candidates.append(Path.cwd() / "workstation.bin")

        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            key = os.path.normcase(str(resolved))
            if key in seen:
                continue
            seen.add(key)
            if resolved.is_file():
                return resolved
        rendered = ", ".join(str(item) for item in candidates) or "<none>"
        raise FileNotFoundError(
            f"Standalone worker executable was not found; checked: {rendered}"
        )

    def _launch_context(self) -> tuple[list[str], str | None]:
        """Build a worker command and a safe working directory.

        All worker arguments are absolute.  A bundled process therefore does
        not need a ``cwd`` at all; omitting it avoids failures when Nuitka's
        embedded ``__file__`` resolves to a virtual source path.
        """

        if self._is_bundled_executable():
            return [str(self._bundled_executable()), "--acquisition-worker"], None
        return [
            sys.executable,
            "-m",
            "eeg_tools.workstation.acquisition_worker",
        ], str(ROOT) if ROOT.is_dir() else None

    def _command(self, config: CaptureConfig) -> list[str]:
        command, _ = self._launch_context()
        mode = CaptureMode(config.mode)
        if mode is not CaptureMode.CYTON:
            raise ValueError("validation.real_hardware_only")
        board = "cyton"
        channel_config = Path(config.channel_config or self.channel_config_path).resolve()
        command.extend(
            [
                "--protocol",
                str(self.protocol_path),
                "--channel-config",
                str(channel_config),
                "--output-root",
                str(Path(config.save_directory).expanduser().resolve()),
                "--participant",
                config.user_id or config.participant,
                "--session-name",
                config.name,
                "--user-id",
                config.user_id,
                "--user-name",
                config.user_name,
                "--board",
                board,
                "--port",
                config.port,
                "--repetitions",
                str(config.repetitions),
                "--stimulus-seconds",
                str(config.stimulus_seconds),
                "--rest-seconds",
                str(config.rest_seconds),
                "--screen-index",
                str(config.screen_index),
                "--status-file",
                str(self._status_file),
                "--cancel-file",
                str(self._cancel_file),
                "--acknowledge-flicker-risk",
            ]
        )
        if config.allow_draft_hardware_config:
            command.extend(["--allow-draft-protocol", "--allow-draft-channel-config"])
        if self._preflight_file is not None and self._preflight_file.is_file():
            command.extend(["--preflight-file", str(self._preflight_file)])
        return command

    def preflight_cyton(self, seconds: float = 3.0, port: str = "AUTO") -> dict[str, Any]:
        from .preflight import run_cyton_preflight
        report = run_cyton_preflight(port or self.config.port, seconds=seconds)
        self._preflight_report = report.as_dict()
        return self._preflight_report

    def test_cyton_channel(
        self,
        channel_number: int,
        seconds: float = 3.0,
        port: str = "AUTO",
    ) -> dict[str, Any]:
        """Run a short real Cyton sample and return one channel's checks."""

        if not 1 <= int(channel_number) <= 8:
            raise ValueError("validation.calibration_channel")
        report = self.preflight_cyton(seconds=seconds, port=port or "AUTO")
        checks = report.get("checks", [])
        channel_check = next(
            (
                check
                for check in checks
                if isinstance(check, dict) and check.get("name") == "channels"
            ),
            None,
        )
        metrics = channel_check.get("metrics", {}) if isinstance(channel_check, dict) else {}
        channel_metrics = metrics.get("channels", []) if isinstance(metrics, dict) else []
        selected = next(
            (
                item
                for item in channel_metrics
                if isinstance(item, dict) and int(item.get("channel", 0) or 0) == int(channel_number)
            ),
            None,
        )
        if selected is None:
            return {
                "status": "failed",
                "channel": int(channel_number),
                "report": report,
                "metrics": {},
                "detail": "Selected channel metrics were not returned.",
            }
        flat_fraction = float(selected.get("flat_fraction", 1.0) or 1.0)
        saturation_fraction = float(selected.get("saturation_fraction", 1.0) or 1.0)
        finite_fraction = float(selected.get("finite_fraction", 0.0) or 0.0)
        if finite_fraction < 0.99 or flat_fraction >= 0.95 or saturation_fraction >= 0.95:
            channel_status = "failed"
            detail = "Channel is mostly flat, saturated, or contains invalid samples."
        elif str(report.get("status") or "") in {"failed", "degraded"}:
            channel_status = "warning"
            detail = "The selected channel is changing, but the overall preflight requires review."
        else:
            channel_status = "passed"
            detail = "Channel samples are finite and changing."
        return {
            "status": channel_status,
            "channel": int(channel_number),
            "selected_port": report.get("selected_port", ""),
            "report_status": report.get("status", "unknown"),
            "report": report,
            "metrics": selected,
            "detail": detail,
        }

    def start_ssvep(self, config: CaptureConfig, speed: float = 1) -> TaskSnapshot:
        if self.snapshot.active:
            raise RuntimeError("validation.busy")
        config.validate()
        mode = CaptureMode(config.mode)
        if mode is not CaptureMode.CYTON:
            raise ValueError("validation.real_hardware_only")
        if speed != 1:
            raise ValueError("validation.production_speed")
        channel_path = Path(config.channel_config or self.channel_config_path).resolve()
        if not channel_path.is_file():
            raise ValueError("validation.channel_missing")

        output_root = Path(config.save_directory).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.device = DeviceInfo(
            name="OpenBCI Cyton",
            port=config.port,
            channels=8,
            sample_rate=250,
            connected=False,
            simulated=False,
        )
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="neurostation-worker-"))
        self._status_file = self._runtime_dir / "status.json"
        self._cancel_file = self._runtime_dir / "cancel.request"
        self._stderr_file = self._runtime_dir / "worker.log"
        self._preflight_file = None
        if self._preflight_report is not None and mode == CaptureMode.CYTON:
            self._preflight_file = self._runtime_dir / "preflight.json"
            write_json(self._preflight_file, self._preflight_report)
        self._last_status_mtime_ns = -1
        self._last_status = {}
        self._live_waveform_path = None
        self._live_waveform_position = 0
        self._live_waveform_header = []
        channel_config = Path(config.channel_config or self.channel_config_path).resolve()
        self._live_channel_names = channel_labels_from_path(channel_config, self.device.channels)
        protocol = SSVEPProtocol.load(self.protocol_path).with_runtime_parameters(
            repetitions=config.repetitions,
            stimulus_s=config.stimulus_seconds,
            rest_s=config.rest_seconds,
        )
        self._snapshot = TaskSnapshot(
            phase=Phase.COUNTDOWN,
            countdown=protocol.countdown_s,
            remaining=protocol.recording_duration_s,
            trial_count=protocol.trial_count,
            speed=1,
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            command = self._command(config)
            _, working_directory = self._launch_context()
            with self._stderr_file.open("wb") as stderr_stream:
                self._process = subprocess.Popen(
                    command,
                    cwd=working_directory,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_stream,
                    creationflags=creation_flags,
                )
        except OSError as error:
            self._snapshot = TaskSnapshot(phase=Phase.FAILED, error=str(error))
            rendered_command = locals().get("command", [])
            rendered_cwd = locals().get("working_directory")
            detail = f"{error}; command={rendered_command!r}; cwd={rendered_cwd!r}"
            raise RuntimeError(f"validation.worker_start: {detail}") from error
        return self._snapshot

    def tick(self) -> TaskSnapshot:
        if self._process is None:
            return self._snapshot
        status = self._read_status()
        if status:
            self._snapshot = self._map_status(status)
        return_code = self._process.poll()
        if return_code is not None and self._snapshot.phase not in {
            Phase.COMPLETED,
            Phase.CANCELLED,
            Phase.FAILED,
        }:
            # A final atomic status write normally precedes process exit. Retry
            # once after poll, then surface the captured worker diagnostic.
            status = self._read_status(force=True)
            if status:
                self._snapshot = self._map_status(status)
            if self._snapshot.phase not in {Phase.COMPLETED, Phase.CANCELLED, Phase.FAILED}:
                detail = self._read_worker_error()
                self._snapshot = TaskSnapshot(
                    phase=Phase.FAILED,
                    protocol="ssvep",
                    error=detail or f"Acquisition worker exited with code {return_code}",
                )
        return self._snapshot

    def read_live_waveform(self, maximum_rows: int = 1000) -> dict[str, Any] | None:
        """Read only newly flushed samples from the worker's live preview file."""

        if maximum_rows <= 0:
            return None
        if self._runtime_dir is None:
            return None
        path = self._runtime_dir / "live_waveform.tsv"
        if not path.is_file():
            return None
        try:
            if self._live_waveform_path != path or path.stat().st_size < self._live_waveform_position:
                self._live_waveform_path = path
                self._live_waveform_position = 0
                self._live_waveform_header = []
            with path.open("r", encoding="utf-8", newline="") as handle:
                if not self._live_waveform_header:
                    header_line = handle.readline()
                    if not header_line:
                        return None
                    self._live_waveform_header = next(csv.reader([header_line.rstrip("\r\n")], delimiter="\t"), [])
                    self._live_waveform_position = handle.tell()
                handle.seek(self._live_waveform_position)
                rows: list[list[str]] = []
                while len(rows) < maximum_rows:
                    line_position = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(("\n", "\r")):
                        handle.seek(line_position)
                        break
                    rows.append(next(csv.reader([line.rstrip("\r\n")], delimiter="\t"), []))
                self._live_waveform_position = handle.tell()
        except (OSError, UnicodeError, csv.Error):
            return None
        if not rows:
            return None

        eeg_columns = sorted(
            (
                (index, name)
                for index, name in enumerate(self._live_waveform_header)
                if name.casefold().startswith("eeg_ch")
            ),
            key=lambda item: int(item[1][6:]) if item[1][6:].isdigit() else 0,
        )
        if not eeg_columns:
            return None
        samples = [[] for _ in eeg_columns]
        for row in rows:
            for output_index, (column_index, _name) in enumerate(eeg_columns):
                try:
                    samples[output_index].append(float(row[column_index]))
                except (IndexError, TypeError, ValueError):
                    samples[output_index].append(float("nan"))
        channel_names = self._live_channel_names or channel_labels_from_path(
            self.channel_config_path, len(eeg_columns)
        )
        return {
            "channel_names": channel_names[: len(eeg_columns)],
            "sample_rate_hz": int(self.device.sample_rate or 250),
            "samples": samples,
        }

    def cancel(self) -> TaskSnapshot:
        if self._process is None or not self.snapshot.active:
            return self._snapshot
        assert self._cancel_file is not None
        self._cancel_file.write_text("cancel\n", encoding="ascii")

        # Cancellation is a persistence operation as well as a process
        # operation. On fast Linux runners the worker can exit between the
        # first poll and the final atomic status write, so keep polling until
        # the terminal status includes the saved session result.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = self._read_status(force=True)
            if status:
                mapped = self._map_status(status)
                if mapped.phase is Phase.CANCELLED and mapped.result is not None:
                    self._snapshot = mapped
                    if self._process.poll() is not None:
                        return self._snapshot
            if self._process.poll() is not None:
                # The worker has exited; give its final status/session write a
                # short grace period before deciding that it failed to persist.
                deadline = min(deadline, time.monotonic() + 1.0)
            time.sleep(0.02)

        try:
            self._process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
        status = self._read_status(force=True)
        if status:
            self._snapshot = self._map_status(status)
        if self._snapshot.phase not in {Phase.CANCELLED, Phase.COMPLETED}:
            self._snapshot = TaskSnapshot(phase=Phase.CANCELLED, protocol="ssvep")
        return self._snapshot

    def _read_status(self, *, force: bool = False) -> dict[str, Any]:
        if self._status_file is None:
            return {}
        try:
            stat = self._status_file.stat()
            if not force and stat.st_mtime_ns == self._last_status_mtime_ns:
                return self._last_status
            value = json.loads(self._status_file.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return self._last_status
            self._last_status_mtime_ns = stat.st_mtime_ns
            self._last_status = value
            return value
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return self._last_status

    def _read_worker_error(self) -> str:
        if self._stderr_file is None:
            return ""
        try:
            text = self._stderr_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""
        return text[-2000:]

    def _map_status(self, value: dict[str, Any]) -> TaskSnapshot:
        selected_port = str(value.get("serial_port") or "").strip()
        if selected_port and self.config.mode == CaptureMode.CYTON:
            self.device = replace(self.device, port=selected_port)
        raw_phase = str(value.get("phase", ""))
        phase = {
            "preparing": Phase.COUNTDOWN,
            "countdown": Phase.COUNTDOWN,
            "running": Phase.RUNNING,
            "completed": Phase.COMPLETED,
            "aborted": Phase.CANCELLED,
            "error": Phase.FAILED,
        }.get(raw_phase, self._snapshot.phase)
        duration = float(value.get("recording_duration_s", self.config.recording_seconds))
        elapsed = max(0.0, float(value.get("recording_elapsed_s", 0.0)))
        trial_index = int(value.get("trial_index", -1))
        result = None
        if phase in {Phase.COMPLETED, Phase.CANCELLED, Phase.FAILED}:
            result = self._dataset_from_terminal(value)
            if result is not None and not any(item.id == result.id for item in self._datasets):
                self._datasets.append(result)
        return TaskSnapshot(
            phase=phase,
            protocol="ssvep",
            countdown=max(0, int(value.get("countdown_remaining_s", 0))),
            elapsed=elapsed,
            remaining=max(0.0, duration - elapsed),
            demo_elapsed=elapsed,
            progress=100.0 if phase is Phase.COMPLETED else (0.0 if duration <= 0 else min(100.0, elapsed / duration * 100.0)),
            trial=0 if trial_index < 0 else trial_index + 1,
            trial_count=int(value.get("trial_count", self.config.trials)),
            target=int(value.get("target_index", -1)) + 1,
            frequency=int(value.get("frequency_hz", 0) or 0),
            resting=value.get("trial_phase") == "rest",
            event_count=int(value.get("event_count", 0)),
            speed=1,
            result=result,
            error=str(value.get("result", {}).get("error") or "") if isinstance(value.get("result"), dict) else "",
        )

    def _persist_diagnostics(self, output: Path) -> tuple[str, ...]:
        """Copy worker diagnostics into the durable session directory."""
        if not output.is_dir():
            return ()
        copied: list[str] = []
        for source, name in ((self._stderr_file, "worker.log"), (self._status_file, "status.final.json")):
            if source is None or not source.is_file():
                continue
            target = output / name
            try:
                shutil.copyfile(source, target)
            except OSError:
                continue
            copied.append(name)
        session_path = output / "session.json"
        if not session_path.is_file():
            return tuple(copied)
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
            if not isinstance(session, dict):
                return tuple(copied)
            existing = session.get("diagnostic_files", ())
            names = list(dict.fromkeys([str(item) for item in existing if isinstance(item, str)] + copied))
            session["diagnostic_files"] = names
            write_json(session_path, session)
            files = [item for item in sorted(output.iterdir()) if item.is_file() and item.name != "manifest.csv"]
            write_manifest(output / "manifest.csv", str(session.get("session_id") or output.name), files)
        except (OSError, ValueError, TypeError):
            return tuple(copied)
        return tuple(copied)

    def _dataset_from_terminal(self, value: dict[str, Any]) -> Dataset | None:
        result = value.get("result")
        if not isinstance(result, dict):
            return None
        output = Path(str(result.get("output_dir", "")))
        self._persist_diagnostics(output)
        session_path = output / "session.json"
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if str(session.get("board", "")) != "cyton":
            return None
        source = CaptureMode.CYTON
        quality: dict[str, Any] = {}
        try:
            loaded_quality = json.loads((output / "quality.json").read_text(encoding="utf-8"))
            if isinstance(loaded_quality, dict):
                quality = loaded_quality
        except (OSError, json.JSONDecodeError):
            pass
        channels = quality.get("channels", {})
        flat_channel_count = sum(
            1 for item in channels.values()
            if isinstance(item, dict) and float(item.get("flat_fraction", 0.0) or 0.0) >= 0.95
        ) if isinstance(channels, dict) else 0
        created_at = str(session.get("started_at") or datetime.now(timezone.utc).isoformat())
        return Dataset(
            id=str(session["session_id"]),
            name=str(session.get("session_name", self.config.name)),
            participant=str(session.get("participant_id", self.config.participant)),
            protocol="ssvep",
            recording_seconds=float(session.get("recording_duration_s", 0)),
            preparation_seconds=5,
            demo_seconds=float(session.get("recording_duration_s", 0)) + 5,
            trials=int(session.get("completed_trials", 0)),
            samples_per_channel=int(session.get("recorded_samples_per_channel", 0)),
            event_count=int(session.get("event_count", 0)),
            path=output.resolve(),
            created_at=created_at,
            simulated=bool(session.get("simulated", False)),
            persisted=True,
            source=source,
            status=str(session.get("status", "completed")),
            channel_count=int(session.get("channel_count", 8)),
            error=str(session.get("error") or ""),
            sampling_rate_hz=int(session.get("sampling_rate_hz", 250) or 250),
            files=tuple(
                str(item)
                for item in session.get("files", ())
                if isinstance(item, str)
            ) or tuple(
                item.name
                for item in sorted(output.iterdir())
                if item.is_file() and item.name != "session.json"
            ) if output.is_dir() else (),
            source_path=str(session.get("source_path") or ""),
            imported=bool(session.get("imported", False)),
            origin=str(session.get("origin") or "acquired"),
            user_id=str(session.get("user_id") or ""),
            user_name=str(session.get("user_name") or ""),
            user_link_status=str(session.get("user_link_status") or ("active" if session.get("user_id") else "unlinked")),
            validation_mode=str(session.get("validation_mode") or "technical_validation"),
            protocol_status=str((session.get("protocol_provenance") or {}).get("status") or ""),
            protocol_sha256=str((session.get("protocol_provenance") or {}).get("sha256") or ""),
            channel_config_sha256=str((session.get("channel_config_provenance") or {}).get("sha256") or ""),
            quality_status=str(quality.get("status") or "unknown"),
            timestamp_gap_count=int(quality.get("timestamp_gap_count", 0) or 0),
            dropped_frame_count=int(quality.get("dropped_frame_count", 0) or 0),
            flat_channel_count=flat_channel_count,
        )
