"""Metadata-only diagnostics for the NeuroStation desktop workstation."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback
from typing import Any

from neurostation_contract import (
    PRODUCT_NAME,
    PRODUCT_SEMVER,
    PRODUCT_VERSION,
    default_user_channel_config_path,
    default_user_protocol_path,
)


ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _dependency_status(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
    except Exception as error:  # pragma: no cover - native runtime specific
        return {
            "available": False,
            "version": None,
            "error": f"{type(error).__name__}: {error}",
        }
    installed_version = getattr(module, "__version__", None)
    if not installed_version:
        try:
            installed_version = package_version(module_name)
        except PackageNotFoundError:
            pass
    return {
        "available": True,
        "version": str(installed_version) if installed_version else None,
    }


class DiagnosticStore:
    """Thread-safe rolling event store with optional JSONL persistence."""

    _hook_lock = threading.Lock()
    _hook_installed = False
    _hook_store: "DiagnosticStore | None" = None

    def __init__(
        self,
        root: Path | None = None,
        *,
        persist: bool = True,
        max_events: int = 200,
    ) -> None:
        configured = os.environ.get("NEUROSTATION_DIAGNOSTICS")
        default_root = Path.home() / "Documents" / PRODUCT_NAME / "Diagnostics"
        self.root = Path(configured).expanduser().absolute() if configured else default_root.absolute()
        if root is not None:
            self.root = Path(root).expanduser().absolute()
        self.persist = bool(persist)
        self.log_path = self.root / "workstation-events.jsonl"
        self.max_events = max(20, int(max_events))
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._lock = threading.RLock()
        if self.persist:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_recent()

    def _load_recent(self) -> None:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError, UnicodeError):
            return
        for line in lines[-self.max_events:]:
            try:
                value = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                self._events.append(value)

    def record(
        self,
        level: str,
        source: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": _now(),
            "level": str(level).upper(),
            "source": str(source),
            "message": str(message),
            "context": _json_safe(context or {}),
        }
        with self._lock:
            self._events.append(event)
            if self.persist:
                try:
                    self.root.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(
                            json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                except OSError:
                    pass
        return event

    def info(
        self,
        source: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.record("INFO", source, message, context=context)

    def warning(
        self,
        source: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.record("WARNING", source, message, context=context)

    def error(
        self,
        source: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.record("ERROR", source, message, context=context)

    def exception(
        self,
        error: BaseException,
        *,
        source: str,
        message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        details = dict(context or {})
        details.update(
            {
                "exception_type": type(error).__name__,
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )[-8000:],
            }
        )
        return self.error(
            source,
            message or str(error) or type(error).__name__,
            context=details,
        )

    def events(self, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        with self._lock:
            values = list(self._events)
        if limit is not None:
            count = max(0, int(limit))
            values = values[-count:] if count else []
        return tuple(values)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            if self.persist:
                try:
                    self.log_path.write_text("", encoding="utf-8")
                except OSError:
                    pass

    def install_exception_hook(self) -> None:
        """Capture uncaught process and Python thread exceptions."""

        with self._hook_lock:
            type(self)._hook_store = self
            if type(self)._hook_installed:
                return
            previous = sys.excepthook

            def excepthook(exc_type, error, tb):
                if issubclass(exc_type, KeyboardInterrupt):
                    previous(exc_type, error, tb)
                    return
                store = type(self)._hook_store
                if store is not None:
                    store.error(
                        "process",
                        str(error) or str(exc_type),
                        context={
                            "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                            "traceback": "".join(
                                traceback.format_exception(exc_type, error, tb)
                            )[-8000:],
                        },
                    )
                previous(exc_type, error, tb)

            sys.excepthook = excepthook
            if hasattr(threading, "excepthook"):
                previous_thread_hook = threading.excepthook

                def thread_excepthook(args):
                    store = type(self)._hook_store
                    if store is not None:
                        store.error(
                            "thread",
                            str(args.exc_value) or str(args.exc_type),
                            context={
                                "thread": getattr(args.thread, "name", ""),
                                "exception_type": getattr(
                                    args.exc_type, "__name__", str(args.exc_type)
                                ),
                                "traceback": "".join(
                                    traceback.format_exception(
                                        args.exc_type,
                                        args.exc_value,
                                        args.exc_traceback,
                                    )
                                )[-8000:],
                            },
                        )
                    previous_thread_hook(args)

                threading.excepthook = thread_excepthook
            type(self)._hook_installed = True

    def build_report(self, gateway: Any | None = None) -> dict[str, Any]:
        """Collect JSON-safe runtime checks without opening hardware."""

        bundled_protocol_path = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
        protocol_path = (
            default_user_protocol_path()
            if default_user_protocol_path().is_file()
            else bundled_protocol_path
        )
        stimulus_path = ROOT / "configs" / "ssvep_config_v1.json"
        final_channel_path = default_user_channel_config_path()
        bundled_final_channel_path = ROOT / "configs" / "channel_config_v1.json"
        auto_channel_path = ROOT / "configs" / "channel_config_v1_auto.json"
        template_channel_path = ROOT / "configs" / "channel_config_v1_template.json"
        channel_path = (
            final_channel_path
            if final_channel_path.is_file()
            else (
                bundled_final_channel_path
                if bundled_final_channel_path.is_file()
                else (auto_channel_path if auto_channel_path.is_file() else template_channel_path)
            )
        )

        configuration: dict[str, Any] = {
            "protocol_path": str(protocol_path),
            "stimulus_path": str(stimulus_path),
            "channel_config_path": str(channel_path),
            "final_channel_config_present": (
                final_channel_path.is_file() or bundled_final_channel_path.is_file()
            ),
            "status": "invalid",
            "warnings": [],
            "error": None,
        }
        try:
            from eeg_tools.config import validate_channel_config
            from eeg_tools.workstation.ssvep import SSVEPProtocol

            protocol_value = json.loads(protocol_path.read_text(encoding="utf-8"))
            channel_value = json.loads(channel_path.read_text(encoding="utf-8"))
            protocol = SSVEPProtocol.load(protocol_path)
            warnings = validate_channel_config(channel_value, channel_path)
            protocol_status = str(protocol_value.get("status") or "")
            configuration["protocol_status"] = protocol_status
            if protocol_status.lower().startswith("draft"):
                warnings.append("The protocol is marked as a draft.")
            if (
                protocol.refresh_rate_hz != 60
                or protocol.sampling_rate_hz != 250
                or protocol.channel_count != 8
            ):
                warnings.append("Protocol hardware or display parameters are not the formal 8-channel/250 Hz/60 Hz setup.")
            configuration["warnings"] = list(warnings)
            configuration["status"] = "ready" if not warnings else "draft"
        except Exception as error:
            configuration["error"] = f"{type(error).__name__}: {error}"

        displays: list[dict[str, Any]] = []
        try:
            from PySide6.QtGui import QGuiApplication

            application = QGuiApplication.instance()
            if application is not None:
                for index, screen in enumerate(application.screens()):
                    geometry = screen.geometry()
                    displays.append(
                        {
                            "index": index,
                            "name": str(screen.name()),
                            "geometry": {
                                "x": int(geometry.x()),
                                "y": int(geometry.y()),
                                "width": int(geometry.width()),
                                "height": int(geometry.height()),
                            },
                            "device_pixel_ratio": float(screen.devicePixelRatio()),
                            "refresh_rate_hz": float(screen.refreshRate() or 0.0),
                        }
                    )
        except Exception as error:  # pragma: no cover - display specific
            # A headless or dependency-light CLI should report no displays,
            # matching the desktop preflight contract without fabricating one.
            displays = []

        serial_ports: list[dict[str, Any]] = []
        serial_scan_error = None
        if gateway is not None:
            try:
                serial_ports = [_json_safe(item) for item in gateway.scan_serial_ports()]
            except Exception as error:
                serial_scan_error = f"{type(error).__name__}: {error}"

        device: dict[str, Any] = {}
        snapshot: dict[str, Any] = {}
        save_directory = ""
        dataset_count = 0
        gateway_name = type(gateway).__name__ if gateway is not None else "none"
        if gateway is not None:
            try:
                value = gateway.device
                device = {
                    "name": str(value.name),
                    "port": str(value.port),
                    "channels": int(value.channels),
                    "sample_rate": int(value.sample_rate),
                    "connected": bool(value.connected),
                    "simulated": bool(value.simulated),
                }
            except Exception as error:
                device = {"error": f"{type(error).__name__}: {error}"}
            try:
                value = gateway.snapshot
                snapshot = {
                    "phase": str(
                        value.phase.value if hasattr(value.phase, "value") else value.phase
                    ),
                    "protocol": str(value.protocol),
                    "active": bool(value.active),
                    "elapsed_s": round(float(value.elapsed), 3),
                    "progress": round(float(value.progress), 2),
                    "error": str(value.error or ""),
                }
            except Exception as error:
                snapshot = {"error": f"{type(error).__name__}: {error}"}
            try:
                save_directory = str(Path(gateway.config.save_directory).expanduser().absolute())
            except Exception:
                save_directory = ""
            try:
                dataset_count = len(gateway.datasets)
            except Exception:
                dataset_count = 0

        writable = None
        if save_directory:
            try:
                path = Path(save_directory)
                writable = path.is_dir() and os.access(path, os.W_OK)
            except OSError:
                writable = False

        dependencies = {
            "PySide6": _dependency_status("PySide6"),
            "brainflow": _dependency_status("brainflow"),
            "numpy": _dependency_status("numpy"),
            "scipy": _dependency_status("scipy"),
        }
        resources = {
            "protocol_ready": protocol_path.is_file(),
            "stimulus_ready": stimulus_path.is_file(),
            "channel_config_ready": channel_path.is_file(),
            "locales_ready": all(
                (ROOT / "apps" / "workstation_ui" / "locales" / name).is_file()
                for name in ("zh-CN.json", "en-US.json")
            ),
        }
        checks = [
            {
                "name": "resources",
                "status": "passed" if all(resources.values()) else "failed",
                "detail_key": (
                    "diagnostics.detail.resources_ready"
                    if all(resources.values())
                    else "diagnostics.detail.resources_missing"
                ),
                "values": {},
            },
            {
                "name": "configuration",
                "status": configuration["status"],
                "detail_key": (
                    "diagnostics.detail.configuration_ready"
                    if configuration["status"] == "ready"
                    else "diagnostics.detail.configuration_warning"
                ),
                "detail": configuration["error"],
                "values": {"count": len(configuration["warnings"])},
            },
            {
                "name": "dependencies",
                "status": (
                    "passed"
                    if all(item["available"] for item in dependencies.values())
                    else "failed"
                ),
                "detail_key": (
                    "diagnostics.detail.dependencies_ready"
                    if all(item["available"] for item in dependencies.values())
                    else "diagnostics.detail.dependencies_missing"
                ),
                "values": {},
            },
            {
                "name": "save_directory",
                "status": "passed" if writable else ("warning" if writable is None else "failed"),
                "detail_key": (
                    "diagnostics.detail.save_ready"
                    if writable
                    else (
                        "diagnostics.detail.gateway_missing"
                        if writable is None
                        else "diagnostics.detail.save_failed"
                    )
                ),
                "values": {},
            },
            {
                "name": "serial_scan",
                "status": (
                    "failed" if serial_scan_error else ("passed" if serial_ports else "warning")
                ),
                "detail_key": (
                    "diagnostics.detail.serial_failed"
                    if serial_scan_error
                    else (
                        "diagnostics.detail.serial_ready"
                        if serial_ports
                        else "diagnostics.detail.serial_empty"
                    )
                ),
                "detail": serial_scan_error,
                "values": {"count": len(serial_ports)},
            },
        ]
        return {
            "generated_at": _now(),
            "application": {
                "name": PRODUCT_NAME,
                "version": PRODUCT_VERSION,
                "semantic_version": PRODUCT_SEMVER,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "displays": displays,
            "dependencies": dependencies,
            "resources": resources,
            "configuration": configuration,
            "hardware": {
                "device": device,
                "serial_ports": serial_ports,
                "serial_scan_error": serial_scan_error,
                "note": (
                    "Serial discovery does not open the device; Cyton readiness "
                    "is verified by hardware preflight and the acquisition worker."
                ),
            },
            "runtime": {
                "gateway": gateway_name,
                "task": snapshot,
                "dataset_count": dataset_count,
                "save_directory": save_directory,
                "save_directory_writable": writable,
                "diagnostics_directory": str(self.root),
                "event_count": len(self.events()),
            },
            "checks": checks,
        }

    def export_report(self, path: Path, gateway: Any | None = None) -> Path:
        output = Path(path).expanduser().absolute()
        output.parent.mkdir(parents=True, exist_ok=True)
        report = self.build_report(gateway)
        report["events"] = list(self.events())
        temporary = output.with_suffix(output.suffix + ".pending")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(output)
        self.info(
            "diagnostics",
            "diagnostic report exported",
            context={"path": output},
        )
        return output


__all__ = ["DiagnosticStore"]
