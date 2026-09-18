"""UI-facing contract exports.

The production UI is backed by ``DesktopGateway``. This module only
re-exports the shared contract so the UI cannot silently fall back to a
simulated acquisition implementation.
"""

from neurostation_contract import (
    CHANNEL_COUNT,
    FREQUENCIES,
    PREPARATION_SECONDS,
    SAMPLE_RATE,
    CaptureConfig,
    CaptureGateway,
    CaptureMode,
    EyeSide,
    Dataset,
    DeviceInfo,
    OpenBCIImportReport,
    OpenBCIWorkspaceStatus,
    Phase,
    TaskSnapshot,
    default_save_directory,
    format_duration,
)


__all__ = [
    "CHANNEL_COUNT",
    "FREQUENCIES",
    "PREPARATION_SECONDS",
    "SAMPLE_RATE",
    "CaptureConfig",
    "CaptureGateway",
    "CaptureMode",
    "EyeSide",
    "Dataset",
    "DeviceInfo",
    "OpenBCIImportReport",
    "OpenBCIWorkspaceStatus",
    "Phase",
    "TaskSnapshot",
    "default_save_directory",
    "format_duration",
]
