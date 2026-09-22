"""Qt pages; all acquisition commands are emitted to the window's gateway owner."""
from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import math
import re
from typing import Any

from PySide6.QtCore import Signal, Qt, QUrl
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QProgressBar, QStyle,
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QMessageBox, QTextEdit, QPushButton, QMenu,
)
from PySide6.QtGui import QDesktopServices

from .components import (
    CHANNEL_NAMES,
    Page,
    Section,
    KeyValues,
    AppTile,
    StaticTargets,
    WaveformWidget,
    HeadElectrodeMap,
    CalibrationSignalWidget,
    action,
    label,
)
from .gateway import CaptureConfig, CaptureMode, TaskSnapshot, Phase, Dataset, format_duration
from neurostation_contract import (
    GENDERS,
    MEDICAL_OPTIONS,
    DeviceInfo,
    UserProfile,
    dataset_name_for_eye,
    is_confirmed_position,
)


def _device_connection_text(tr, device: DeviceInfo | None) -> str:
    if device is not None and device.connected and not device.simulated:
        return tr("device.connection.connected", port=device.port or "AUTO")
    return tr("device.connection.waiting")


def _readonly_table(headers: tuple[str, ...]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    table.setAlternatingRowColors(True)
    # Fill rows before enabling sorting; QTableWidget can move the active row
    # while individual cells are being inserted when sorting is already on.
    table.setSortingEnabled(False)
    table.setWordWrap(False)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.setMinimumHeight(72)
    return table


def _table_item(value: object) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    item.setToolTip(str(value))
    return item


def _format_file_size(size: int | None) -> str:
    if size is None:
        return "—"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _dataset_timestamp(timestamp: str) -> float:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("-inf")


def _format_dataset_timestamp(timestamp: str) -> str:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return timestamp or "—"


def _dataset_file_rows(result: Dataset) -> list[tuple[str, str, str, str]]:
    """Build a metadata-only file table; raw EEG content is never loaded."""

    names = result.files
    if not names and result.path.is_dir():
        try:
            names = tuple(
                sorted(
                    path.relative_to(result.path).as_posix()
                    for path in result.path.rglob("*")
                    if path.is_file() and path.name not in {"session.json", "session.json.pending"}
                )
            )
        except OSError:
            names = ()

    source_metadata: dict[str, dict[str, object]] = {}
    metadata_path = result.path / "session.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_files = metadata.get("raw_files", ())
        if isinstance(raw_files, list):
            source_metadata = {
                str(item["name"]): item
                for item in raw_files
                if isinstance(item, dict) and item.get("name")
            }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    rows: list[tuple[str, str, str, str]] = []
    for name in names:
        path = _dataset_file_path(result, name)
        size: int | None = None
        modified_ns: int | None = None
        try:
            stat = path.stat()
            size = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            detail = source_metadata.get(name, {})
            try:
                size = int(detail.get("size_bytes"))
            except (TypeError, ValueError):
                size = None
            try:
                modified_ns = int(detail.get("modified_ns"))
            except (TypeError, ValueError):
                modified_ns = None
        suffix = Path(name).suffix.lower().lstrip(".")
        kind = suffix.upper() if suffix else "FILE"
        modified = "—"
        if modified_ns is not None:
            try:
                modified = datetime.fromtimestamp(modified_ns / 1_000_000_000).astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            except (OSError, OverflowError, ValueError):
                pass
        rows.append((name, kind, _format_file_size(size), modified))
    return rows


def _dataset_file_path(result: Dataset, name: str) -> Path:
    """Resolve a displayed dataset file to its local on-disk path."""

    path = result.path / name
    # Imported sessions reserve session.json for workstation metadata and
    # retain the original source session.json as source_session.json.
    if (
        result.imported
        and name == "session.json"
        and (result.path / "source_session.json").is_file()
    ):
        return result.path / "source_session.json"
    if not path.is_file() and name == "session.json":
        return result.path / "source_session.json"
    return path


def _open_file_manager_folder(path: Path) -> bool:
    """Open the system file manager at a file's containing directory."""

    folder = path.parent
    if not folder.is_dir():
        return False
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))


def _dataset_raw_file_names(result: Dataset) -> tuple[str, ...]:
    names = result.files
    if not names and result.path.is_dir():
        try:
            names = tuple(
                sorted(
                    path.relative_to(result.path).as_posix()
                    for path in result.path.rglob("*")
                    if path.is_file()
                )
            )
        except OSError:
            names = ()
    return tuple(
        name
        for name in names
        if (
            re.fullmatch(r"BrainFlow-RAW_.*\.csv", name, re.IGNORECASE)
            or re.fullmatch(r"OpenBCI-RAW-.*\.txt", name, re.IGNORECASE)
            or Path(name).name.lower() == "raw_brainflow.tsv"
        )
    )


def _load_denoising_metrics(result: Dataset) -> tuple[Path, dict[str, Any] | None]:
    candidates = (
        result.path / "derived" / "denoise_v1" / "denoising_metrics.json",
        result.path / "denoising_metrics.json",
    )
    for metrics_path in candidates:
        try:
            value = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("tracks"), dict):
            return metrics_path.parent, value
    return candidates[0].parent, None


def _load_denoising_preprocessing(output_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((output_dir / "preprocessing.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _denoising_parameter_text(tr, preprocessing: dict[str, Any]) -> str:
    config = preprocessing.get("pipeline_config", {})
    if not isinstance(config, dict):
        return "—"
    bandpass = config.get("bandpass_hz", [])
    line_noise = config.get("line_noise_hz", [])
    try:
        bandpass_text = f"{float(bandpass[0]):g}–{float(bandpass[1]):g} Hz"
    except (IndexError, TypeError, ValueError):
        bandpass_text = "—"
    try:
        line_text = ", ".join(f"{float(value):g} Hz" for value in line_noise) or "—"
    except (TypeError, ValueError):
        line_text = "—"
    order = config.get("bandpass_order", "—")
    quality = config.get("notch_quality_factor", "—")
    max_gap = config.get("max_interpolation_s", "—")
    return tr(
        "result.denoising.parameters_value",
        bandpass=bandpass_text,
        order=order,
        line_noise=line_text,
        quality=quality,
        max_gap=max_gap,
    )


def _denoising_metadata_table(tr, metrics: dict[str, Any], preprocessing: dict[str, Any]) -> QTableWidget:
    table = _readonly_table((tr("result.denoising.metadata"), tr("result.denoising.value")))
    table.setObjectName("denoisingMetadataTable")
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    input_hashes = preprocessing.get("input_hashes", {})
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    hash_rows = []
    for name in ("raw_brainflow.tsv", "raw_columns.tsv", "session.json", "events.tsv", "protocol.json"):
        digest = str(input_hashes.get(name) or "").strip()
        if digest:
            hash_rows.append(f"{name}: {digest}")
    pipeline_hash = str(preprocessing.get("pipeline_sha256") or "").strip()
    if pipeline_hash:
        hash_rows.append(f"pipeline.json: {pipeline_hash}")
    rows = [
        (tr("result.denoising.parameters"), _denoising_parameter_text(tr, preprocessing)),
        (tr("result.denoising.input_hash"), "\n".join(hash_rows) or "—"),
        (tr("result.denoising.generated_at"), str(metrics.get("generated_at") or "—")),
    ]
    for row_values in rows:
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(row_values):
            table.setItem(row, column, _table_item(value))
    table.resizeRowsToContents()
    return table


def _denoising_section(tr, result: Dataset) -> Section | None:
    output_dir, metrics = _load_denoising_metrics(result)
    if metrics is None:
        return None
    preprocessing = _load_denoising_preprocessing(output_dir)
    section = Section(tr("result.denoising.title"))
    section.layout.addWidget(label(tr("result.denoising.note"), "muted"))
    section.layout.addWidget(_denoising_metadata_table(tr, metrics, preprocessing))
    table = _readonly_table((tr("result.denoising.track"), tr("result.denoising.line_noise"), tr("result.denoising.target_snr"), tr("result.denoising.trials"), tr("result.denoising.rejected")))
    table.setObjectName("denoisingMetricsTable")
    header = table.horizontalHeader()
    for column in range(4):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
    tracks = metrics.get("tracks", {})
    analyses = metrics.get("analysis", {})
    for key, title_key in (("annotated", "result.denoising.annotated"), ("50hz", "result.denoising.50hz"), ("60hz", "result.denoising.60hz")):
        track = tracks.get(key, {}) if isinstance(tracks, dict) else {}
        line_values = track.get("line_noise_ratio", {}) if isinstance(track, dict) else {}
        line_text = ", ".join(
            f"{frequency} Hz: {10.0 * math.log10(max(float(value), 1e-12)):.1f} dB"
            for frequency, value in line_values.items()
            if value is not None
        ) or "—"
        target_values = track.get("target_snr_db", {}) if isinstance(track, dict) else {}
        snr_values = [float(value) for value in target_values.values() if value is not None]
        snr_text = f"{sum(snr_values) / len(snr_values):.1f} dB" if snr_values else "—"
        analysis = analyses.get(key, {}) if isinstance(analyses, dict) else {}
        values = (tr(title_key), line_text, snr_text, str(analysis.get("trial_count", "—")), str(analysis.get("rejected_trial_count", "—")))
        row = table.rowCount()
        table.insertRow(row)
        for column, value in enumerate(values):
            table.setItem(row, column, _table_item(value))
    table.resizeRowsToContents()
    section.layout.addWidget(table)
    section.layout.addWidget(label(tr("result.denoising.artifacts", count=metrics.get("artifact_segment_count", 0), fraction=f"{float(metrics.get('artifact_fraction', 0.0)) * 100:.2f}%"), "muted"))
    section.layout.addWidget(label(tr("result.denoising.generated", path=str(output_dir)), "path"))
    section.layout.addWidget(action(tr("result.denoising.open"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir)))))
    return section


_OPENBCI_BRAINFLOW_RAW_COLUMNS = (
    "package_num",
    "eeg_ch1",
    "eeg_ch2",
    "eeg_ch3",
    "eeg_ch4",
    "eeg_ch5",
    "eeg_ch6",
    "eeg_ch7",
    "eeg_ch8",
    "accel_x",
    "accel_y",
    "accel_z",
    "other_ch1",
    "other_ch2",
    "other_ch3",
    "other_ch4",
    "other_ch5",
    "other_ch6",
    "other_ch7",
    "analog_ch1",
    "analog_ch2",
    "analog_ch3",
    "timestamp_s",
    "marker",
)


def _headerless_brainflow_raw_columns(
    path: Path, column_count: int
) -> tuple[str, ...] | None:
    """Restore OpenBCI field names omitted by BrainFlow's raw CSV export."""

    if (
        column_count == len(_OPENBCI_BRAINFLOW_RAW_COLUMNS)
        and re.fullmatch(r"BrainFlow-RAW_.*\.csv", path.name, re.IGNORECASE)
    ):
        return _OPENBCI_BRAINFLOW_RAW_COLUMNS
    return None


def _fixed_decimal_text(value: str) -> str:
    """Render scientific notation without losing the source precision."""

    try:
        decimal = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        return value
    if not decimal.is_finite():
        return value
    return format(decimal, "f")


def _openbci_raw_timestamp_values(path: Path, limit: int) -> list[str]:
    """Read high-precision timestamps from the paired OpenBCI GUI export."""

    candidates = sorted(path.parent.glob("OpenBCI-RAW-*.txt"))
    for candidate in candidates:
        values: list[str] = []
        timestamp_index: int | None = None
        try:
            with candidate.open(
                "r", encoding="utf-8-sig", errors="replace", newline=""
            ) as handle:
                reader = csv.reader(handle, delimiter=",")
                for row in reader:
                    if not row or not any(cell.strip() for cell in row):
                        continue
                    first = row[0].strip()
                    if first.startswith("%"):
                        continue
                    if timestamp_index is None:
                        try:
                            float(first)
                        except ValueError:
                            timestamp_index = next(
                                (
                                    index
                                    for index, header in enumerate(row)
                                    if header.strip().casefold() == "timestamp"
                                ),
                                None,
                            )
                            continue
                        timestamp_index = 22
                    try:
                        float(first)
                    except ValueError:
                        continue
                    if timestamp_index is None or timestamp_index >= len(row):
                        continue
                    values.append(_fixed_decimal_text(row[timestamp_index]))
                    if len(values) >= limit:
                        break
        except (OSError, UnicodeError, csv.Error):
            continue
        if values:
            return values
    return []


def _restore_imported_timestamp_precision(
    path: Path, headers: tuple[str, ...], rows: list[list[str]]
) -> None:
    """Use the paired GUI export when BrainFlow CSV rounded timestamps."""

    if not re.fullmatch(r"BrainFlow-RAW_.*\.csv", path.name, re.IGNORECASE):
        return
    try:
        timestamp_index = next(
            index
            for index, header in enumerate(headers)
            if header.casefold() == "timestamp_s"
        )
    except StopIteration:
        return
    source_values = _openbci_raw_timestamp_values(path, len(rows))
    if not source_values or not rows:
        return
    try:
        if abs(float(rows[0][timestamp_index]) - float(source_values[0])) > 1e-3:
            return
    except (IndexError, ValueError):
        return
    for row, value in zip(rows, source_values):
        if timestamp_index < len(row):
            row[timestamp_index] = value


def _read_formal_timestamp_anchors(path: Path) -> list[tuple[int, str]]:
    """Read high-precision event timestamps saved beside a formal dataset."""

    events_path = path.parent / "events.tsv"
    if not events_path.is_file():
        return []
    anchors: dict[int, str] = {}
    try:
        with events_path.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                try:
                    sample_index = int(str(record.get("sample_index") or "").strip())
                    timestamp = _fixed_decimal_text(
                        str(record.get("sample_time_s") or "").strip()
                    )
                    if "." not in timestamp:
                        continue
                    float(timestamp)
                except (TypeError, ValueError, InvalidOperation):
                    continue
                anchors[sample_index] = timestamp
    except (OSError, UnicodeError, csv.Error):
        return []
    return sorted(anchors.items())


def _restore_formal_timestamp_precision(
    path: Path, headers: tuple[str, ...], rows: list[list[str]]
) -> None:
    """Recover preview precision for old formal files that stored integer timestamps.

    Older packaged builds wrote the device timestamp column as whole seconds,
    while the adjacent event log retained precise sample timestamps.  Use the
    event timestamps only for the read-only preview; never rewrite the raw file.
    """

    if path.name.casefold() != "raw_brainflow.tsv" or not rows:
        return
    try:
        timestamp_index = next(
            index
            for index, header in enumerate(headers)
            if header.casefold() == "timestamp_s"
        )
        sample_index_column = next(
            index
            for index, header in enumerate(headers)
            if header.casefold() == "sample_index"
        )
    except StopIteration:
        return
    if any(
        timestamp_index >= len(row)
        or "." in str(row[timestamp_index]).strip()
        for row in rows
    ):
        return
    anchors = _read_formal_timestamp_anchors(path)
    if len(anchors) < 2:
        return
    anchor_values = dict(anchors)

    def interpolate(sample_index: int) -> str | None:
        if sample_index <= anchors[0][0]:
            left, right = anchors[0], anchors[1]
        elif sample_index >= anchors[-1][0]:
            left, right = anchors[-2], anchors[-1]
        else:
            left, right = next(
                (pair for pair in zip(anchors, anchors[1:]) if pair[0][0] <= sample_index <= pair[1][0]),
                (anchors[-2], anchors[-1]),
            )
        left_index, left_value = left
        right_index, right_value = right
        if right_index == left_index:
            return left_value
        ratio = Decimal(sample_index - left_index) / Decimal(right_index - left_index)
        value = Decimal(left_value) + (Decimal(right_value) - Decimal(left_value)) * ratio
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text

    for row in rows:
        try:
            sample_index = int(str(row[sample_index_column]).strip())
        except (IndexError, ValueError):
            continue
        if sample_index in anchor_values:
            row[timestamp_index] = anchor_values[sample_index]
            continue
        value = interpolate(sample_index)
        if value is not None:
            row[timestamp_index] = value


def _read_data_preview(path: Path, limit: int = 100) -> tuple[tuple[str, ...], list[list[str]]]:
    """Read only a bounded prefix for the Excel-like read-only data preview."""

    rows: list[list[str]] = []
    headers: list[str] | None = None
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            first_line = handle.readline()
            if not first_line:
                return (), []
            delimiter = "\t" if "\t" in first_line else ","
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                first = row[0].strip()
                if first.startswith("%"):
                    continue
                try:
                    float(first)
                except ValueError:
                    if headers is None:
                        headers = row
                    continue
                rows.append(row)
                if len(rows) >= limit:
                    break
    except (OSError, UnicodeError, csv.Error):
        return (), []

    column_count = max((len(row) for row in rows), default=len(headers or ()))
    if column_count == 0:
        return (), []
    if headers is None:
        headers = list(
            _headerless_brainflow_raw_columns(path, column_count)
            or tuple(f"Column {index}" for index in range(1, column_count + 1))
        )
    elif len(headers) < column_count:
        headers.extend(
            f"Column {index}" for index in range(len(headers) + 1, column_count + 1)
        )
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    normalized_headers = tuple(headers[:column_count])
    _restore_imported_timestamp_precision(path, normalized_headers, normalized)
    _restore_formal_timestamp_precision(path, normalized_headers, normalized)
    return normalized_headers, normalized


def _sort_preview_rows_by_sample_index(
    headers: tuple[str, ...], rows: list[list[str]]
) -> list[list[str]]:
    """Present raw preview rows in stable numeric sample-index order."""

    try:
        sample_index_column = next(
            index
            for index, header in enumerate(headers)
            if header.casefold() == "sample_index"
        )
    except StopIteration:
        return rows

    def sort_key(item: tuple[int, list[str]]) -> tuple[bool, int, int]:
        original_position, row = item
        try:
            sample_index = int(row[sample_index_column].strip())
        except (IndexError, ValueError):
            return True, 0, original_position
        return False, sample_index, original_position

    return [row for _, row in sorted(enumerate(rows), key=sort_key)]


def _preview_header_labels(tr, headers: tuple[str, ...]) -> tuple[str, ...]:
    """Return localized display labels while retaining each raw field name."""

    labels: list[str] = []
    patterns = (
        (r"eeg_ch(\d+)", "dataset_summary.column.eeg"),
        (r"analog_ch(\d+)", "dataset_summary.column.analog"),
        (r"other_ch(\d+)", "dataset_summary.column.other"),
        (r"board_row_(\d+)", "dataset_summary.column.board_row"),
    )
    fixed = {
        "sample_index": "dataset_summary.column.sample_index",
        "package_num": "dataset_summary.column.package_num",
        "timestamp_s": "dataset_summary.column.timestamp",
        "marker": "dataset_summary.column.marker",
        "accel_x": "dataset_summary.column.accel_x",
        "accel_y": "dataset_summary.column.accel_y",
        "accel_z": "dataset_summary.column.accel_z",
    }
    for position, header in enumerate(headers, start=1):
        raw_name = str(header).strip()
        if not raw_name:
            labels.append(tr("dataset_summary.column.unnamed", index=position))
            continue

        key = fixed.get(raw_name.lower())
        values: dict[str, object] = {}
        if key is None:
            for pattern, candidate in patterns:
                match = re.fullmatch(pattern, raw_name, re.IGNORECASE)
                if match:
                    key = candidate
                    values["index"] = int(match.group(1))
                    break
        if key is None or raw_name.lower().startswith("column "):
            display = tr("dataset_summary.column.unnamed", index=position)
        else:
            display = tr(key, **values)
        labels.append(tr("dataset_summary.column.with_raw", label=display, raw=raw_name))
    return tuple(labels)


def _preview_column_indexes(headers: tuple[str, ...], group: str) -> tuple[int, ...]:
    """Select preview columns without changing their source-file order."""

    hidden_prefixes = ("other_ch", "analog_ch")
    if group == "all":
        return tuple(
            index
            for index, header in enumerate(headers)
            if not header.casefold().startswith(hidden_prefixes)
        )
    required = {"sample_index", "package_num", "timestamp_s", "marker"}
    return tuple(
        index
        for index, header in enumerate(headers)
        if not header.casefold().startswith(hidden_prefixes)
        and (
            header.casefold() in required
            or header.casefold().startswith("eeg_ch")
        )
    )


def _dataset_raw_preview_section(tr, result: Dataset) -> Section | None:
    raw_names = _dataset_raw_file_names(result)
    if not raw_names:
        return None

    preview = Section(tr("dataset_summary.data_preview"))
    preview.layout.addWidget(label(tr("dataset_summary.data_preview_note"), "muted"))
    preview_file = QComboBox()
    preview_file.addItems(list(raw_names))
    preview_file.setAccessibleName(tr("dataset_summary.preview_file"))
    preview.layout.addWidget(preview_file)
    preview_columns = QComboBox()
    preview_columns.setObjectName("datasetPreviewColumns")
    preview_columns.setAccessibleName(tr("dataset_summary.preview_columns"))
    preview_columns_form = QFormLayout()
    preview_columns_form.addRow(
        tr("dataset_summary.preview_columns"), preview_columns
    )
    preview.layout.addLayout(preview_columns_form)

    headers, values = _read_data_preview(result.path / raw_names[0])
    preview_table = _readonly_table(headers or (tr("dataset_summary.no_columns"),))
    preview_table.setObjectName("datasetRawPreviewTable")
    preview_table.setAccessibleName(tr("dataset_summary.raw_preview_table"))
    preview.layout.addWidget(preview_table)

    state_headers = headers
    state_values = _sort_preview_rows_by_sample_index(headers, values)

    def render_preview_table() -> None:
        preview_table.setSortingEnabled(False)
        preview_table.clearContents()
        indexes = _preview_column_indexes(
            state_headers, str(preview_columns.currentData() or "all")
        )
        if not state_headers:
            preview_table.setColumnCount(1)
            preview_table.setHorizontalHeaderLabels(
                [tr("dataset_summary.no_columns")]
            )
            preview_table.setRowCount(0)
            return

        display_headers = _preview_header_labels(tr, state_headers)
        preview_table.setColumnCount(len(indexes) + 1)
        preview_table.setHorizontalHeaderLabels(
            [tr("dataset_summary.column.preview_row")]
            + [display_headers[index] for index in indexes]
        )
        for column, index in enumerate(indexes, start=1):
            header_item = preview_table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(str(state_headers[index]))
        row_header = preview_table.horizontalHeaderItem(0)
        if row_header is not None:
            row_header.setToolTip(tr("dataset_summary.column.preview_row"))
        preview_table.setRowCount(0)
        for row_number, values_row in enumerate(state_values, start=1):
            row = preview_table.rowCount()
            preview_table.insertRow(row)
            preview_table.setItem(row, 0, _table_item(row_number))
            for column, index in enumerate(indexes, start=1):
                value = values_row[index] if index < len(values_row) else ""
                preview_table.setItem(row, column, _table_item(value))
        preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        preview_table.horizontalScrollBar().setValue(
            preview_table.horizontalScrollBar().minimum()
        )
        preview_table.setMinimumHeight(
            min(440, max(108, 30 * min(10, max(1, len(state_values))) + 44))
        )
        preview_table.setSortingEnabled(False)

    def set_preview_table(new_headers: tuple[str, ...], new_values: list[list[str]]) -> None:
        nonlocal state_headers, state_values
        state_headers = new_headers
        state_values = _sort_preview_rows_by_sample_index(new_headers, new_values)
        current_group = preview_columns.currentData()
        previous_group = str(current_group) if current_group else ""
        groups = [("all", len(_preview_column_indexes(new_headers, "all")))]
        preview_columns.blockSignals(True)
        preview_columns.clear()
        for group, count in groups:
            preview_columns.addItem(
                tr(f"dataset_summary.preview_column_group.{group}", count=count),
                group,
            )
        target_group = previous_group
        if not target_group:
            target_group = "all"
        target_index = preview_columns.findData(target_group)
        preview_columns.setCurrentIndex(max(0, target_index))
        preview_columns.blockSignals(False)
        render_preview_table()

    def preview_file_changed(name: str) -> None:
        set_preview_table(*_read_data_preview(result.path / name))

    set_preview_table(headers, values)
    preview_file.currentTextChanged.connect(preview_file_changed)
    preview_columns.currentIndexChanged.connect(
        lambda _index: render_preview_table()
    )
    return preview


class UserDialog(QDialog):
    def __init__(self, tr, profile: UserProfile | None = None, next_id: str = "U0001"):
        super().__init__()
        self.tr = tr
        self.original_id = profile.user_id if profile else ""
        self.setWindowTitle(tr("users.edit_title") if profile else tr("users.add_title"))
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.user_id = QLineEdit(profile.user_id if profile else next_id)
        self.name = QLineEdit(profile.name if profile else "")
        self.age = QSpinBox()
        self.age.setRange(0, 150)
        self.age.setValue(profile.age if profile else 0)
        self.gender = QComboBox()
        for key in GENDERS:
            self.gender.addItem(tr("users.gender." + key), key)
        if profile:
            self.gender.setCurrentIndex(max(0, self.gender.findData(profile.gender)))
        form.addRow(tr("users.field_id"), self.user_id)
        form.addRow(tr("users.field_name"), self.name)
        form.addRow(tr("users.field_age"), self.age)
        form.addRow(tr("users.field_gender"), self.gender)
        layout.addLayout(form)
        medical = Section(tr("users.field_medical"))
        self.medical_checks: dict[str, QCheckBox] = {}
        current = set(profile.medical_conditions if profile else ("none",))
        grid = QGridLayout()
        for index, key in enumerate(MEDICAL_OPTIONS):
            check = QCheckBox(tr("users.medical." + key))
            check.setChecked(key in current)
            self.medical_checks[key] = check
            grid.addWidget(check, index // 2, index % 2)
        medical.layout.addLayout(grid)
        self.medical_other = QTextEdit(profile.medical_other if profile else "")
        self.medical_other.setPlaceholderText(tr("users.medical_other_placeholder"))
        self.medical_other.setMaximumHeight(72)
        medical.layout.addWidget(self.medical_other)
        layout.addWidget(medical)
        self.error = label("", "error")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def profile(self) -> UserProfile:
        conditions = tuple(key for key, check in self.medical_checks.items() if check.isChecked())
        return UserProfile(
            user_id=self.user_id.text().strip(),
            name=self.name.text().strip(),
            age=self.age.value(),
            gender=str(self.gender.currentData()),
            medical_conditions=conditions,
            medical_other=self.medical_other.toPlainText().strip(),
            is_demo=False,
        )

    def _accept(self):
        try:
            self.profile().validate()
        except ValueError as error:
            self.error.setText(self.tr(str(error)))
            return
        self.accept()


class UserManagementPage(Page):
    export_requested = Signal()
    def __init__(self, tr, users, trash, callbacks):
        super().__init__(tr, tr("nav.users"), tr("users.subtitle"))
        self._callbacks = callbacks
        self._users = tuple(users)
        self._trash = tuple(trash)
        controls = QHBoxLayout()
        self.add_button = action(tr("users.add"), self._add, True)
        self.edit_button = action(tr("users.edit"), self._edit)
        self.delete_button = action(tr("users.delete"), self._delete)
        self.export_button = action(tr("users.export_public"), self.export_requested.emit)
        controls.addWidget(self.add_button)
        controls.addWidget(self.edit_button)
        controls.addWidget(self.delete_button)
        controls.addWidget(self.export_button)
        controls.addStretch()
        self.layout.addLayout(controls)
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("users.search_placeholder"))
        self.search.setClearButtonEnabled(True)
        self.search.setObjectName("userSearch")
        self.search.textChanged.connect(lambda _text: self._refresh_tables())
        self.layout.addWidget(self.search)
        self.status = label("", "muted")
        self.layout.addWidget(self.status)
        self.table = _readonly_table((tr("users.field_id"), tr("users.field_name"), tr("users.field_age"),
                                      tr("users.field_gender"), tr("users.field_medical"),
                                      tr("users.updated"), tr("users.status")))
        self.table.setObjectName("usersTable")
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.layout.addWidget(self.table)
        trash_section = Section(tr("users.trash"))
        trash_controls = QHBoxLayout()
        self.restore_button = action(tr("users.restore"), self._restore)
        self.purge_button = action(tr("users.purge"), self._purge)
        trash_controls.addWidget(self.restore_button)
        trash_controls.addWidget(self.purge_button)
        trash_controls.addStretch()
        trash_section.layout.addLayout(trash_controls)
        self.trash_table = _readonly_table((tr("users.field_id"), tr("users.field_name"), tr("users.deleted_at")))
        self.trash_table.setObjectName("usersTrashTable")
        self.trash_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        trash_section.layout.addWidget(self.trash_table)
        self.layout.addWidget(trash_section)
        self._refresh_tables()

    def set_users(self, users, trash):
        self._users = tuple(users)
        self._trash = tuple(trash)
        self._refresh_tables()

    def _refresh_tables(self):
        self.table.setRowCount(0)
        query = self.search.text().strip().casefold() if hasattr(self, "search") else ""
        self._visible_users = tuple(
            profile for profile in self._users
            if not query or query in profile.user_id.casefold() or query in profile.name.casefold()
        )
        for profile in self._visible_users:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = (profile.user_id, profile.name, profile.age,
                      self.tr("users.gender." + profile.gender),
                      self.tr("users.screening_recorded") if profile.medical_conditions else self.tr("users.screening_missing"),
                      profile.updated_at or "—",
                      self.tr("users.active"))
            for col, value in enumerate(values):
                self.table.setItem(row, col, _table_item(value))
        self.trash_table.setRowCount(0)
        self._visible_trash = self._trash
        for profile in self._visible_trash:
            row = self.trash_table.rowCount()
            self.trash_table.insertRow(row)
            for col, value in enumerate((profile.user_id, profile.name, profile.deleted_at or "—")):
                self.trash_table.setItem(row, col, _table_item(value))
        self._selection_changed()
        self.status.setText(self.tr("users.empty") if not self._visible_users else "")

    def _selected(self, table):
        rows = table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        values = self._visible_users if table is self.table else self._visible_trash
        return values[index] if 0 <= index < len(values) else None

    def _selection_changed(self):
        self.edit_button.setEnabled(self._selected(self.table) is not None)
        self.delete_button.setEnabled(self._selected(self.table) is not None)
        self.restore_button.setEnabled(self._selected(self.trash_table) is not None)
        self.purge_button.setEnabled(self._selected(self.trash_table) is not None)

    def _add(self):
        dialog = UserDialog(self.tr, next_id=self._callbacks["next_id"]())
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self._callbacks["add"](dialog.profile())
        except ValueError as error:
            self.status.setText(self.tr(str(error)))
            return
        self._callbacks["refresh"]()

    def _edit(self):
        selected = self._selected(self.table)
        if selected is None:
            return
        dialog = UserDialog(self.tr, selected)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self._callbacks["update"](selected.user_id, dialog.profile())
        except ValueError as error:
            self.status.setText(self.tr(str(error)))
            return
        self._callbacks["refresh"]()

    def _delete(self):
        selected = self._selected(self.table)
        if selected is None or selected.is_demo:
            self.status.setText(self.tr("users.demo_protected"))
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle(self.tr("users.delete_title"))
        dialog.setText(self.tr("users.delete_prompt", name=selected.name, user_id=selected.user_id))
        keep = dialog.addButton(self.tr("users.delete_keep_data"), QMessageBox.ButtonRole.AcceptRole)
        move = dialog.addButton(self.tr("users.delete_move_data"), QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked not in (keep, move):
            return
        try:
            self._callbacks["delete"](selected.user_id, move_data=clicked is move)
            self._callbacks["refresh"]()
        except (ValueError, RuntimeError) as error:
            self.status.setText(self.tr(str(error)))

    def _restore(self):
        selected = self._selected(self.trash_table)
        if selected:
            try:
                self._callbacks["restore"](selected.user_id)
                self._callbacks["refresh"]()
            except (ValueError, RuntimeError) as error:
                self.status.setText(self.tr(str(error)))

    def _purge(self):
        selected = self._selected(self.trash_table)
        if selected is None:
            return
        answer = QMessageBox.question(self, self.tr("users.purge_title"),
                                      self.tr("users.purge_prompt", name=selected.name),
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self._callbacks["purge"](selected.user_id)
                self._callbacks["refresh"]()
            except (ValueError, RuntimeError) as error:
                self.status.setText(self.tr(str(error)))


class HomePage(Page):
    def __init__(self, tr, navigate, device_info: DeviceInfo | None = None):
        super().__init__(tr, tr("home.ready"), tr("home.subtitle"))
        device_section = Section("OpenBCI Cyton")
        device_section.layout.addWidget(label("AUTO · 8 CH · 250 Hz"))
        self.device_status = label(_device_connection_text(tr, device_info), "muted")
        device_section.layout.addWidget(self.device_status)
        self.layout.addWidget(device_section)
        self.layout.addWidget(action(tr("home.check"), lambda: navigate("devices")))
        self.layout.addWidget(action(tr("home.apps"), lambda: navigate("apps"), True))
        self.layout.addStretch()

    def set_device_info(self, device: DeviceInfo) -> None:
        self.device_status.setText(_device_connection_text(self.tr, device))


class DevicesPage(Page):
    DEFAULT_CHANNEL_POSITIONS = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
    ELECTRODE_POSITIONS = (
        "Fp1", "Fp2", "AF3", "AF4", "Fz", "Cz", "C3", "C4",
        "P3", "P4", "P7", "P8", "T7", "T8", "O1", "O2", "Oz",
    )
    AUXILIARY_POSITIONS = ("ear_clip", "left_earlobe", "right_earlobe", "mastoid")

    calibration_test_requested = Signal(int)
    calibration_test_stop_requested = Signal()
    calibration_save_requested = Signal(object)

    def __init__(
        self,
        tr,
        navigate,
        preflight=None,
        channel_test=None,
        save_calibration=None,
        channel_config=None,
        protocol_status="draft",
        channel_test_stop=None,
        device_info: DeviceInfo | None = None,
    ):
        super().__init__(tr, tr("nav.devices"), tr("device.serial_note"))
        device_section = Section("OpenBCI Cyton · AUTO")
        self.device_summary = KeyValues([
            (tr("device.title"), _device_connection_text(tr, device_info)),
            (tr("device.channels"), "8 CH / 250 Hz"),
            (tr("device.transport"), "USB Dongle"),
            (tr("device.owner"), "NeuroStation"),
        ])
        self.device_status = self.device_summary.values[0]
        device_section.layout.addWidget(self.device_summary)
        self.layout.addWidget(device_section)
        mapping = Section(tr("device.mapping"))
        mapping.layout.addWidget(label(tr("device.mapping_fixed")))
        mapping.layout.addWidget(label(tr("device.mapping_note"), "muted"))
        self.layout.addWidget(mapping)

        calibration = Section(tr("device.calibration_title"))
        calibration.layout.addWidget(label(tr("device.calibration_note"), "muted"))
        protocol_key = self._protocol_status_key(protocol_status)
        self.protocol_status = label(
            tr("device.protocol_status", status=tr("device.protocol_status." + protocol_key)),
            "muted",
        )
        calibration.layout.addWidget(self.protocol_status)

        channel_value = channel_config if isinstance(channel_config, dict) else {}
        self.calibration_port = QLineEdit(str(channel_value.get("serial_port") or "AUTO"))
        self.calibration_port.setPlaceholderText("AUTO")
        port_form = QFormLayout()
        port_form.addRow(tr("device.calibration_port"), self.calibration_port)
        calibration.layout.addLayout(port_form)

        visual = Section(tr("device.calibration_visual_title"))
        visual.layout.addWidget(label(tr("device.calibration_visual_note"), "muted"))
        visual_row = QHBoxLayout()
        self.head_map = HeadElectrodeMap()
        self.head_map.set_hint_text(tr("device.calibration_map_hint"))
        self.head_map.position_clicked.connect(self._assign_map_position)
        self.head_map.channel_clicked.connect(self._select_calibration_channel)
        visual_row.addWidget(self.head_map, 1)
        signal_panel = QWidget()
        signal_layout = QVBoxLayout(signal_panel)
        signal_layout.setContentsMargins(0, 0, 0, 0)
        current_form = QFormLayout()
        self.calibration_channel_selector = QComboBox()
        for index in range(8):
            self.calibration_channel_selector.addItem(f"CH{index + 1}", index)
        self.calibration_channel_selector.currentIndexChanged.connect(
            self._selected_channel_changed
        )
        current_form.addRow(tr("device.calibration_selected_channel"), self.calibration_channel_selector)
        signal_layout.addLayout(current_form)
        self.calibration_signal = CalibrationSignalWidget()
        self.calibration_signal.set_empty_text(tr("device.calibration_signal_waiting"))
        signal_layout.addWidget(self.calibration_signal, 1)
        self.calibration_test_toggle = action(
            tr("device.calibration_start_test"),
            self._toggle_channel_test,
            True,
        )
        self.calibration_test_toggle.setObjectName("startChannelCalibrationButton")
        signal_layout.addWidget(self.calibration_test_toggle)
        self.calibration_signal_status = label(tr("device.calibration_signal_idle"), "muted")
        signal_layout.addWidget(self.calibration_signal_status)
        visual_row.addWidget(signal_panel, 1)
        visual.layout.addLayout(visual_row)
        calibration.layout.addWidget(visual)

        self.calibration_table = QTableWidget(8, 5)
        self.calibration_table.setObjectName("channelCalibrationTable")
        self.calibration_table.setHorizontalHeaderLabels([
            tr("device.calibration_channel"),
            tr("device.calibration_input"),
            tr("device.calibration_position"),
            tr("device.calibration_result"),
            tr("device.calibration_action"),
        ])
        self.calibration_table.verticalHeader().setVisible(False)
        self.calibration_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.calibration_table.setMinimumHeight(320)
        self.calibration_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.calibration_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.calibration_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.position_boxes: list[QComboBox] = []
        self.test_buttons: list[QPushButton] = []
        self.test_statuses: list[QLabel] = []
        self._test_results: list[dict[str, Any] | None] = [None] * 8
        configured_channels = channel_value.get("channels")
        configured_channels = configured_channels if isinstance(configured_channels, list) else []
        for index in range(8):
            channel = configured_channels[index] if index < len(configured_channels) else {}
            channel = channel if isinstance(channel, dict) else {}
            self.calibration_table.setItem(index, 0, _table_item(f"CH{index + 1}"))
            self.calibration_table.setItem(index, 1, _table_item(str(channel.get("board_input") or f"N{index + 1}P")))
            position = QComboBox()
            position.setEditable(True)
            position.addItem("", "")
            position.addItems(self.ELECTRODE_POSITIONS)
            configured_position = str(channel.get("electrode_position") or "").strip()
            if configured_position.casefold() in {"", "unspecified", "未指定"}:
                configured_position = self.DEFAULT_CHANNEL_POSITIONS[index]
            position.setCurrentText(configured_position)
            if position.lineEdit() is not None:
                position.lineEdit().setPlaceholderText(tr("device.calibration_position_placeholder"))
            position.currentTextChanged.connect(self._calibration_form_changed)
            self.position_boxes.append(position)
            self.calibration_table.setCellWidget(index, 2, position)
            result_label = label(tr("device.calibration_not_tested"), "muted")
            self.test_statuses.append(result_label)
            self.calibration_table.setCellWidget(index, 3, result_label)
            test_button = action(
                tr("device.calibration_test"),
                lambda _checked=False, channel_index=index: self._test_channel(channel_index),
            )
            self.test_buttons.append(test_button)
            self.calibration_table.setCellWidget(index, 4, test_button)
            self.calibration_table.setRowHeight(index, 38)
        calibration.layout.addWidget(self.calibration_table)
        self._sync_head_map()

        auxiliary_form = QFormLayout()
        self.reference_position = self._editable_choice(
            self.AUXILIARY_POSITIONS,
            str((channel_value.get("reference") or {}).get("position") or "ear_clip"),
            tr("device.calibration_aux_placeholder"),
        )
        self.bias_position = self._editable_choice(
            self.AUXILIARY_POSITIONS,
            str((channel_value.get("bias") or {}).get("position") or "ear_clip"),
            tr("device.calibration_aux_placeholder"),
        )
        self.ground_position = self._editable_choice(
            self.AUXILIARY_POSITIONS,
            str((channel_value.get("ground") or {}).get("position") or ""),
            tr("device.calibration_aux_placeholder"),
        )
        auxiliary_form.addRow(tr("device.calibration_reference"), self.reference_position)
        auxiliary_form.addRow(tr("device.calibration_bias"), self.bias_position)
        auxiliary_form.addRow(tr("device.calibration_ground"), self.ground_position)
        calibration.layout.addLayout(auxiliary_form)
        for combo in (self.reference_position, self.bias_position, self.ground_position):
            combo.currentTextChanged.connect(self._calibration_form_changed)

        self.protocol_reviewed = QCheckBox(tr("device.protocol_reviewed"))
        self.protocol_reviewed.toggled.connect(self._calibration_form_changed)
        calibration.layout.addWidget(self.protocol_reviewed)
        self.calibration_status = label("", "muted")
        calibration.layout.addWidget(self.calibration_status)
        self.calibration_path = label("", "path")
        calibration.layout.addWidget(self.calibration_path)
        self.save_calibration_button = action(
            tr("device.save_formal_calibration"), self._save_calibration, True
        )
        calibration.layout.addWidget(self.save_calibration_button)
        self.layout.addWidget(calibration)

        self.preflight_status = label(tr("device.preflight_idle"), "muted")
        self.layout.addWidget(self.preflight_status)
        if preflight is not None:
            self.layout.addWidget(action(tr("action.preflight"), preflight, True))
        self.layout.addStretch()

        self._set_loaded_calibration(channel_value)

        if channel_test is not None:
            self.calibration_test_requested.connect(channel_test)
        if channel_test_stop is not None:
            self.calibration_test_stop_requested.connect(channel_test_stop)
        if save_calibration is not None:
            self.calibration_save_requested.connect(save_calibration)

    def _test_channel(self, channel_index: int) -> None:
        self._select_calibration_channel(channel_index)
        self.calibration_test_requested.emit(channel_index)

    def _toggle_channel_test(self) -> None:
        if getattr(self, "_channel_test_busy", False):
            self.calibration_test_stop_requested.emit()
            return
        index = int(self.calibration_channel_selector.currentData() or 0)
        self._test_channel(index)

    def _select_calibration_channel(self, channel_index: int) -> None:
        try:
            index = int(channel_index)
        except (TypeError, ValueError):
            return
        if not 0 <= index < self.calibration_channel_selector.count():
            return
        if self.calibration_channel_selector.currentIndex() != index:
            self.calibration_channel_selector.setCurrentIndex(index)
        else:
            self._selected_channel_changed(index)

    def _selected_channel_changed(self, index: int) -> None:
        index = int(index)
        self.head_map.set_selected_channel(index)
        position = ""
        if hasattr(self, "position_boxes") and index < len(self.position_boxes):
            position = self.position_boxes[index].currentText().strip()
        suffix = f" · {position}" if position else ""
        self.calibration_signal.set_channel(f"CH{index + 1}{suffix}")

    def _assign_map_position(self, position: str) -> None:
        index = int(self.calibration_channel_selector.currentData() or 0)
        if not 0 <= index < len(self.position_boxes):
            return
        self.position_boxes[index].setCurrentText(str(position))
        self._select_calibration_channel(index)
        self.calibration_status.setText(
            self.tr(
                "device.calibration_position_assigned",
                channel=f"CH{index + 1}",
                position=position,
            )
        )

    def _sync_head_map(self) -> None:
        if hasattr(self, "head_map"):
            self.head_map.set_assignments([box.currentText() for box in self.position_boxes])

    def append_channel_calibration_samples(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        try:
            channel = int(payload.get("channel", 0)) - 1
        except (TypeError, ValueError):
            return
        channels = payload.get("channels")
        if not 0 <= channel < 8 or not isinstance(channels, list) or channel >= len(channels):
            return
        values = channels[channel]
        if not isinstance(values, list):
            return
        self._select_calibration_channel(channel)
        self.calibration_signal.append_samples(values)
        self.calibration_signal_status.setText(
            self.tr(
                "device.calibration_signal_live",
                channel=f"CH{channel + 1}",
                samples=self.calibration_signal.sample_count,
                variation=f"{self.calibration_signal.variation:.2f} uV",
            )
        )

    @staticmethod
    def _editable_choice(values: tuple[str, ...], current: str, placeholder: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItem("", "")
        combo.addItems(values)
        combo.setCurrentText(current)
        if combo.lineEdit() is not None:
            combo.lineEdit().setPlaceholderText(placeholder)
        return combo

    @staticmethod
    def _protocol_status_key(status: str) -> str:
        normalized = str(status or "").strip().lower()
        if normalized.startswith("draft"):
            return "draft"
        if normalized in {"formal", "formal_candidate"}:
            return "formal_candidate"
        return normalized or "unknown"

    def _set_loaded_calibration(self, config: dict[str, Any]) -> None:
        path = str(config.get("calibration", {}).get("completed_at") or "") if isinstance(config.get("calibration"), dict) else ""
        if path:
            self.calibration_status.setText(self.tr("device.calibration_loaded", value=path))

    def _calibration_form_changed(self, *_args) -> None:
        self._sync_head_map()
        self._selected_channel_changed(self.calibration_channel_selector.currentIndex())

    def _save_calibration(self) -> None:
        positions = [box.currentText().strip() for box in self.position_boxes]
        if any(not is_confirmed_position(value) for value in positions):
            self.calibration_status.setText(self.tr("validation.calibration_missing"))
            return
        if len({value.casefold() for value in positions}) != len(positions):
            self.calibration_status.setText(self.tr("validation.calibration_duplicate"))
            return
        results = [result or {} for result in self._test_results]
        if any(str(result.get("status")) != "passed" for result in results):
            self.calibration_status.setText(self.tr("validation.calibration_test_required"))
            return
        if not all(
            is_confirmed_position(combo.currentText())
            for combo in (self.reference_position, self.bias_position, self.ground_position)
        ):
            self.calibration_status.setText(self.tr("validation.calibration_aux_missing"))
            return
        if not self.protocol_reviewed.isChecked():
            self.calibration_status.setText(self.tr("validation.calibration_protocol_ack"))
            return
        answer = QMessageBox.question(
            self,
            self.tr("device.save_formal_title"),
            self.tr("device.save_formal_prompt"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.calibration_save_requested.emit({
            "serial_port": self.calibration_port.text().strip() or "AUTO",
            "channels": [
                {
                    "name": f"CH{index + 1}",
                    "gui_index": index,
                    "board_input": f"N{index + 1}P",
                    "electrode_position": position,
                }
                for index, position in enumerate(positions)
            ],
            "reference": {
                "label": "SRB",
                "position": self.reference_position.currentText().strip(),
                "hardware_connection": "SRB ear clip",
            },
            "bias": {
                "label": "BIAS",
                "position": self.bias_position.currentText().strip(),
                "hardware_connection": "BIAS ear clip",
            },
            "ground": {
                "label": "AGND",
                "position": self.ground_position.currentText().strip(),
                "hardware_connection": "AGND on acquisition board",
            },
            "tests": results,
            "protocol_reviewed": True,
            "notes": self.tr("device.calibration_notes"),
        })

    def set_channel_test_busy(self, channel_index: int, busy: bool) -> None:
        self._channel_test_busy = bool(busy)
        if busy and 0 <= int(channel_index) < 8:
            self._select_calibration_channel(int(channel_index))
        self.calibration_signal.set_active(busy)
        self.calibration_channel_selector.setEnabled(not busy)
        self.head_map.setEnabled(not busy)
        self.calibration_test_toggle.setEnabled(True)
        self.calibration_test_toggle.setText(
            self.tr("device.calibration_stop_test") if busy
            else self.tr("device.calibration_start_test")
        )
        self.calibration_signal_status.setText(
            self.tr("device.calibration_signal_running") if busy
            else self.tr("device.calibration_signal_idle")
        )
        for button in self.test_buttons:
            button.setEnabled(not busy)
        self.save_calibration_button.setEnabled(not busy)

    def set_channel_test_stopping(self, stopping: bool) -> None:
        if stopping:
            self.calibration_test_toggle.setEnabled(False)
            self.calibration_test_toggle.setText(self.tr("device.calibration_stopping_test"))

    def set_channel_test_result(self, channel_index: int, result: dict[str, Any]) -> None:
        if not 0 <= channel_index < len(self.test_statuses):
            return
        self._test_results[channel_index] = result
        status = str(result.get("status") or "failed")
        status_text = self.tr("device.calibration_status." + status)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        detail = self.tr(
            "device.calibration_result_detail",
            status=status_text,
            finite=f"{float(metrics.get('finite_fraction', 0.0) or 0.0) * 100:.1f}%",
            flat=f"{float(metrics.get('flat_fraction', 0.0) or 0.0) * 100:.1f}%",
            saturation=f"{float(metrics.get('saturation_fraction', 0.0) or 0.0) * 100:.1f}%",
        )
        self.test_statuses[channel_index].setText(detail)
        self.calibration_signal.set_active(False)
        self.calibration_signal.set_status(detail)
        self.calibration_signal_status.setText(detail)
        selected_port = str(result.get("selected_port") or "").strip()
        if selected_port:
            self.calibration_port.setText(selected_port)
        self.calibration_status.setText(str(result.get("detail") or detail))

    def set_calibration_saved(self, result: dict[str, Any]) -> None:
        self.calibration_path.setText(
            self.tr("device.calibration_saved", path=str(result.get("channel_config_path") or ""))
        )
        self.calibration_status.setText(self.tr("device.calibration_ready"))

    def set_protocol_status(self, status: str) -> None:
        key = self._protocol_status_key(status)
        self.protocol_status.setText(
            self.tr("device.protocol_status", status=self.tr("device.protocol_status." + key))
        )

    def set_device_info(self, device: DeviceInfo) -> None:
        self.device_status.setText(_device_connection_text(self.tr, device))


class DiagnosticsPage(Page):
    """Developer-facing runtime checks and recent metadata-only events."""

    def __init__(
        self,
        tr,
        report_provider,
        events_provider,
        refresh_callback,
        export_callback,
        clear_callback,
    ):
        super().__init__(tr, tr("nav.diagnostics"), tr("diagnostics.subtitle"))
        controls = QHBoxLayout()
        controls.addWidget(action(tr("diagnostics.refresh"), refresh_callback, True))
        controls.addWidget(action(tr("diagnostics.export"), export_callback))
        controls.addWidget(action(tr("diagnostics.clear"), clear_callback))
        controls.addStretch()
        self.layout.addLayout(controls)

        self.status = label("", "muted")
        self.layout.addWidget(self.status)
        self.checks = _readonly_table((
            tr("diagnostics.check"),
            tr("diagnostics.status"),
            tr("diagnostics.detail"),
        ))
        self.checks.setObjectName("diagnosticsChecksTable")
        self.checks.setAccessibleName(tr("diagnostics.checks_table"))
        self.checks.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.checks.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.checks.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.layout.addWidget(self.checks)

        events = Section(tr("diagnostics.events"))
        self.events = QTextEdit()
        self.events.setObjectName("diagnosticsEvents")
        self.events.setReadOnly(True)
        self.events.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.events.setMinimumHeight(220)
        events.layout.addWidget(self.events)
        self.layout.addWidget(events)
        self.refresh(report_provider(), events_provider())

    def refresh(self, report: dict[str, Any], events: tuple[dict[str, Any], ...]) -> None:
        checks = report.get("checks", []) if isinstance(report, dict) else []
        self.checks.setRowCount(0)
        failures = 0
        warnings = 0
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status", "unknown"))
            failures += status == "failed"
            warnings += status in {"warning", "draft"}
            row = self.checks.rowCount()
            self.checks.insertRow(row)
            self.checks.setItem(row, 0, _table_item(self.tr("diagnostics.check." + str(check.get("name", "unknown")))))
            self.checks.setItem(row, 1, _table_item(self.tr("diagnostics.status." + status)))
            detail = str(check.get("detail") or "")
            if not detail and check.get("detail_key"):
                detail = self.tr(str(check["detail_key"]), **(check.get("values") or {}))
            self.checks.setItem(row, 2, _table_item(detail))
        self.checks.resizeRowsToContents()
        generated = str(report.get("generated_at") or "") if isinstance(report, dict) else ""
        self.status.setText(self.tr(
            "diagnostics.summary",
            failed=failures,
            warning=warnings,
            generated=generated,
        ))
        lines = []
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            context = event.get("context")
            context_text = ""
            if isinstance(context, dict) and context:
                context_text = " | " + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            lines.append(
                f"{event.get('timestamp', '')} [{event.get('level', 'INFO')}] "
                f"{event.get('source', '')}: {event.get('message', '')}{context_text}"
            )
        self.events.setPlainText("\n".join(lines) or self.tr("diagnostics.no_events"))


class AppsPage(Page):
    def __init__(self, tr, navigate, device_info: DeviceInfo | None = None):
        super().__init__(tr, tr("nav.apps"), tr("apps.subtitle"))
        grid = QGridLayout()
        grid.setSpacing(18)
        entries = [
            ("apps.ssvep", "apps.ssvep_note", QStyle.StandardPixmap.SP_ComputerIcon, lambda: navigate("ssvep")),
            ("apps.rest", "apps.rest_note", QStyle.StandardPixmap.SP_MediaPlay, None),
            ("apps.motor", "apps.soon", QStyle.StandardPixmap.SP_DialogApplyButton, None),
            ("apps.p300", "apps.soon", QStyle.StandardPixmap.SP_MediaVolume, None),
        ]
        for index, (title, subtitle, icon_type, callback) in enumerate(entries):
            grid.addWidget(AppTile(tr(title), tr(subtitle), icon_type, callback), index//2, index%2)
        self.layout.addLayout(grid)
        device_section = Section("OpenBCI Cyton · AUTO · 8 CH · 250 Hz")
        self.device_status = label(_device_connection_text(tr, device_info), "muted")
        device_section.layout.addWidget(self.device_status)
        self.layout.addWidget(device_section)
        self.layout.addWidget(label(tr("apps.flow"), "muted"))
        self.layout.addStretch()

    def set_device_info(self, device: DeviceInfo) -> None:
        self.device_status.setText(_device_connection_text(self.tr, device))


class SSVEPPage(Page):
    start_requested = Signal(object, float)
    serial_scan_requested = Signal()
    create_user_requested = Signal()
    refresh_users_requested = Signal()

    def __init__(self, tr, config: CaptureConfig, navigate, users=()):
        super().__init__(tr, "SSVEP", tr("ssvep.subtitle"))
        self.layout.addWidget(action(tr("action.back_apps"), lambda: navigate("apps")))
        self.layout.addWidget(label(tr("ssvep.workflow_steps"), "estimate"))
        self.layout.addWidget(label(tr("ssvep.description"), "muted"))
        self.layout.addWidget(label(tr("ssvep.defaults"), "muted"))
        self.layout.addWidget(label(tr("ssvep.safety"), "notice"))
        section = Section(tr("ssvep.parameters"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.participant = QLineEdit(config.participant)
        self.participant.setReadOnly(True)
        self.name = QLineEdit(config.name)
        self.user = QComboBox()
        self.user.setObjectName("captureUserSelector")
        self.user_picker = QWidget()
        user_picker_layout = QHBoxLayout(self.user_picker)
        user_picker_layout.setContentsMargins(0, 0, 0, 0)
        user_picker_layout.addWidget(self.user, 1)
        user_picker_layout.addWidget(action(tr("users.add_short"), self.create_user_requested.emit))
        user_picker_layout.addWidget(action(tr("users.refresh"), self.refresh_users_requested.emit))
        self.stimulus = self._spin(1, 30, config.stimulus_seconds, tr("field.seconds"))
        self.rest = self._spin(0, 30, config.rest_seconds, tr("field.seconds"))
        self.repetitions = self._spin(1, 10, config.repetitions)
        self.mode = QComboBox()
        for mode in (CaptureMode.CYTON,):
            self.mode.addItem(tr("mode." + mode.value), mode)
        self.mode.setCurrentIndex(max(0, self.mode.findData(config.mode)))
        self.port = QLineEdit(config.port)
        self.port.setPlaceholderText("AUTO")
        port_picker = QWidget()
        port_picker_layout = QHBoxLayout(port_picker)
        port_picker_layout.setContentsMargins(0, 0, 0, 0)
        port_picker_layout.addWidget(self.port, 1)
        port_picker_layout.addWidget(action(tr("action.scan_ports"), self.serial_scan_requested.emit))
        self.port_status = label(tr("field.port_auto"), "muted")
        self.eye_side = QComboBox()
        self.eye_side.addItem(tr("field.eye_select"), "")
        self.eye_side.addItem(tr("eye.left"), "left")
        self.eye_side.addItem(tr("eye.right"), "right")
        config_eye_side = getattr(config.eye_side, "value", config.eye_side)
        if str(config_eye_side) in {"left", "right"}:
            self.eye_side.setCurrentIndex(
                max(0, self.eye_side.findData(str(config_eye_side)))
            )
        self.screen_mapping = label("", "muted")
        self.dataset_preview = label("", "estimate")
        self.save = QLineEdit(str(config.save_directory))
        picker = QWidget()
        picker_layout = QHBoxLayout(picker)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.addWidget(self.save, 1)
        picker_layout.addWidget(action(tr("action.browse"), self._choose_directory))
        self.channel = QLineEdit(str(config.channel_config or ""))
        self.channel.setPlaceholderText(tr("field.channel_manual_placeholder"))
        self.channel_manual = QCheckBox(tr("field.channel_manual"))
        self.channel_manual.setChecked(config.channel_config is not None)
        self.channel_browse = None
        self.channel_picker = QWidget()
        channel_picker_layout = QHBoxLayout(self.channel_picker)
        channel_picker_layout.setContentsMargins(0, 0, 0, 0)
        channel_picker_layout.addWidget(self.channel, 1)
        self.channel_browse = action(tr("action.browse"), self._choose_channel_config)
        channel_picker_layout.addWidget(self.channel_browse)
        channel_controls = QWidget()
        channel_controls_layout = QVBoxLayout(channel_controls)
        channel_controls_layout.setContentsMargins(0, 0, 0, 0)
        channel_controls_layout.setSpacing(6)
        self.channel_auto_label = label(tr("field.channel_auto"), "muted")
        channel_controls_layout.addWidget(self.channel_auto_label)
        channel_controls_layout.addWidget(self.channel_picker)
        self.acknowledge = QCheckBox(tr("ssvep.acknowledge"))
        self.acknowledge.setChecked(config.acknowledge_flicker_risk)
        self.allow_draft = QCheckBox(tr("ssvep.allow_draft"))
        self.allow_draft.setChecked(config.allow_draft_hardware_config)
        form.addRow(label(tr("ssvep.step_identity"), "sectionTitle"), QWidget())
        for key, widget in (("field.user", self.user_picker), ("field.participant", self.participant), ("field.name", self.name),
                            ("field.mode", self.mode), ("field.port", port_picker),
                            ("field.eye_side", self.eye_side),
                            ("field.stimulus", self.stimulus), ("field.rest", self.rest),
                             ("field.repetitions", self.repetitions), ("field.refresh", label("60 Hz")),
                             ("field.save", picker),
                            ("field.channel_config", channel_controls)):
            form.addRow(tr(key), widget)
        form.addRow(label(tr("ssvep.step_device"), "sectionTitle"), QWidget())
        form.addRow("", self.channel_manual)
        form.addRow("", self.acknowledge)
        form.addRow("", self.allow_draft)
        section.layout.addWidget(label(tr("ssvep.step_protocol"), "sectionTitle"))
        section.layout.addLayout(form)
        section.layout.addWidget(self.port_status)
        section.layout.addWidget(self.screen_mapping)
        section.layout.addWidget(self.dataset_preview)
        section.layout.addWidget(label(tr("ssvep.refresh_note"), "muted"))
        self.estimate = label("", "estimate")
        section.layout.addWidget(self.estimate)
        section.layout.addWidget(label(tr("ssvep.formula"), "muted"))
        self.layout.addWidget(section)
        self.layout.addWidget(StaticTargets(tr))
        self.device_summary = KeyValues([(tr("device.channels"), "OpenBCI Cyton · AUTO · 8 CH / 250 Hz"),
            (tr("ssvep.frequencies"), "10 / 12 / 15 / 20 Hz"), ("Marker", tr("ssvep.marker_map"))])
        self.layout.addWidget(self.device_summary)
        self.error = label("", "error")
        self.layout.addWidget(self.error)
        self.layout.addWidget(label(tr("ssvep.countdown_note"), "muted"))
        self.start_button = action(tr("action.start_task"), self._start, True)
        self.start_button.setObjectName("startTaskButton")
        self.layout.addWidget(self.start_button)
        for spin in (self.stimulus, self.rest, self.repetitions):
            spin.valueChanged.connect(self._update_estimate)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.port.textChanged.connect(lambda _text: self._update_device_summary())
        self.eye_side.currentIndexChanged.connect(lambda _index: self._update_name_preview())
        self.name.textChanged.connect(lambda _text: self._update_name_preview())
        self.user.currentIndexChanged.connect(self._user_changed)
        self.channel_manual.toggled.connect(self._channel_manual_changed)
        self.set_users(users, selected_id=config.user_id)
        self._mode_changed()
        self._update_screen_mapping()
        self._update_name_preview()
        self._update_estimate()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, suffix="") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def config(self) -> CaptureConfig:
        mode = CaptureMode.CYTON
        profile = self.user.currentData()
        user_id = str(profile.user_id) if isinstance(profile, UserProfile) else ""
        user_name = str(profile.name) if isinstance(profile, UserProfile) else ""
        channel_config = (
            self.channel.text().strip()
            if mode == CaptureMode.CYTON and self.channel_manual.isChecked()
            else ""
        )
        return CaptureConfig(participant=user_id or self.participant.text().strip(), name=self.name.text().strip(),
            user_id=user_id, user_name=user_name,
            stimulus_seconds=self.stimulus.value(), rest_seconds=self.rest.value(),
            repetitions=self.repetitions.value(), save_directory=Path(self.save.text()).expanduser(),
            mode=CaptureMode(self.mode.currentData()), port=self.port.text().strip(),
            eye_side=str(self.eye_side.currentData() or ""),
            acknowledge_flicker_risk=self.acknowledge.isChecked(),
            channel_config=Path(channel_config).expanduser() if channel_config else None,
            allow_draft_hardware_config=self.allow_draft.isChecked())

    def set_users(self, users, selected_id: str = ""):
        if not selected_id:
            selected_id = self.config().user_id
        self.user.blockSignals(True)
        self.user.clear()
        for profile in users:
            if profile.status != "active" or profile.is_demo:
                continue
            self.user.addItem(profile.display_name, profile)
        # UserProfile objects are stored as item data, so QComboBox's
        # findData cannot compare them to a string ID reliably across Qt
        # bindings. Resolve the ID explicitly to keep refresh/new-user flows
        # deterministic.
        for index in range(self.user.count()):
            profile = self.user.itemData(index)
            if isinstance(profile, UserProfile) and profile.user_id == selected_id:
                self.user.setCurrentIndex(index)
                break
        self.user.blockSignals(False)
        self._user_changed()

    def _user_changed(self):
        profile = self.user.currentData()
        if isinstance(profile, UserProfile):
            self.participant.setText(profile.user_id)
        else:
            self.participant.clear()

    def _update_estimate(self):
        config = self.config()
        self.estimate.setText(self.tr("ssvep.estimate", total=format_duration(config.total_seconds),
            recording=format_duration(config.recording_seconds), trials=config.trials))

    @staticmethod
    def _ordered_screens():
        application = QApplication.instance()
        if application is None:
            return ()
        return tuple(
            sorted(
                enumerate(application.screens()),
                key=lambda item: (
                    int(item[1].geometry().x()),
                    int(item[1].geometry().y()),
                    str(item[1].name()),
                ),
            )
        )

    def _update_screen_mapping(self):
        screens = self._ordered_screens()
        if len(screens) < 2:
            self.screen_mapping.setText(self.tr("ssvep.screen_mapping_missing"))
            return
        if int(screens[0][1].geometry().x()) == int(screens[-1][1].geometry().x()):
            self.screen_mapping.setText(self.tr("ssvep.screen_mapping_horizontal_missing"))
            return
        left_index, left = screens[0]
        right_index, right = screens[-1]
        self.screen_mapping.setText(
            self.tr(
                "ssvep.screen_mapping",
                left=f"{left_index} · {left.name()}",
                right=f"{right_index} · {right.name()}",
            )
        )

    def _update_name_preview(self):
        base = self.name.text().strip()
        eye_side = str(self.eye_side.currentData() or "")
        if not base or not eye_side:
            self.dataset_preview.setText(self.tr("ssvep.dataset_name_pending"))
            return
        try:
            generated = dataset_name_for_eye(base, eye_side)
        except ValueError:
            self.dataset_preview.setText(self.tr("ssvep.dataset_name_pending"))
            return
        self.dataset_preview.setText(
            self.tr("ssvep.dataset_name_preview", name=generated)
        )

    def _choose_directory(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("field.save"), self.save.text())
        if path:
            self.save.setText(path)

    def _choose_channel_config(self):
        path, _selected = QFileDialog.getOpenFileName(
            self, self.tr("field.channel_config"), self.channel.text(), "JSON (*.json)"
        )
        if path:
            self.channel.setText(path)

    def set_detected_ports(self, ports: tuple[dict[str, str], ...] | list[dict[str, str]]) -> None:
        devices = [str(item.get("device", "")).strip() for item in ports if item.get("device")]
        if devices:
            current = self.port.text().strip()
            selected = current if current in devices else devices[0]
            self.port.setText(selected)
            self.port_status.setText(self.tr(
                "field.port_detected",
                ports=", ".join(devices),
                selected=selected,
            ))
        else:
            self.port.setText("AUTO")
            self.port_status.setText(self.tr("field.port_none"))
        self._update_device_summary()

    def _update_device_summary(self):
        if not hasattr(self, "device_summary"):
            return
        port = self.port.text().strip() or "AUTO"
        value = f"OpenBCI Cyton · {port} · 8 CH / 250 Hz"
        self.device_summary.values[0].setText(value)

    def _mode_changed(self):
        cyton = True
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(CaptureMode.CYTON)))
        self.mode.blockSignals(False)
        self.port.setEnabled(True)
        self.channel_manual.setEnabled(True)
        self.channel_auto_label.setVisible(True)
        manual_channels = self.channel_manual.isChecked()
        self.channel_picker.setVisible(manual_channels)
        # Keep this control enabled for keyboard/accessibility tooling while
        # the advanced path picker itself stays hidden in automatic mode.
        self.channel.setEnabled(cyton)
        self.channel.setReadOnly(not manual_channels)
        if self.channel_browse is not None:
            self.channel_browse.setEnabled(manual_channels)
        self.allow_draft.setEnabled(True)
        self.acknowledge.setEnabled(True)
        self._update_device_summary()

    def _channel_manual_changed(self, checked: bool):
        cyton = CaptureMode(self.mode.currentData()) == CaptureMode.CYTON
        manual_channels = cyton and checked
        self.channel_picker.setVisible(manual_channels)
        self.channel.setEnabled(cyton)
        self.channel.setReadOnly(not manual_channels)
        if self.channel_browse is not None:
            self.channel_browse.setEnabled(manual_channels)

    def _start(self):
        try:
            config = self.config()
            config.validate()
            screens = self._ordered_screens()
            if len(screens) < 2:
                raise ValueError("validation.screens")
            if int(screens[0][1].geometry().x()) == int(screens[-1][1].geometry().x()):
                raise ValueError("validation.screens_horizontal")
        except ValueError as error:
            self.error.setText(self.tr(str(error)))
            return
        self.error.clear()
        self.start_requested.emit(config, 1.0)


class TaskPage(Page):
    cancel_requested = Signal()

    def __init__(self, tr):
        super().__init__(tr, "SSVEP")
        self.cancel_button = action(tr("action.cancel"), self.cancel_requested.emit)
        self.layout.addWidget(self.cancel_button)
        self.quality_notice = label("", "notice")
        self.quality_notice.setVisible(False)
        self.layout.addWidget(self.quality_notice)
        self.countdown_section = Section(tr("task.prepare_note"))
        self.countdown = label("5", "countdown")
        self.countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown.setMinimumHeight(130)
        self.countdown_section.layout.addWidget(self.countdown)
        self.layout.addWidget(self.countdown_section)
        self.run_section = Section(tr("task.running"))
        self.trial_banner = label(tr("task.trial_idle"), "estimate")
        self.trial_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_section.layout.addWidget(self.trial_banner)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p%")
        self.run_section.layout.addWidget(self.progress)
        self.stats = KeyValues([(tr(key), "—") for key in (
            "task.elapsed", "task.remaining", "task.trial", "task.target", "task.phase", "task.events", "task.total_elapsed")])
        self.run_section.layout.addWidget(self.stats)
        self.capture_label = label("", "muted")
        self.run_section.layout.addWidget(self.capture_label)
        self.speed_label = label("", "muted")
        self.run_section.layout.addWidget(self.speed_label)
        self.layout.addWidget(self.run_section)
        waveform_section = Section(tr("task.waveform"))
        waveform_section.layout.addWidget(label(tr("task.waveform_note"), "muted"))
        self.waveform = WaveformWidget(
            tr("task.waveform"),
            tr("task.waveform_badge"),
            empty_text=tr("task.waveform_empty"),
        )
        waveform_section.layout.addWidget(self.waveform)
        self.layout.addWidget(waveform_section)
        self.layout.addWidget(StaticTargets(tr))
        self.layout.addStretch()

    def set_quality_warning(self, text: str = ""):
        self.quality_notice.setText(text)
        self.quality_notice.setVisible(bool(text))

    def update_snapshot(self, snapshot: TaskSnapshot, config: CaptureConfig):
        countdown = snapshot.phase == Phase.COUNTDOWN
        self.waveform.set_active(snapshot.active and config.mode is CaptureMode.CYTON)
        self.title_label.setText("SSVEP · "+self.tr("task.preparing" if countdown else "task.running"))
        self.countdown_section.setVisible(countdown)
        self.run_section.setVisible(not countdown)
        self.cancel_button.setText(self.tr("action.cancel" if countdown else "action.stop_task"))
        self.countdown.setText(str(snapshot.countdown))
        self.progress.setValue(round(snapshot.progress*10))
        phase_key = "task.rest" if snapshot.resting else "task.stimulus_visual"
        if snapshot.trial_count and snapshot.trial:
            self.trial_banner.setText(self.tr("task.trial_banner", trial=snapshot.trial, total=snapshot.trial_count, frequency=snapshot.frequency))
        else:
            self.trial_banner.setText(self.tr("task.trial_idle"))
        values = [format_duration(snapshot.elapsed), format_duration(math.ceil(snapshot.remaining)),
            f"{snapshot.trial} / {snapshot.trial_count}", f"{snapshot.target} / {snapshot.frequency} Hz",
            self.tr(phase_key),
            str(snapshot.event_count), format_duration(snapshot.demo_elapsed)]
        for node, value in zip(self.stats.values, values):
            node.setText(value)
        eye_label = (
            self.tr("eye." + str(getattr(config.eye_side, "value", config.eye_side)))
            if str(getattr(config.eye_side, "value", config.eye_side)) in {"left", "right"}
            else self.tr("dataset_summary.unlabeled")
        )
        screen_text = (
            " · ".join(
                item
                for item in (str(snapshot.screen_index), snapshot.screen_name)
                if item
            )
            if snapshot.screen_index is not None
            else self.tr("dataset_summary.unlabeled")
        )
        self.capture_label.setText(
            self.tr(
                "task.capture_eye",
                mode=self.tr("task.capture." + config.mode.value),
                eye=eye_label,
                screen=screen_text,
            )
        )
        self.speed_label.setText(self.tr("task.realtime", duration=format_duration(config.recording_seconds)))


class ResultPage(Page):
    def __init__(self, tr, result: Dataset, navigate):
        title_key = {
            "completed": "result.title",
            "aborted": "result.aborted_title",
            "error": "result.error_title",
        }.get(result.status, "result.title")
        imported = result.imported or result.source is CaptureMode.IMPORTED_OPENBCI
        legacy = result.simulated or result.source in {
            CaptureMode.DEMO,
            CaptureMode.VISUAL_PREVIEW,
            CaptureMode.SYNTHETIC,
        }
        status_key = (
            "result.legacy_readonly"
            if legacy
            else "result.imported_openbci_saved"
            if imported and result.status == "completed"
            else "result." + result.status + "_saved"
            if result.status != "completed"
            else "result.cyton_saved"
        )
        notice_key = (
            "result.legacy_notice"
            if legacy
            else "result.imported_notice"
            if imported
            else "result.recorded_notice"
        )
        path_key = "result.recorded_path"
        sample_key = "result.recorded_samples"
        values_key = "result.recorded_values"
        super().__init__(tr, tr(title_key), f"{result.name} · {tr(status_key)}")
        self.layout.addWidget(label(tr(notice_key), "notice"))
        self.layout.addWidget(KeyValues([
            (tr("field.participant"), result.participant),
            (tr("field.eye_side"), tr("eye." + result.eye_side) if result.eye_side in {"left", "right"} else tr("dataset_summary.unlabeled")),
            (tr("field.screen"), " · ".join(item for item in (str(result.screen_index), result.screen_name) if item) if result.screen_index is not None else tr("dataset_summary.unlabeled")),
            (tr("field.user_id"), result.user_id or tr("users.unlinked")),
            (tr("field.user_name"), result.user_name or tr("users.unlinked")),
            (tr("field.user_status"), tr("users.link_" + result.user_link_status)),
            (tr("result.recording"), format_duration(result.recording_seconds)),
            (tr("result.preparation"), format_duration(result.preparation_seconds)),
            (tr("task.total_elapsed"), format_duration(math.ceil(result.demo_seconds))),
            (tr("result.trials"), str(result.trials)),
            (tr("result.channels"), str(result.channel_count)),
            (tr(sample_key), f"{result.samples_per_channel:,}"),
            (tr(values_key), f"{result.sample_values:,}"),
            (tr("result.events"), str(result.event_count)),
            (tr("result.validation_mode"), tr("result.validation." + result.validation_mode)),
            (tr("result.protocol_status"), result.protocol_status or "—"),
            (tr("result.quality_status"), tr("result.quality." + result.quality_status)),
            (tr("result.timestamp_gaps"), str(result.timestamp_gap_count)),
            (tr("result.dropped_frames"), str(result.dropped_frame_count)),
            (tr("result.flat_channels"), str(result.flat_channel_count)),
        ]))
        denoising = _denoising_section(tr, result)
        if denoising is not None:
            self.layout.addWidget(denoising)
        raw_preview = _dataset_raw_preview_section(tr, result)
        if raw_preview is not None:
            self.layout.addWidget(raw_preview)
        path = Section(tr(path_key))
        path_value = str(result.path)
        path.layout.addWidget(label(path_value, "path"))
        self.layout.addWidget(path)
        if result.error:
            error_section = Section(tr("result.error_details"))
            error_section.layout.addWidget(label(result.error, "error"))
            self.layout.addWidget(error_section)
        self.layout.addWidget(action(tr("app.workstation")+" / "+tr("app.datasets"), lambda: navigate("datasets"), True))
        self.layout.addWidget(action(tr("action.configure"), lambda: navigate("ssvep")))
        self.layout.addStretch()


class DatasetSummaryPage(Page):
    def __init__(self, tr, result: Dataset, navigate):
        super().__init__(tr, tr("dataset_summary.title"), result.name)
        self._result_path = result.path
        self.layout.addWidget(label(tr("dataset_summary.readonly"), "notice"))
        source_name = (
            tr("dataset_summary.legacy")
            if result.simulated or result.source in {
                CaptureMode.DEMO,
                CaptureMode.VISUAL_PREVIEW,
                CaptureMode.SYNTHETIC,
            }
            else tr("dataset_summary.imported")
            if result.imported
            else tr("dataset_summary.acquired")
        )
        session_rows = [
            (tr("dataset_summary.session"), result.name),
            (tr("field.eye_side"), tr("eye." + result.eye_side) if result.eye_side in {"left", "right"} else tr("dataset_summary.unlabeled")),
            (tr("field.screen"), " · ".join(item for item in (str(result.screen_index), result.screen_name) if item) if result.screen_index is not None else tr("dataset_summary.unlabeled")),
            (tr("dataset_summary.source"), source_name),
            (tr("field.participant"), result.participant or tr("dataset_summary.unlabeled")),
            (tr("dataset_summary.markers"), tr("dataset_summary.no_markers") if result.imported else str(result.event_count)),
            (tr("result.channels"), str(result.channel_count)),
            (tr("dataset_summary.sampling_rate"), f"{result.sampling_rate_hz:g} Hz"),
            (tr("dataset_summary.duration"), format_duration(result.recording_seconds)),
            (tr("dataset_summary.samples_per_channel"), f"{result.samples_per_channel:,}"),
            (tr("dataset_summary.file_count"), str(len(_dataset_file_rows(result)))),
            (
                tr("dataset_summary.workstation_copy"),
                str(result.path),
            ),
            (tr("dataset_summary.original_source"), result.source_path or "—"),
        ]
        # Always show the association fields, including for legacy/imported
        # sessions.  An explicit “unlinked” value is safer than hiding the
        # absence of a user record or guessing an identity from participant_id.
        session_rows[3:3] = [
            (tr("field.user_id"), result.user_id or tr("users.unlinked")),
            (tr("field.user_name"), result.user_name or tr("users.unlinked")),
            (tr("field.user_status"), tr("users.link_" + result.user_link_status)),
        ]
        session_table = _readonly_table(
            (tr("dataset_summary.field"), tr("dataset_summary.value"))
        )
        session_table.setObjectName("datasetSessionTable")
        session_table.setAccessibleName(tr("dataset_summary.session_table"))
        session_table.setColumnWidth(0, 210)
        session_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        for key, value in session_rows:
            row = session_table.rowCount()
            session_table.insertRow(row)
            session_table.setItem(row, 0, _table_item(key))
            session_table.setItem(row, 1, _table_item(value))
        session_table.resizeRowsToContents()
        session_table.setSortingEnabled(True)
        session_table.setMinimumHeight(min(420, max(150, session_table.sizeHintForRow(0) * len(session_rows) + 44)))
        self.layout.addWidget(session_table)

        denoising = _denoising_section(tr, result)
        if denoising is not None:
            self.layout.addWidget(denoising)

        files = Section(tr("dataset_summary.files"))
        file_rows = _dataset_file_rows(result)
        file_table = _readonly_table(
            (
                tr("dataset_summary.file_name"),
                tr("dataset_summary.file_type"),
                tr("dataset_summary.file_size"),
                tr("dataset_summary.file_modified"),
            )
        )
        file_table.setObjectName("datasetFilesTable")
        file_table.setAccessibleName(tr("dataset_summary.file_table"))
        file_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in (1, 2, 3):
            file_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        for values in file_rows:
            row = file_table.rowCount()
            file_table.insertRow(row)
            for column, value in enumerate(values):
                item = _table_item(value)
                if column == 0:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(_dataset_file_path(result, values[0])),
                    )
                file_table.setItem(row, column, item)
        file_table.resizeRowsToContents()
        file_table.setSortingEnabled(True)
        file_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        def show_file_context_menu(position) -> None:
            item = file_table.itemAt(position)
            if item is None:
                return
            row = item.row()
            name_item = file_table.item(row, 0)
            if name_item is None:
                return
            raw_path = name_item.data(Qt.ItemDataRole.UserRole)
            file_path = Path(str(raw_path or ""))
            if not file_path.is_file():
                return
            file_table.selectRow(row)
            menu = QMenu(file_table)
            open_action = menu.addAction(tr("dataset_summary.open_in_explorer"))
            selected_action = menu.exec(
                file_table.viewport().mapToGlobal(position)
            )
            if selected_action is open_action:
                _open_file_manager_folder(file_path)

        file_table.customContextMenuRequested.connect(show_file_context_menu)
        file_table.setMinimumHeight(min(360, max(84, file_table.sizeHintForRow(0) * max(1, len(file_rows)) + 44)))
        files.layout.addWidget(file_table)
        self.layout.addWidget(files)

        raw_preview = _dataset_raw_preview_section(tr, result)
        if raw_preview is not None:
            self.layout.addWidget(raw_preview)
        self.layout.addWidget(action(tr("app.datasets"), lambda: navigate("datasets"), True))
        self.layout.addStretch()


class _DatasetCollectionPage(Page):
    filter_changed = Signal(str, str, str)

    def __init__(self, tr, title: str, subtitle: str, datasets: tuple[Dataset, ...],
                 latest: str | None, show_result, navigate, *, trashed: bool = False,
                 search_text: str = "", source_value: str = "", status_value: str = "",
                 status_text: str = "", callbacks: dict[str, Any] | None = None):
        super().__init__(tr, title, subtitle)
        self._callbacks = callbacks or {}
        self._navigate = navigate
        self._datasets = tuple(datasets)
        self._trashed = trashed
        self._latest = latest
        self._show_result = show_result
        controls = QHBoxLayout()
        self._build_header_controls(controls)
        controls.addStretch()
        self.layout.addLayout(controls)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr("datasets.search_placeholder"))
        self.search.setClearButtonEnabled(True)
        self.search.setText(search_text)
        self.search.textChanged.connect(self._apply_filters)
        filters.addWidget(self.search, 2)
        self.source_filter = QComboBox()
        self.source_filter.addItem(tr("datasets.filter_all_sources"), "")
        for value in ("cyton", "imported_openbci", "demo", "preview", "synthetic"):
            self.source_filter.addItem(tr("mode." + value), value)
        source_index = self.source_filter.findData(source_value)
        if source_index >= 0:
            self.source_filter.setCurrentIndex(source_index)
        self.source_filter.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.source_filter, 1)
        self.status_filter = QComboBox()
        self.status_filter.addItem(tr("datasets.filter_all_statuses"), "")
        for value in ("completed", "aborted", "error"):
            self.status_filter.addItem(tr("datasets.status." + value), value)
        status_index = self.status_filter.findData(status_value)
        if status_index >= 0:
            self.status_filter.setCurrentIndex(status_index)
        self.status_filter.currentIndexChanged.connect(self._apply_filters)
        filters.addWidget(self.status_filter, 1)
        self.layout.addLayout(filters)
        self.status = label(status_text, "muted")
        self.status.setVisible(bool(status_text))
        self.layout.addWidget(self.status)
        self._records_host = QWidget()
        self._records_layout = QVBoxLayout(self._records_host)
        self._records_layout.setContentsMargins(0, 0, 0, 0)
        self._records_layout.setSpacing(16)
        self.layout.addWidget(self._records_host)
        self._records_widgets: list[QWidget] = []
        self._render_records()
        self.layout.addStretch()

    def _build_header_controls(self, controls: QHBoxLayout):
        controls.addWidget(action(self.tr("nav.apps"), lambda: self._navigate("apps")))

    def _render_records(self):
        for widget in self._records_widgets:
            self._records_layout.removeWidget(widget)
            widget.deleteLater()
        self._records_widgets = []
        query = self.search.text().strip().casefold()
        source = str(self.source_filter.currentData() or "")
        status = str(self.status_filter.currentData() or "")
        visible = tuple(
            sorted(
                (
                    dataset
                    for dataset in self._datasets
                    if bool(dataset.deleted_at) == self._trashed
                    and self._matches(dataset, query, source, status)
                ),
                key=lambda dataset: (
                    _dataset_timestamp(
                        dataset.deleted_at if self._trashed else dataset.created_at
                    ),
                    dataset.id,
                ),
                reverse=True,
            )
        )
        if not visible:
            empty_key = "datasets.empty_trash" if self._trashed else "datasets.empty"
            empty = label(self.tr(empty_key), "muted")
            self._records_layout.addWidget(empty)
            self._records_widgets.append(empty)
        else:
            for dataset in visible:
                item = self._dataset_section(dataset, trashed=self._trashed)
                self._records_layout.addWidget(item)
                self._records_widgets.append(item)

    def _matches(self, dataset: Dataset, query: str, source: str, status: str) -> bool:
        haystack = " ".join(
            (
                dataset.id,
                dataset.name,
                dataset.participant,
                dataset.user_id,
                dataset.user_name,
                dataset.source_path,
                " ".join(dataset.files),
            )
        ).casefold()
        return (
            (not query or query in haystack)
            and (not source or dataset.source.value == source)
            and (not status or dataset.status == status)
        )

    def _dataset_section(self, dataset: Dataset, *, trashed: bool) -> Section:
        item = Section(dataset.name)
        item.setObjectName(
            "datasetTrashItem" if trashed else
            "latestDataset" if dataset.id == self._latest else "section"
        )
        if trashed:
            item.layout.addWidget(label(self.tr("datasets.deleted_at", value=dataset.deleted_at or "—"), "muted"))

        source_label = self.tr("datasets.source." + dataset.source.value)
        status_key = dataset.status if dataset.status in {"completed", "aborted", "error"} else "unknown"
        status_label = self.tr("datasets.status." + status_key)
        summary = QHBoxLayout()
        summary.addWidget(label(source_label, "muted"))
        summary.addWidget(label("·", "muted"))
        summary.addWidget(label(status_label, "muted"))
        summary.addStretch()
        item.layout.addLayout(summary)

        if dataset.simulated or dataset.source in {
            CaptureMode.DEMO,
            CaptureMode.VISUAL_PREVIEW,
            CaptureMode.SYNTHETIC,
        }:
            item.layout.addWidget(label(self.tr("result.legacy_readonly"), "muted"))

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(24)
        metrics.setVerticalSpacing(8)
        metric_rows = (
            (
                self.tr("dataset_summary.recorded_at"),
                _format_dataset_timestamp(dataset.created_at),
            ),
            (self.tr("dataset_summary.session_id"), dataset.id),
            (self.tr("field.participant"), dataset.participant or self.tr("dataset_summary.unlabeled")),
            (self.tr("dataset_summary.duration"), format_duration(dataset.recording_seconds)),
            (self.tr("dataset_summary.sampling_rate"), f"{dataset.sampling_rate_hz:g} Hz"),
            (self.tr("dataset_summary.channel_count"), f"{dataset.channel_count} CH"),
            (self.tr("dataset_summary.samples_per_channel"), f"{dataset.samples_per_channel:,}"),
            (self.tr("dataset_summary.file_count"), str(len(dataset.files)) if dataset.files else "—"),
            (
                self.tr("dataset_summary.markers"),
                self.tr("dataset_summary.no_markers")
                if dataset.imported
                else str(dataset.event_count),
            ),
        )
        for index, (key, value) in enumerate(metric_rows):
            row = index // 2
            column = (index % 2) * 2
            metrics.addWidget(label(key, "muted"), row, column)
            value_label = label(value)
            value_label.setToolTip(value)
            metrics.addWidget(value_label, row, column + 1)
        metrics.setColumnStretch(1, 1)
        metrics.setColumnStretch(3, 1)
        item.layout.addLayout(metrics)

        item.layout.addWidget(label(
            self.tr("datasets.user_details", user_id=dataset.user_id or self.tr("users.unlinked"),
               user_name=dataset.user_name or self.tr("users.unlinked"),
               status=self.tr("users.link_" + dataset.user_link_status)), "muted"))

        if dataset.source_path:
            source_path = label(
                self.tr("dataset_summary.original_source") + ": " + Path(dataset.source_path).name,
                "path",
            )
            source_path.setToolTip(dataset.source_path)
            item.layout.addWidget(source_path)
        workstation_path = label(
            self.tr("dataset_summary.workstation_copy") + ": " + dataset.path.name,
            "path",
        )
        workstation_path.setToolTip(str(dataset.path))
        item.layout.addWidget(workstation_path)
        controls = QHBoxLayout()
        if not trashed:
            controls.addWidget(action(self.tr("action.view_dataset"), lambda _checked=False, d=dataset: self._show_result_callback(d)))
            controls.addWidget(action(self.tr("datasets.delete"), lambda _checked=False, d=dataset: self._delete_dataset(d)))
        else:
            controls.addWidget(action(self.tr("datasets.restore"), lambda _checked=False, d=dataset: self._restore_dataset(d)))
            controls.addWidget(action(self.tr("datasets.purge"), lambda _checked=False, d=dataset: self._purge_dataset(d)))
        controls.addStretch()
        item.layout.addLayout(controls)
        return item

    def _run_callback(self, name: str, dataset_id: str):
        callback = self._callbacks.get(name)
        if callback is None:
            return
        callback(dataset_id)
        refresh = self._callbacks.get("refresh")
        if refresh is not None:
            refresh()

    def _delete_dataset(self, dataset: Dataset):
        answer = QMessageBox.question(
            self,
            self.tr("datasets.delete_title"),
            self.tr("datasets.delete_prompt", name=dataset.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._run_callback("delete", dataset.id)
        except (ValueError, RuntimeError) as error:
            self.status.setText(self.tr(str(error)))
            self.status.setVisible(True)

    def _restore_dataset(self, dataset: Dataset):
        try:
            self._run_callback("restore", dataset.id)
        except (ValueError, RuntimeError) as error:
            self.status.setText(self.tr(str(error)))
            self.status.setVisible(True)

    def _purge_dataset(self, dataset: Dataset):
        answer = QMessageBox.question(
            self,
            self.tr("datasets.purge_title"),
            self.tr("datasets.purge_prompt", name=dataset.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._run_callback("purge", dataset.id)
        except (ValueError, RuntimeError) as error:
            self.status.setText(self.tr(str(error)))
            self.status.setVisible(True)

    def _show_result_callback(self, dataset):
        self._show_result(dataset)

    def _apply_filters(self):
        self.filter_changed.emit(
            self.search.text(),
            str(self.source_filter.currentData() or ""),
            str(self.status_filter.currentData() or ""),
        )
        self._render_records()

    def _choose_import_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("datasets.import_openbci"), self.import_directory
        )
        if path:
            self.import_requested.emit(path)


class DatasetsPage(_DatasetCollectionPage):
    import_requested = Signal(object)

    def __init__(self, tr, datasets: tuple[Dataset, ...], latest: str | None, show_result, navigate,
                 import_busy: bool = False, import_status: str = "", import_directory: str = "",
                 search_text: str = "", source_value: str = "", status_value: str = "",
                 callbacks: dict[str, Any] | None = None):
        self._import_busy = import_busy
        self.import_directory = import_directory
        super().__init__(
            tr, tr("app.datasets"), tr("datasets.subtitle"), datasets, latest,
            show_result, navigate, search_text=search_text, source_value=source_value,
            status_value=status_value, status_text=import_status, callbacks=callbacks,
        )

    def _build_header_controls(self, controls: QHBoxLayout):
        super()._build_header_controls(controls)
        self.import_button = action(self.tr("datasets.import_openbci"), self._choose_import_directory, True)
        self.import_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.import_button.setEnabled(not self._import_busy)
        controls.addWidget(self.import_button)


class DatasetTrashPage(_DatasetCollectionPage):
    def __init__(self, tr, datasets: tuple[Dataset, ...], navigate,
                 search_text: str = "", source_value: str = "", status_value: str = "",
                 callbacks: dict[str, Any] | None = None):
        super().__init__(
            tr, tr("datasets.trash"), tr("datasets.trash_subtitle"), datasets, None,
            None, navigate, trashed=True, search_text=search_text, source_value=source_value,
            status_value=status_value, callbacks=callbacks,
        )
        self.setObjectName("datasetTrashPage")

    def _build_header_controls(self, controls: QHBoxLayout):
        controls.addWidget(action(self.tr("datasets.back_to_datasets"), lambda: self._navigate("datasets")))


class InfoPage(Page):
    def __init__(self, tr, kind: str, openbci_status=None, launch_openbci=None):
        if kind == "openbci":
            super().__init__(tr, tr("nav.openbci"), tr("openbci.subtitle"))
            if openbci_status is not None:
                self.layout.addWidget(KeyValues([
                    (tr("openbci.source"), tr("status.ready") if openbci_status.source_ready else tr("status.missing")),
                    (tr("openbci.chinese"), tr("status.ready") if openbci_status.overlay_ready else tr("status.missing")),
                    (tr("openbci.executable"), tr("status.ready") if openbci_status.executable_ready else tr("status.not_built")),
                    (tr("openbci.revision"), openbci_status.revision or "—"),
                ]))
            for key in ("openbci.notice", "openbci.tools"):
                section = Section()
                section.layout.addWidget(label(tr(key)))
                self.layout.addWidget(section)
            if launch_openbci is not None:
                launch = action(tr("openbci.launch"), launch_openbci, True)
                launch.setEnabled(bool(openbci_status and openbci_status.executable_ready))
                self.layout.addWidget(launch)
        else:
            super().__init__(tr, tr("nav.integrations"), tr("integration.subtitle"))
            for key in ("integration.brainflow", "integration.openbci", "integration.lsl", "integration.analysis"):
                section = Section()
                section.layout.addWidget(label(tr(key)))
                self.layout.addWidget(section)
        self.layout.addStretch()
