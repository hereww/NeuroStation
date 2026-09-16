from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_tools.workstation.dataset import DatasetRepository
from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataGateway


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"


def write_csv(path: Path, count: int) -> None:
    rows = ["\t".join([str(index), *(["0"] * 23)]) for index in range(count)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_txt(path: Path, count: int, channels: int = 8, rate: int = 250) -> None:
    rows = [
        "%OpenBCI Raw EXG Data",
        f"%Number of channels = {channels}",
        f"%Sample Rate = {rate} Hz",
        "Sample Index, EXG Channel 0",
    ]
    rows.extend(f"{index}, 0" for index in range(count))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class DatasetImportTests(unittest.TestCase):
    def test_csv_shards_are_streamed_and_copied_without_touching_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-01_00-00-00"
            session.mkdir(parents=True)
            first = session / "BrainFlow-RAW_2026-01-01_00-00-00_0.csv"
            second = session / "BrainFlow-RAW_2026-01-01_00-00-00_1.csv"
            write_csv(first, 3)
            write_csv(second, 2)
            before = {path.name: path.read_bytes() for path in session.iterdir()}

            repository = DatasetRepository(root / "Datasets")
            report = repository.import_openbci_recordings(source)
            record = repository.list_records()[0]

            self.assertEqual((1, 0, 0), (report.imported_count, report.skipped_count, report.failed_count))
            self.assertEqual(5, record.recorded_samples_per_channel)
            self.assertEqual(250, record.sampling_rate_hz)
            self.assertEqual(8, record.channel_count)
            self.assertEqual(0.02, record.duration_s)
            self.assertEqual({first.name, second.name}, set(record.files))
            self.assertEqual(before, {path.name: path.read_bytes() for path in session.iterdir()})
            self.assertTrue((record.output_dir / "session.json").is_file())

    def test_txt_metadata_and_duplicate_or_changed_source_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-02_00-00-00"
            session.mkdir(parents=True)
            raw = session / "OpenBCI-RAW-2026-01-02_00-00-00.txt"
            write_txt(raw, 4)
            repository = DatasetRepository(root / "Datasets")

            self.assertEqual(1, repository.import_openbci_recordings(source).imported_count)
            duplicate = repository.import_openbci_recordings(source)
            self.assertEqual((0, 1, 0), (duplicate.imported_count, duplicate.skipped_count, duplicate.failed_count))
            raw.write_text(raw.read_text(encoding="utf-8") + "4, 0\n", encoding="utf-8")
            changed = repository.import_openbci_recordings(source)

            self.assertEqual(1, changed.imported_count)
            records = repository.list_records()
            self.assertEqual(2, len(records))
            self.assertEqual({4, 5}, {record.recorded_samples_per_channel for record in records})
            self.assertTrue(any(record.output_dir.name.endswith("-2") for record in records))

    def test_import_skips_a_source_that_is_already_in_trash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-03_00-00-00"
            session.mkdir(parents=True)
            write_csv(session / "BrainFlow-RAW_2026-01-03_00-00-00_0.csv", 3)
            repository = DatasetRepository(root / "Datasets")

            self.assertEqual(1, repository.import_openbci_recordings(source).imported_count)
            dataset_id = repository.list_records()[0].session_id
            repository.delete_record(dataset_id)

            report = repository.import_openbci_recordings(source)

            self.assertEqual((0, 1, 0), (report.imported_count, report.skipped_count, report.failed_count))
            self.assertEqual([], repository.list_records())
            self.assertEqual([dataset_id], [record.session_id for record in repository.list_trashed_records()])

    def test_empty_and_unknown_sessions_fail_without_half_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            empty = source / "OpenBCISession_empty"
            unknown = source / "OpenBCISession_unknown"
            empty.mkdir(parents=True)
            unknown.mkdir(parents=True)
            (empty / "BrainFlow-RAW_empty.csv").write_text("", encoding="utf-8")
            (unknown / "notes.txt").write_text("not EEG", encoding="utf-8")

            repository = DatasetRepository(root / "Datasets")
            report = repository.import_openbci_recordings(source)

            self.assertEqual(2, report.failed_count)
            self.assertEqual([], repository.list_records())
            self.assertEqual([], list((root / "Datasets" / "imports").glob("*/session.json")))

    def test_old_workstation_session_shape_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session_20260101_000000_000"
            session.mkdir()
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": session.name,
                        "status": "completed",
                        "participant_id": "P001",
                        "session_name": "Legacy",
                        "recording_duration_s": 2,
                        "expected_duration_s": 2,
                        "completed_trials": 1,
                        "recorded_samples_per_channel": 500,
                        "event_count": 4,
                        "simulated": False,
                        "board": "cyton",
                        "sampling_rate_hz": 250,
                    }
                ),
                encoding="utf-8",
            )

            record = DatasetRepository(root).list_records()[0]
            self.assertEqual("Legacy", record.session_name)
            self.assertEqual("cyton", record.source)
            self.assertEqual(500, record.recorded_samples_per_channel)
            self.assertEqual(250, record.sampling_rate_hz)

    def test_dataset_delete_restore_and_purge_manage_the_whole_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            session = root / "session_delete"
            session.mkdir(parents=True)
            (session / "raw.tsv").write_text("sample\n", encoding="utf-8")
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": "session_delete",
                        "session_name": "Delete me",
                        "duration_s": 1,
                        "recorded_samples_per_channel": 250,
                        "sampling_rate_hz": 250,
                        "simulated": False,
                    }
                ),
                encoding="utf-8",
            )
            repository = DatasetRepository(root)

            deleted = repository.delete_record("session_delete")
            trash_path = root / "Trash" / "Datasets" / "session_delete"
            self.assertFalse(session.exists())
            self.assertTrue(trash_path.is_dir())
            self.assertEqual(trash_path, deleted.output_dir)
            self.assertEqual([], repository.list_records())
            self.assertEqual(["session_delete"], [item.session_id for item in repository.list_trashed_records()])
            metadata = json.loads((trash_path / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(str(session.resolve()), metadata["trash_original_path"])
            self.assertTrue(metadata["deleted_at"])

            restored = repository.restore_record("session_delete")
            self.assertTrue(session.is_dir())
            self.assertEqual(session, restored.output_dir)
            self.assertEqual([], repository.list_trashed_records())

            repository.delete_record("session_delete")
            repository.purge_record("session_delete")
            self.assertFalse(session.exists())
            self.assertFalse(trash_path.exists())

    def test_active_records_do_not_shadow_records_already_in_trash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Datasets"
            active = root / "imports" / "imported_openbci_same_session"
            trashed = root / "Trash" / "Datasets" / "imported_openbci_same_session"
            for path, deleted_at in ((active, ""), (trashed, "2026-09-16T10:00:00+08:00")):
                path.mkdir(parents=True)
                (path / "session.json").write_text(
                    json.dumps({
                        "session_id": "imported_openbci_same_session",
                        "session_name": "Same session",
                        "duration_s": 1,
                        "recorded_samples_per_channel": 250,
                        "sampling_rate_hz": 250,
                        "deleted_at": deleted_at,
                    }),
                    encoding="utf-8",
                )

            repository = DatasetRepository(root)

            self.assertEqual([], repository.list_records())
            self.assertEqual(
                ["imported_openbci_same_session"],
                [record.session_id for record in repository.list_trashed_records()],
            )

    def test_unknown_legacy_source_remains_visible_in_desktop_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session_20260101_000001_000"
            session.mkdir()
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "session_id": session.name,
                        "status": "completed",
                        "session_name": "Legacy unknown source",
                        "duration_s": 1,
                        "recorded_samples_per_channel": 250,
                        "source": "legacy_board_label",
                        "simulated": False,
                    }
                ),
                encoding="utf-8",
            )

            gateway = MetadataGateway(
                protocol_path=PROTOCOL,
                dataset_root=root,
            )

            self.assertEqual(1, len(gateway.datasets))
            self.assertEqual("Legacy unknown source", gateway.datasets[0].name)
            self.assertEqual("cyton", gateway.datasets[0].source.value)

    def test_desktop_gateway_refreshes_imported_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-03_00-00-00"
            session.mkdir(parents=True)
            write_csv(session / "BrainFlow-RAW_0.csv", 2)
            gateway = MetadataGateway(protocol_path=PROTOCOL, dataset_root=root / "Datasets")

            report = gateway.import_openbci_recordings(source)

            self.assertEqual(1, report.imported_count)
            self.assertEqual(1, len(gateway.datasets))
            self.assertTrue(gateway.datasets[0].imported)
            self.assertEqual("imported_openbci", gateway.datasets[0].source.value)

    def test_desktop_gateway_exposes_default_import_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = DesktopGateway(
                protocol_path=PROTOCOL,
                channel_config_path=ROOT / "configs" / "channel_config_v1_auto.json",
                dataset_root=root / "Datasets",
            )

            self.assertFalse(gateway.auto_import_default)
            with patch.object(DatasetRepository, "default_root", return_value=root / "DefaultDatasets"):
                default_gateway = DesktopGateway(
                    protocol_path=PROTOCOL,
                    channel_config_path=ROOT / "configs" / "channel_config_v1_auto.json",
                )
            self.assertTrue(default_gateway.auto_import_default)


if __name__ == "__main__":
    unittest.main()
