from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eeg_tools.workstation.dataset import DatasetRepository
from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataGateway
from neurostation_contract import Phase, TaskSnapshot


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


class DatasetRenameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "Datasets"
        self.session = self.root / "session_original"
        self.session.mkdir(parents=True)
        self.metadata = {
            "session_id": "session_original", "session_name": "Original name",
            "source": "cyton", "duration_s": 1, "recorded_samples_per_channel": 1,
            "created_at": "2026-01-01T00:00:00", "eye_side": "left",
            "user_id": "U0001", "files": ["raw_brainflow.tsv", "events.tsv"],
        }
        (self.session / "session.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.session / "raw_brainflow.tsv").write_text("sample_index\ttimestamp_s\n0\t1789611232.6144743\n", encoding="utf-8")
        (self.session / "events.tsv").write_text("sample_index\tevent_name\n0\tstart\n", encoding="utf-8")
        (self.session / "manifest.csv").write_text("original manifest\n", encoding="utf-8")
        self.repository = DatasetRepository(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_rename_persists_without_changing_original_files_or_identity(self):
        before = {path.name: path.read_bytes() for path in self.session.iterdir()}
        original = self.repository.list_records()[0]

        renamed = self.repository.rename_record(original.session_id, "  新名称 左眼  ")
        reloaded = DatasetRepository(self.root).list_records()[0]

        self.assertEqual("新名称 左眼", renamed.session_name)
        self.assertEqual(renamed, reloaded)
        self.assertEqual(replace(original, session_name=renamed.session_name), renamed)
        self.assertEqual(before, {name: (self.session / name).read_bytes() for name in before})
        self.assertEqual(self.session, renamed.output_dir)
        self.assertEqual("Original name", json.loads((self.session / "session.json").read_text(encoding="utf-8"))["session_name"])
        self.assertFalse((self.session / "workstation_display.json.pending").exists())

    def test_unchanged_name_does_not_create_display_metadata(self):
        self.repository.rename_record("session_original", " Original name ")
        self.assertFalse((self.session / "workstation_display.json").exists())

    def test_invalid_name_and_missing_dataset_leave_files_unchanged(self):
        for name in ("", "  ", "name\nnext", "name\x00", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "validation.dataset_name"):
                self.repository.rename_record("session_original", name)
        with self.assertRaisesRegex(ValueError, "validation.dataset_not_found"):
            self.repository.rename_record("missing", "new name")
        self.assertFalse((self.session / "workstation_display.json").exists())
        self.assertEqual("Original name", self.repository.list_records()[0].session_name)

    def test_failed_atomic_write_keeps_previous_name(self):
        self.repository.rename_record("session_original", "First name")
        display_path = self.session / "workstation_display.json"
        before = display_path.read_bytes()
        with patch.object(Path, "replace", side_effect=PermissionError("read only")):
            with self.assertRaises(PermissionError):
                self.repository.rename_record("session_original", "Second name")
        self.assertEqual(before, display_path.read_bytes())
        self.assertEqual("First name", self.repository.list_records()[0].session_name)
        self.assertFalse((self.session / "workstation_display.json.pending").exists())

    def test_rename_survives_delete_and_restore(self):
        renamed = self.repository.rename_record("session_original", "Restored name")
        deleted = self.repository.delete_record(renamed.session_id)
        self.assertEqual(renamed.session_name, deleted.session_name)
        with self.assertRaisesRegex(ValueError, "validation.dataset_not_found"):
            self.repository.rename_record(renamed.session_id, "trash rename")
        restored = self.repository.restore_record(renamed.session_id)
        self.assertEqual(renamed.session_name, restored.session_name)
        self.assertEqual(renamed.output_dir, restored.output_dir)

    def test_bad_display_metadata_does_not_hide_dataset(self):
        path = self.session / "workstation_display.json"
        for contents in ("not json", "[]", '{"name": ""}', '{"name": 123}'):
            with self.subTest(contents=contents):
                path.write_text(contents, encoding="utf-8")
                self.assertEqual("Original name", self.repository.list_records()[0].session_name)

    def test_imported_rename_preserves_source_and_duplicate_detection(self):
        source = Path(self.temp.name) / "OpenBCISession_2026-01-02_00-00-00"
        source.mkdir()
        raw = source / "BrainFlow-RAW_0.csv"
        raw.write_text("\t".join(["0"] * 24) + "\n", encoding="utf-8")
        before = raw.read_bytes()
        self.repository.import_openbci_recordings(source)
        imported = next(record for record in self.repository.list_records() if record.imported)
        renamed = self.repository.rename_record(imported.session_id, "导入记录")
        self.assertEqual(imported.recorded_at, renamed.recorded_at)
        self.assertEqual(imported.fingerprint, renamed.fingerprint)
        self.assertEqual(before, raw.read_bytes())
        report = self.repository.import_openbci_recordings(source)
        self.assertEqual((0, 1, 0), (report.imported_count, report.skipped_count, report.failed_count))

    def test_legacy_import_retains_date_derived_from_original_name(self):
        self.metadata.update({
            "session_name": "OpenBCISession_2026-01-03_00-00-00",
            "source": "imported_openbci", "imported": True,
        })
        (self.session / "session.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        self.repository.rename_record("session_original", "Legacy display name")
        record = DatasetRepository(self.root).list_records()[0]
        self.assertEqual("2026-01-03T00:00:00", record.recorded_at)

    def test_desktop_gateway_refreshes_name_and_discards_stale_worker_copy(self):
        gateway = DesktopGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS, dataset_root=self.root)
        old = gateway.datasets[0]
        gateway._acquisition._datasets.append(old)
        result = gateway.rename_dataset(old.id, "New desktop name")
        gateway.refresh_datasets()
        self.assertEqual([result], list(gateway.datasets))
        self.assertEqual((), gateway._acquisition.datasets)
        restarted = MetadataGateway(protocol_path=PROTOCOL, dataset_root=self.root)
        self.assertEqual(result.name, restarted.datasets[0].name)

    def test_worker_only_dataset_can_be_renamed_before_metadata_refresh(self):
        gateway = DesktopGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS, dataset_root=self.root)
        old = gateway.datasets[0]
        gateway._metadata._datasets.clear()
        gateway._acquisition._datasets.append(old)
        self.assertEqual("Worker result", gateway.rename_dataset(old.id, "Worker result").name)
        self.assertEqual("Worker result", gateway.datasets[0].name)

    def test_transient_name_stays_in_memory_without_creating_dataset_files(self):
        gateway = DesktopGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS, dataset_root=self.root)
        transient = replace(gateway.datasets[0], id="transient", path=self.root / "missing", persisted=False)
        gateway._acquisition._datasets.append(transient)
        renamed = gateway.rename_dataset(transient.id, "Temporary display name")
        gateway.refresh_datasets()
        self.assertIn(renamed, gateway.datasets)
        self.assertFalse(transient.path.exists())

    def test_worker_dataset_in_custom_directory_keeps_renamed_cache(self):
        import shutil

        custom_root = Path(self.temp.name) / "Custom recordings"
        custom_session = custom_root / "session_custom"
        shutil.copytree(self.session, custom_session)
        metadata = {**self.metadata, "session_id": "session_custom"}
        (custom_session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
        custom = MetadataGateway(protocol_path=PROTOCOL, dataset_root=custom_root).datasets[0]
        gateway = DesktopGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS, dataset_root=self.root)
        gateway._acquisition._datasets.append(custom)
        gateway._acquisition._snapshot = TaskSnapshot(phase=Phase.COMPLETED, result=custom)

        renamed = gateway.rename_dataset(custom.id, "Custom name")
        gateway.refresh_datasets()

        self.assertIn(renamed, gateway.datasets)
        self.assertEqual(renamed, gateway._acquisition.snapshot.result)
        restarted = MetadataGateway(protocol_path=PROTOCOL, dataset_root=custom_root)
        self.assertEqual("Custom name", restarted.datasets[0].name)
        self.assertEqual("Original name", gateway._metadata.datasets[0].name)

    def test_acquisition_blocks_rename(self):
        gateway = DesktopGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS, dataset_root=self.root)
        gateway._metadata._snapshot = TaskSnapshot(phase=Phase.RUNNING)
        with self.assertRaisesRegex(RuntimeError, "validation.busy"):
            gateway.rename_dataset("session_original", "Busy rename")
        self.assertEqual("Original name", gateway.datasets[0].name)


if __name__ == "__main__":
    unittest.main()
