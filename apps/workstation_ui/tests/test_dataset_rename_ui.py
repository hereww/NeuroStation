from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class DatasetRenameUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import MetadataGateway

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "Datasets"
        self.session = self.root / "session_test"
        self.session.mkdir(parents=True)
        (self.session / "session.json").write_text(json.dumps({
            "session_id": "session_test", "session_name": "Original name",
            "source": "cyton", "created_at": "2026-01-01T00:00:00",
            "duration_s": 1, "recorded_samples_per_channel": 1,
            "files": ["raw_brainflow.tsv"],
        }), encoding="utf-8")
        (self.session / "raw_brainflow.tsv").write_text("sample_index\tpackage_num\n0\t1\n", encoding="utf-8")
        self.gateway = MetadataGateway(protocol_path=PROTOCOL, dataset_root=self.root)
        self.window = MainWindow(gateway=self.gateway, timer_enabled=False)
        self.window.navigate("datasets")

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def rename_button(self):
        from PySide6.QtWidgets import QPushButton

        return self.window.pages["datasets"].findChild(QPushButton, "datasetRenameButton")

    def test_button_saves_name_and_updates_list_details_and_restart(self):
        from PySide6.QtWidgets import QLabel
        from eeg_tools.workstation.desktop_gateway import MetadataGateway

        self.window.show_dataset(self.gateway.datasets[0])
        self.window.navigate("datasets")
        button = self.rename_button()
        self.assertTrue(button.isEnabled())
        self.assertEqual("修改名称", button.text())
        with patch("apps.workstation_ui.pages.QInputDialog.getText", return_value=("  新数据集名称  ", True)) as dialog:
            button.click()
        self.application.processEvents()
        self.assertEqual("Original name", dialog.call_args.args[-1])
        self.assertEqual("新数据集名称", self.gateway.datasets[0].name)
        self.assertEqual("新数据集名称", self.window.result.name)
        labels = self.window.pages["datasets"].findChildren(QLabel, "sectionTitle")
        self.assertIn("新数据集名称", [label.text() for label in labels])
        self.window.navigate("result")
        self.assertIn("新数据集名称", [label.text() for label in self.window.pages["result"].findChildren(QLabel)])
        restarted = MetadataGateway(protocol_path=PROTOCOL, dataset_root=self.root)
        self.assertEqual("新数据集名称", restarted.datasets[0].name)

    def test_cancel_or_unchanged_name_does_not_write_or_refresh(self):
        for response in (("Ignored", False), (" Original name ", True)):
            with self.subTest(response=response), patch(
                "apps.workstation_ui.pages.QInputDialog.getText", return_value=response
            ), patch.object(self.gateway, "rename_dataset") as rename, patch.object(self.window, "refresh_datasets") as refresh:
                self.rename_button().click()
            rename.assert_not_called()
            refresh.assert_not_called()
        self.assertFalse((self.session / "workstation_display.json").exists())

    def test_empty_name_displays_error_and_retains_old_name(self):
        with patch("apps.workstation_ui.pages.QInputDialog.getText", return_value=("  ", True)):
            self.rename_button().click()
        page = self.window.pages["datasets"]
        self.assertFalse(page.status.isHidden())
        self.assertEqual(self.window.tr("validation.dataset_name"), page.status.text())
        self.assertEqual("Original name", self.gateway.datasets[0].name)
        self.assertFalse((self.session / "workstation_display.json").exists())

    def test_write_failure_displays_error_and_retains_old_name(self):
        with patch("apps.workstation_ui.pages.QInputDialog.getText", return_value=("New name", True)), patch.object(
            self.gateway, "rename_dataset", side_effect=PermissionError("read only")
        ):
            self.rename_button().click()
        page = self.window.pages["datasets"]
        self.assertFalse(page.status.isHidden())
        self.assertEqual(self.window.tr("datasets.rename_failed", reason="read only"), page.status.text())
        self.assertEqual("Original name", page._datasets[0].name)

    def test_rename_keeps_filters_and_updates_search_matches(self):
        from PySide6.QtWidgets import QPushButton

        page = self.window.pages["datasets"]
        page.search.setText("Original")
        page.source_filter.setCurrentIndex(page.source_filter.findData("cyton"))
        page.status_filter.setCurrentIndex(page.status_filter.findData("completed"))
        self.application.processEvents()
        with patch("apps.workstation_ui.pages.QInputDialog.getText", return_value=("Renamed dataset", True)):
            self.rename_button().click()
        self.application.processEvents()
        page = self.window.pages["datasets"]
        self.assertEqual("Original", page.search.text())
        self.assertEqual("cyton", page.source_filter.currentData())
        self.assertEqual("completed", page.status_filter.currentData())
        self.assertIsNone(page.findChild(QPushButton, "datasetRenameButton"))
        page.search.setText("Renamed")
        self.application.processEvents()
        self.assertIsNotNone(page.findChild(QPushButton, "datasetRenameButton"))

    def test_display_metadata_is_not_listed_as_original_data(self):
        from dataclasses import replace
        from apps.workstation_ui.pages import _dataset_file_rows

        self.gateway.rename_dataset("session_test", "New name")
        result = replace(self.gateway.datasets[0], files=())
        self.assertEqual(["raw_brainflow.tsv"], [row[0] for row in _dataset_file_rows(result)])

    def test_trash_does_not_offer_name_edit(self):
        from PySide6.QtWidgets import QPushButton

        self.gateway.delete_dataset("session_test")
        self.window.navigate("trash")
        self.assertIsNone(self.window.pages["trash"].findChild(QPushButton, "datasetRenameButton"))


if __name__ == "__main__":
    unittest.main()
