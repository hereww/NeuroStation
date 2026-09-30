from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class DatasetFileActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtWidgets import QWidget
        from apps.workstation_ui import pages
        from apps.workstation_ui.i18n import Translator

        self.pages = pages
        self.tr = Translator("zh-CN")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = QWidget()

    def tearDown(self):
        self.parent.close()
        self.parent.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def test_finds_libreoffice_on_path(self):
        with patch.object(self.pages.shutil, "which", return_value="/usr/bin/libreoffice"), patch.object(
            self.pages.sys, "platform", "linux"
        ):
            self.assertEqual("/usr/bin/libreoffice", self.pages._find_libreoffice_executable())

    def test_finds_custom_windows_install_in_registry(self):
        executable = self.root / "Custom Office" / "soffice.exe"
        executable.parent.mkdir()
        executable.touch()
        registry = MagicMock()
        registry.QueryValueEx.return_value = (f'"{executable}"', 1)

        with patch.object(self.pages.shutil, "which", return_value=None), patch.object(
            self.pages.sys, "platform", "win32"
        ), patch.dict("sys.modules", {"winreg": registry}):
            self.assertEqual(str(executable), self.pages._find_libreoffice_executable())
        registry.QueryValueEx.assert_called_once()

    def test_finds_standard_windows_install_when_registry_missing(self):
        executable = self.root / "LibreOffice" / "program" / "soffice.exe"
        executable.parent.mkdir(parents=True)
        executable.touch()
        registry = MagicMock()
        registry.OpenKey.side_effect = FileNotFoundError

        with patch.object(self.pages.shutil, "which", return_value=None), patch.object(
            self.pages.sys, "platform", "win32"
        ), patch.dict("sys.modules", {"winreg": registry}), patch.dict(
            os.environ, {"ProgramFiles": str(self.root)}, clear=True
        ):
            self.assertEqual(str(executable), self.pages._find_libreoffice_executable())

    def test_returns_none_without_libreoffice(self):
        registry = MagicMock()
        registry.OpenKey.side_effect = FileNotFoundError
        with patch.object(self.pages.shutil, "which", return_value=None), patch.object(
            self.pages.sys, "platform", "win32"
        ), patch.dict("sys.modules", {"winreg": registry}), patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.pages._find_libreoffice_executable())

    def test_launch_passes_calc_and_complete_path_without_changing_file(self):
        path = self.root / "原始 数据.tsv"
        contents = "sample_index\ttimestamp_s\n0\t1789611232.6144743\n"
        path.write_text(contents, encoding="utf-8")
        executable = str(self.root / "LibreOffice" / "program" / "soffice.exe")

        with patch.object(self.pages, "_find_libreoffice_executable", return_value=executable), patch.object(
            self.pages.QProcess, "startDetached", return_value=(True, 123)
        ) as launch, patch.object(self.pages.QMessageBox, "warning") as warning:
            self.assertTrue(self.pages._open_in_libreoffice_calc(path, self.parent, self.tr))

        launch.assert_called_once_with(executable, ["--calc", str(path.resolve())])
        warning.assert_not_called()
        self.assertEqual(contents, path.read_text(encoding="utf-8"))

    def test_uninstalled_libreoffice_warns_without_launch(self):
        path = self.root / "data.csv"
        path.touch()
        with patch.object(self.pages, "_find_libreoffice_executable", return_value=None), patch.object(
            self.pages.QProcess, "startDetached"
        ) as launch, patch.object(self.pages.QMessageBox, "warning") as warning:
            self.assertFalse(self.pages._open_in_libreoffice_calc(path, self.parent, self.tr))
        launch.assert_not_called()
        warning.assert_called_once_with(
            self.parent, self.tr("error.title"), self.tr("dataset_summary.libreoffice_missing")
        )

    def test_launch_failure_warns(self):
        path = self.root / "data.csv"
        path.touch()
        for outcome in ((False, 0), OSError("launch failed")):
            with self.subTest(outcome=outcome):
                result = {"side_effect": outcome} if isinstance(outcome, OSError) else {"return_value": outcome}
                with patch.object(self.pages, "_find_libreoffice_executable", return_value="soffice.exe"), patch.object(
                    self.pages.QProcess, "startDetached", **result
                ), patch.object(self.pages.QMessageBox, "warning") as warning:
                    self.assertFalse(self.pages._open_in_libreoffice_calc(path, self.parent, self.tr))
                warning.assert_called_once_with(
                    self.parent, self.tr("error.title"),
                    self.tr("dataset_summary.libreoffice_failed", path=str(path)),
                )

    def test_removed_file_warns_without_launch(self):
        path = self.root / "missing.csv"
        with patch.object(self.pages, "_find_libreoffice_executable") as find, patch.object(
            self.pages.QProcess, "startDetached"
        ) as launch, patch.object(self.pages.QMessageBox, "warning") as warning:
            self.assertFalse(self.pages._open_in_libreoffice_calc(path, self.parent, self.tr))
        find.assert_not_called()
        launch.assert_not_called()
        warning.assert_called_once_with(
            self.parent, self.tr("error.title"), self.tr("dataset_summary.file_missing", path=str(path))
        )

    def test_context_menu_dispatches_selected_file_after_sorting(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtWidgets import QTableWidget
        from apps.workstation_ui.gateway import Dataset

        names = ("z data.csv", "a data.tsv", "session.json")
        for name in names:
            (self.root / name).write_text("{}" if name.endswith(".json") else "0\t1\n", encoding="utf-8")
        (self.root / "source_session.json").write_text("{}", encoding="utf-8")
        dataset = Dataset(
            id="test", name="Test dataset", participant="", protocol="ssvep",
            recording_seconds=1, preparation_seconds=0, demo_seconds=0,
            trials=0, samples_per_channel=1, event_count=0, path=self.root,
            created_at="2026-01-01T00:00:00", files=names, imported=True,
        )
        page = self.pages.DatasetSummaryPage(self.tr, dataset, lambda _: None)
        page.setParent(self.parent)
        table = page.findChild(QTableWidget, "datasetFilesTable")
        table.sortItems(0, Qt.SortOrder.AscendingOrder)

        for filename in ("a data.tsv", "session.json"):
            row = next(row for row in range(table.rowCount()) if table.item(row, 0).text() == filename)
            position = table.visualItemRect(table.item(row, 1)).center()
            expected_path = self.root / ("source_session.json" if filename == "session.json" else filename)
            for selection in (0, 1, None):
                with self.subTest(filename=filename, selection=selection):
                    actions = (object(), object())
                    menu = MagicMock()
                    menu.addAction.side_effect = actions
                    menu.exec.return_value = actions[selection] if selection is not None else None
                    with patch.object(self.pages, "QMenu", return_value=menu), patch.object(
                        self.pages, "_open_in_libreoffice_calc"
                    ) as calc, patch.object(self.pages, "_open_file_manager_folder") as explorer:
                        table.customContextMenuRequested.emit(position)
                    self.assertEqual(
                        [self.tr("dataset_summary.open_in_explorer"), self.tr("dataset_summary.open_in_libreoffice_calc")],
                        [call.args[0] for call in menu.addAction.call_args_list],
                    )
                    menu.exec.assert_called_once()
                    if selection == 1:
                        calc.assert_called_once_with(expected_path, page, self.tr)
                    else:
                        calc.assert_not_called()
                    if selection == 0:
                        explorer.assert_called_once_with(expected_path)
                    else:
                        explorer.assert_not_called()

        with patch.object(self.pages, "QMenu") as menu:
            table.customContextMenuRequested.emit(QPoint(-1, -1))
        menu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
