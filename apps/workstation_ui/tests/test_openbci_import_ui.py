from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None


def write_csv(path: Path) -> None:
    rows = ["\t".join([str(index), *(["0"] * 23)]) for index in range(3)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@unittest.skipUnless(HAS_QT, "PySide6 not installed; pure logic tests still run")
class OpenBCIImportUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_import_button_and_readonly_summary(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import MetadataSimulationGateway

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-04_00-00-00"
            session.mkdir(parents=True)
            write_csv(session / "BrainFlow-RAW_0.csv")
            gateway = MetadataSimulationGateway(
                protocol_path=Path("configs/protocols/ssvep_four_target_v2.json"),
                dataset_root=root / "Datasets",
            )
            window = MainWindow(gateway=gateway, timer_enabled=False)
            window.navigate("datasets")
            self.assertEqual("导入 OpenBCI 记录", window.pages["datasets"].import_button.text())

            window.pages["datasets"].import_requested.emit(str(source))
            while window.import_thread is not None:
                self.application.processEvents()
            window.navigate("datasets")
            self.assertEqual(1, len(gateway.datasets))
            self.assertTrue(gateway.datasets[0].imported)
            window.show_result(gateway.datasets[0])
            self.assertEqual("result", window.current_page)
            self.assertEqual("DatasetSummaryPage", type(window.pages["result"]).__name__)
            from PySide6.QtWidgets import QLabel, QPushButton

            texts = "\n".join(node.text() for node in window.pages["result"].findChildren(QLabel))
            self.assertIn("工作站副本路径", texts)
            window.navigate("datasets")
            buttons = window.pages["datasets"].findChildren(QPushButton)
            open_button = next(button for button in buttons if button.text() == "打开数据集")
            open_button.click()
            self.assertEqual("DatasetSummaryPage", type(window.pages["result"]).__name__)
            window.close()


if __name__ == "__main__":
    unittest.main()
