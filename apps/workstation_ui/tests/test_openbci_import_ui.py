from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None


def write_csv(path: Path, count: int = 3) -> None:
    rows = ["\t".join([str(index), *(["0"] * 23)]) for index in range(count)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@unittest.skipUnless(HAS_QT, "PySide6 not installed; pure logic tests still run")
class OpenBCIImportUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])
        cls.qt = Qt

    def test_import_button_and_readonly_summary(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import MetadataGateway

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            session = source / "OpenBCISession_2026-01-04_00-00-00"
            session.mkdir(parents=True)
            write_csv(session / "BrainFlow-RAW_0.csv")
            gateway = MetadataGateway(
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
            from PySide6.QtWidgets import QFrame, QPushButton

            buttons = window.pages["datasets"].findChildren(QPushButton)
            self.assertTrue(any(button.text() == "删除数据集" for button in buttons))
            self.assertIsNone(window.pages["datasets"].findChild(QFrame, "datasetsTrashSection"))
            self.assertIsNone(window.pages["datasets"].findChild(QFrame, "datasetTrashItem"))
            dataset = gateway.datasets[0]
            gateway.delete_dataset(dataset.id)
            window.refresh_datasets()
            window.navigate("trash")
            self.assertEqual("trash", window.current_page)
            self.assertEqual("数据集回收站", window.pages["trash"].title_label.text())
            self.assertIsNotNone(window.pages["trash"].findChild(QFrame, "datasetTrashItem"))
            window.navigate("datasets")
            self.assertIsNone(window.pages["datasets"].findChild(QFrame, "datasetTrashItem"))
            gateway.restore_dataset(dataset.id)
            window.refresh_datasets()
            window.navigate("datasets")
            derived = dataset.path / "derived" / "denoise_v1"
            derived.mkdir(parents=True)
            (derived / "denoising_metrics.json").write_text(json.dumps({
                "artifact_segment_count": 2,
                "artifact_fraction": 0.01,
                "tracks": {
                    "annotated": {"line_noise_ratio": {"50.0": 10}, "target_snr_db": {"top_right": 2}},
                    "50hz": {"line_noise_ratio": {"50.0": 0.1}, "target_snr_db": {"top_right": 4}},
                    "60hz": {"line_noise_ratio": {"60.0": 0.1}, "target_snr_db": {"top_right": 4}},
                },
                "analysis": {
                    "annotated": {"trial_count": 1, "rejected_trial_count": 1},
                    "50hz": {"trial_count": 1, "rejected_trial_count": 0},
                    "60hz": {"trial_count": 1, "rejected_trial_count": 0},
                },
            }), encoding="utf-8")
            (derived / "preprocessing.json").write_text(json.dumps({
                "pipeline_config": {
                    "bandpass_hz": [1.0, 45.0],
                    "bandpass_order": 4,
                    "line_noise_hz": [50.0, 60.0],
                    "notch_quality_factor": 30.0,
                    "max_interpolation_s": 0.2,
                },
                "pipeline_sha256": "pipeline-hash",
                "input_hashes": {
                    "raw_brainflow.tsv": "raw-hash",
                    "events.tsv": "events-hash",
                },
            }), encoding="utf-8")
            window.show_result(dataset)
            self.assertEqual("result", window.current_page)
            self.assertEqual("DatasetSummaryPage", type(window.pages["result"]).__name__)
            from PySide6.QtWidgets import QLabel, QPushButton, QComboBox, QTableWidget

            session_table = window.pages["result"].findChild(QTableWidget, "datasetSessionTable")
            file_table = window.pages["result"].findChild(QTableWidget, "datasetFilesTable")
            self.assertIsNotNone(session_table)
            self.assertIsNotNone(file_table)
            self.assertEqual(16, session_table.rowCount())
            denoising_table = window.pages["result"].findChild(QTableWidget, "denoisingMetricsTable")
            self.assertIsNotNone(denoising_table)
            self.assertEqual(3, denoising_table.rowCount())
            metadata_table = window.pages["result"].findChild(QTableWidget, "denoisingMetadataTable")
            self.assertIsNotNone(metadata_table)
            self.assertEqual(3, metadata_table.rowCount())
            metadata_texts = [
                metadata_table.item(row, column).text()
                for row in range(metadata_table.rowCount())
                for column in range(metadata_table.columnCount())
            ]
            self.assertIn("处理参数", metadata_texts)
            self.assertIn("输入 hash", metadata_texts)
            self.assertTrue(any("raw_brainflow.tsv: raw-hash" in value for value in metadata_texts))
            self.assertEqual(4, file_table.columnCount())
            self.assertEqual(1, file_table.rowCount())
            session_texts = [
                session_table.item(row, column).text()
                for row in range(session_table.rowCount())
                for column in range(session_table.columnCount())
            ]
            self.assertIn("工作站副本路径", session_texts)
            self.assertIn("未标注", session_texts)
            self.assertEqual("BrainFlow-RAW_0.csv", file_table.item(0, 0).text())
            raw_table = window.pages["result"].findChild(QTableWidget, "datasetRawPreviewTable")
            self.assertIsNotNone(raw_table)
            self.assertEqual(25, raw_table.columnCount())
            self.assertEqual(3, raw_table.rowCount())
            self.assertFalse(raw_table.item(0, 0).flags() & self.qt.ItemFlag.ItemIsEditable)
            self.assertEqual("1", raw_table.item(0, 0).text())
            raw_headers = [
                raw_table.horizontalHeaderItem(column).text()
                for column in range(raw_table.columnCount())
            ]
            self.assertEqual("序号", raw_headers[0])
            self.assertIn("package_num", raw_headers[1])
            self.assertTrue(any("other_ch1" in header for header in raw_headers))
            self.assertTrue(any("other_ch7" in header for header in raw_headers))
            self.assertTrue(any("analog_ch1" in header for header in raw_headers))
            self.assertTrue(any("analog_ch3" in header for header in raw_headers))
            preview_columns = window.pages["result"].findChild(
                QComboBox, "datasetPreviewColumns"
            )
            self.assertIsNotNone(preview_columns)
            self.assertEqual(
                ("all", "other", "analog"),
                tuple(preview_columns.itemData(index) for index in range(preview_columns.count())),
            )
            window.navigate("datasets")
            buttons = window.pages["datasets"].findChildren(QPushButton)
            open_button = next(button for button in buttons if button.text() == "打开数据集")
            open_button.click()
            self.assertEqual("DatasetSummaryPage", type(window.pages["result"]).__name__)
            window.close()

    def test_standard_brainflow_preview_headers_have_chinese_meanings(self):
        from apps.workstation_ui.i18n import Translator
        from apps.workstation_ui.pages import (
            _preview_column_indexes,
            _preview_header_labels,
            _sort_preview_rows_by_sample_index,
        )

        labels = _preview_header_labels(
            Translator("zh-CN"),
            (
                "sample_index",
                "eeg_ch1",
                "package_num",
                "timestamp_s",
                "marker",
                "accel_x",
                "other_ch1",
                "analog_ch1",
            ),
        )
        self.assertEqual("样本序号（sample_index）", labels[0])
        self.assertEqual("EEG 通道 1（eeg_ch1）", labels[1])
        self.assertEqual("设备时间戳（秒）（timestamp_s）", labels[3])
        self.assertEqual("事件标记码（marker）", labels[4])
        self.assertEqual("其他辅助通道 1（other_ch1）", labels[6])
        self.assertEqual("模拟辅助通道 1（analog_ch1）", labels[7])
        self.assertEqual(
            (0, 2, 3, 4, 6),
            _preview_column_indexes(
                (
                    "sample_index",
                    "eeg_ch1",
                    "package_num",
                    "timestamp_s",
                    "marker",
                    "accel_x",
                    "other_ch1",
                    "analog_ch1",
                ),
                "other",
            ),
        )
        self.assertEqual(
            (0, 2, 3, 4, 7),
            _preview_column_indexes(
                (
                    "sample_index",
                    "eeg_ch1",
                    "package_num",
                    "timestamp_s",
                    "marker",
                    "accel_x",
                    "other_ch1",
                    "analog_ch1",
                ),
                "analog",
            ),
        )
        self.assertEqual(
            [
                ["1", "101"],
                ["2", "202"],
                ["10", "303"],
                ["", "404"],
            ],
            _sort_preview_rows_by_sample_index(
                ("sample_index", "package_num"),
                [
                    ["10", "303"],
                    ["", "404"],
                    ["2", "202"],
                    ["1", "101"],
                ],
            ),
        )

    def test_dataset_list_sorts_by_original_openbci_recording_time(self):
        from PySide6.QtWidgets import QLabel

        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import MetadataGateway

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Recordings"
            older = source / "OpenBCISession_2026-01-04_08-00-00"
            newer = source / "OpenBCISession_2026-01-05_09-30-00"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            write_csv(older / "BrainFlow-RAW_0.csv")
            write_csv(newer / "BrainFlow-RAW_0.csv", count=4)
            gateway = MetadataGateway(
                protocol_path=Path("configs/protocols/ssvep_four_target_v2.json"),
                dataset_root=root / "Datasets",
            )
            gateway.import_openbci_recordings(source)
            window = MainWindow(gateway=gateway, timer_enabled=False)

            window.navigate("datasets")
            page = window.pages["datasets"]
            titles = [
                item.findChild(QLabel, "sectionTitle").text()
                for item in page._records_widgets
            ]
            text = [
                child.text()
                for item in page._records_widgets
                for child in item.findChildren(QLabel)
            ]

            self.assertEqual([newer.name, older.name], titles)
            self.assertIn("采集时间", text)
            self.assertIn("2026-01-05 09:30:00", text)
            self.assertTrue(any(value.startswith("原始来源路径: OpenBCISession_") for value in text))
            self.assertTrue(any(value.startswith("工作站副本路径: imported_openbci_") for value in text))
            window.close()


if __name__ == "__main__":
    unittest.main()
