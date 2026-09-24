from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neurostation_contract import (
    CaptureConfig,
    dataset_name_for_eye,
    normalize_dataset_name_base,
)
from eeg_tools.session_files import EVENT_FIELDS
from eeg_tools.workstation.acquisition_worker import (
    _create_stimulus_window,
    _write_frame_log,
    build_parser,
)
from neurostation_display import eye_half_geometry, resolve_eye_regions


class _Geometry:
    def __init__(self, x: int, y: int = 0, width: int = 1920, height: int = 1080) -> None:
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _Screen:
    def __init__(self, name: str, x: int, y: int = 0) -> None:
        self._name = name
        self._geometry = _Geometry(x, y)

    def name(self) -> str:
        return self._name

    def geometry(self) -> _Geometry:
        return self._geometry


class _Application:
    def __init__(self, screens: list[_Screen], primary: _Screen | None = None) -> None:
        self._screens = screens
        self._primary = primary or (screens[0] if screens else None)

    def screens(self) -> list[_Screen]:
        return self._screens

    def primaryScreen(self) -> _Screen | None:
        return self._primary


class SsvepEyeSideTests(unittest.TestCase):
    def test_dataset_name_has_one_eye_suffix_and_rejects_invalid_side(self) -> None:
        self.assertEqual("task", normalize_dataset_name_base("task_右眼_左眼"))
        self.assertEqual("task_左眼", dataset_name_for_eye("task", "left"))
        self.assertEqual("task_右眼", dataset_name_for_eye("task_左眼", "right"))
        self.assertEqual(
            "task_左眼",
            dataset_name_for_eye("task_右眼_左眼_右眼", "left"),
        )
        with self.assertRaisesRegex(ValueError, "validation.eye_side"):
            dataset_name_for_eye("task", "middle")

    def test_capture_config_requires_eye_side(self) -> None:
        config = CaptureConfig(
            participant="P001",
            name="task",
            acknowledge_flicker_risk=True,
        )
        with self.assertRaisesRegex(ValueError, "validation.eye_side"):
            config.validate()
        config = CaptureConfig(
            participant="P001",
            name="task_右眼",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        config.validate()
        self.assertEqual("task_左眼", config.dataset_name)

    def test_both_eye_regions_use_the_primary_screen(self) -> None:
        application = _Application([
            _Screen("middle", 0),
            _Screen("right", 1920),
            _Screen("left", -1920),
        ], primary=None)
        mapping = resolve_eye_regions(application)
        self.assertEqual("middle", mapping["left"]["screen"].name())
        self.assertEqual("middle", mapping["right"]["screen"].name())
        self.assertEqual(0, mapping["left"]["screen_index"])
        self.assertEqual(0, mapping["right"]["screen_index"])
        self.assertEqual({"x": 0, "y": 0, "width": 960, "height": 1080}, mapping["left"]["region_geometry"])
        self.assertEqual({"x": 960, "y": 0, "width": 960, "height": 1080}, mapping["right"]["region_geometry"])

    def test_single_screen_maps_both_regions(self) -> None:
        mapping = resolve_eye_regions(_Application([_Screen("only", -50)]))
        self.assertEqual(-50, mapping["left"]["region_geometry"]["x"])
        self.assertEqual(910, mapping["right"]["region_geometry"]["x"])

    def test_non_first_primary_screen_is_selected_for_both_eyes(self) -> None:
        screens = [_Screen("secondary", -1920), _Screen("primary", 0)]
        mapping = resolve_eye_regions(_Application(screens, primary=screens[1]))
        self.assertEqual(1, mapping["left"]["screen_index"])
        self.assertEqual(1, mapping["right"]["screen_index"])

    def test_no_screen_blocks_visual_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "validation.screens"):
            resolve_eye_regions(_Application([]))

    def test_odd_screen_width_keeps_all_pixels_and_rejects_unusable_screen(self) -> None:
        self.assertEqual({"x": 11, "y": 7, "width": 2, "height": 4}, eye_half_geometry(11, 7, 5, 4, "left"))
        self.assertEqual({"x": 13, "y": 7, "width": 3, "height": 4}, eye_half_geometry(11, 7, 5, 4, "right"))
        with self.assertRaisesRegex(RuntimeError, "validation.screens"):
            eye_half_geometry(0, 0, 1, 4, "left")
        with self.assertRaisesRegex(ValueError, "validation.eye_side"):
            eye_half_geometry(0, 0, 5, 4, "middle")

    def test_worker_requires_eye_side_and_frame_log_has_capture_metadata(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--output-root", ".",
                "--participant", "P001",
                "--session-name", "task",
            ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_timing.tsv"
            _write_frame_log(path, [{
                "trial_index": 0,
                "eye_side": "right",
                "dataset_name_base": "task",
                "dataset_name": "task_右眼",
                "screen_index": 2,
                "screen_name": "DISPLAY2",
                "screen_geometry": '{"x":1920,"y":0,"width":1920,"height":1080}',
                "screen_mapping": '{"left":null,"right":null}',
                "stimulus_region": '{"x":1920,"y":0,"width":960,"height":1080}',
                "frame_index": 0,
                "scheduled_s": 0.0,
                "actual_s": 0.0,
                "lateness_ms": 0.0,
                "dropped_since_previous": 0,
                "lit": 1,
            }])
            header = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
            self.assertIn("eye_side", header)
            self.assertIn("dataset_name_base", header)
            self.assertIn("dataset_name", header)
            self.assertIn("screen_name", header)
            self.assertIn("screen_geometry", header)
            self.assertIn("screen_mapping", header)
            self.assertIn("stimulus_region", header)
        self.assertIn("eye_side", EVENT_FIELDS)
        self.assertIn("dataset_name_base", EVENT_FIELDS)
        self.assertIn("dataset_name", EVENT_FIELDS)
        self.assertIn("screen_geometry", EVENT_FIELDS)
        self.assertIn("screen_mapping", EVENT_FIELDS)
        self.assertIn("stimulus_region", EVENT_FIELDS)

    def test_visual_stimulus_stays_inside_selected_half(self) -> None:
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        for side in ("left", "right"):
            _, window, mapping = _create_stimulus_window(side)
            try:
                self.assertEqual(mapping["left"]["screen_index"], mapping["right"]["screen_index"])
                window.show_target(0, True)
                image = window.grab().toImage()
                width, height = image.width(), image.height()
                active_x = width // 8 if side == "left" else width * 5 // 8
                inactive_x = width * 3 // 4 if side == "left" else width // 4
                self.assertEqual((0, 0, 0), image.pixelColor(inactive_x, height // 3).getRgb()[:3])
                self.assertEqual((255, 255, 255), image.pixelColor(active_x, height // 3).getRgb()[:3])
                window.show_message("5", flash=True)
                window.message_flash = True
                window.repaint()
                image = window.grab().toImage()
                self.assertEqual((0, 0, 0), image.pixelColor(inactive_x, height // 3).getRgb()[:3])
                self.assertEqual((255, 255, 255), image.pixelColor(active_x, height // 3).getRgb()[:3])
                QTest.keyClick(window, Qt.Key.Key_Q)
                self.assertTrue(window.abort_requested)
            finally:
                window.close()
                application.processEvents()


class CliEyeSideTests(unittest.TestCase):
    def test_compatibility_cli_validation_does_not_require_eye_side(self) -> None:
        import run_ssvep_session

        result = run_ssvep_session.main([
            "--validate-only",
            "--channel-config",
            "configs/channel_config_v1_auto.json",
        ])
        self.assertEqual(0, result)

    def test_compatibility_cli_forwards_eye_side(self) -> None:
        import run_ssvep_session

        with patch.object(run_ssvep_session, "worker_main", return_value=0) as worker:
            result = run_ssvep_session.main(["--eye-side", "right", "--headless"])
        self.assertEqual(0, result)
        forwarded = worker.call_args.args[0]
        eye_index = forwarded.index("--eye-side")
        self.assertEqual("right", forwarded[eye_index + 1])
        self.assertNotIn("--screen-index", forwarded)


if __name__ == "__main__":
    unittest.main()
