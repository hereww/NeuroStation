from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neurostation_contract import CaptureConfig, dataset_name_for_eye
from eeg_tools.session_files import EVENT_FIELDS
from eeg_tools.workstation.acquisition_worker import (
    _resolve_eye_screens,
    _write_frame_log,
    build_parser,
)


class _Geometry:
    def __init__(self, x: int, y: int = 0) -> None:
        self._x = x
        self._y = y

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y


class _Screen:
    def __init__(self, name: str, x: int, y: int = 0) -> None:
        self._name = name
        self._geometry = _Geometry(x, y)

    def name(self) -> str:
        return self._name

    def geometry(self) -> _Geometry:
        return self._geometry


class _Application:
    def __init__(self, screens: list[_Screen]) -> None:
        self._screens = screens

    def screens(self) -> list[_Screen]:
        return self._screens


class SsvepEyeSideTests(unittest.TestCase):
    def test_dataset_name_has_one_eye_suffix_and_rejects_invalid_side(self) -> None:
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

    def test_screens_are_mapped_by_physical_horizontal_position(self) -> None:
        application = _Application([
            _Screen("middle", 0),
            _Screen("right", 1920),
            _Screen("left", -1920),
        ])
        mapping = _resolve_eye_screens(application)
        self.assertEqual("left", mapping["left"]["screen"].name())
        self.assertEqual("right", mapping["right"]["screen"].name())
        self.assertEqual(2, mapping["left"]["screen_index"])
        self.assertEqual(1, mapping["right"]["screen_index"])

    def test_single_screen_blocks_visual_mapping(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "validation.screens"):
            _resolve_eye_screens(_Application([_Screen("only", 0)]))

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
        self.assertIn("eye_side", EVENT_FIELDS)
        self.assertIn("dataset_name_base", EVENT_FIELDS)
        self.assertIn("dataset_name", EVENT_FIELDS)
        self.assertIn("screen_geometry", EVENT_FIELDS)
        self.assertIn("screen_mapping", EVENT_FIELDS)


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
