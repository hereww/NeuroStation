"""Window coordination tests; never create a flickering stimulus or open hardware."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from neurostation_display import allow_stimulus_process_foreground, show_stimulus_fullscreen

ROOT = Path(__file__).resolve().parents[1]


class ForegroundPermissionTests(unittest.TestCase):
    def test_windows_grants_only_the_specific_acquisition_child(self):
        for granted in (True, False):
            user32 = Mock()
            user32.AllowSetForegroundWindow.return_value = granted
            with self.subTest(granted=granted), patch("sys.platform", "win32"), \
                 patch("ctypes.WinDLL", return_value=user32, create=True):
                self.assertEqual(granted, allow_stimulus_process_foreground(12345))
            user32.AllowSetForegroundWindow.assert_called_once_with(12345)

    def test_other_platforms_and_invalid_ids_do_not_call_native_api(self):
        with patch("sys.platform", "linux"), patch("ctypes.WinDLL", create=True) as native:
            self.assertIsNone(allow_stimulus_process_foreground(123))
        native.assert_not_called()
        for value in (None, 0, -1, True, "123", 0xFFFFFFFF):
            with self.subTest(value=value), patch("sys.platform", "win32"), \
                 patch("ctypes.WinDLL", create=True) as native:
                self.assertFalse(allow_stimulus_process_foreground(value))
            native.assert_not_called()

    def test_unavailable_native_api_is_not_a_capture_failure(self):
        with patch("sys.platform", "win32"), \
             patch("ctypes.WinDLL", side_effect=OSError("unavailable"), create=True):
            self.assertFalse(allow_stimulus_process_foreground(123))

    def test_gateway_grants_foreground_after_spawn_without_changing_capture_arguments(self):
        from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway
        from neurostation_contract import CaptureConfig, Phase
        gateway = AcquisitionProcessGateway(
            protocol_path=ROOT / "configs/protocols/ssvep_four_target_v2.json",
            channel_config_path=ROOT / "configs/channel_config_v1_auto.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            runtime.mkdir()
            config = CaptureConfig(participant="test", eye_side="left",
                                   save_directory=root / "datasets", acknowledge_flicker_risk=True)
            child = Mock(pid=12345)
            with patch("eeg_tools.workstation.process_gateway.tempfile.mkdtemp", return_value=str(runtime)), \
                 patch("eeg_tools.workstation.process_gateway.subprocess.Popen", return_value=child) as spawn, \
                 patch("eeg_tools.workstation.process_gateway.allow_stimulus_process_foreground", return_value=False) as grant, \
                 patch.object(gateway, "_start_learned_monitor"):
                snapshot = gateway.start_ssvep(config)
            grant.assert_called_once_with(12345)
            self.assertEqual(Phase.COUNTDOWN, snapshot.phase)
            self.assertNotIn("--headless", spawn.call_args.args[0])
            self.assertIn("--acknowledge-flicker-risk", spawn.call_args.args[0])
            self.assertNotIn("--allow-draft-protocol", spawn.call_args.args[0])


class StimulusWindowPlacementTests(unittest.TestCase):
    def test_screen_and_topmost_flags_are_set_before_show_and_activation(self):
        from PySide6.QtCore import Qt
        window, screen = Mock(), Mock()
        window.flags.return_value = Qt.WindowType.Window
        geometry = screen.geometry.return_value
        show_stimulus_fullscreen(window, screen)
        flags = window.setFlags.call_args.args[0]
        self.assertTrue(flags & Qt.WindowType.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowType.WindowStaysOnTopHint)
        self.assertEqual([
            call.flags(), call.setFlags(flags), call.setScreen(screen),
            call.setGeometry(geometry), call.showFullScreen(), call.raise_(), call.requestActivate(),
        ], window.mock_calls)


if __name__ == "__main__":
    unittest.main()
