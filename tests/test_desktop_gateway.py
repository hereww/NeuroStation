from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase
from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataSimulationGateway
from neurostation_contract import UserProfile


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class MetadataSimulationGatewayTests(unittest.TestCase):
    def test_completion_creates_the_path_displayed_by_the_ui(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "中文数据集"
            gateway = MetadataSimulationGateway(
                protocol_path=PROTOCOL_PATH,
                dataset_root=root,
                clock=clock,
            )
            config = CaptureConfig(save_directory=root)
            self.assertEqual(Phase.COUNTDOWN, gateway.start_ssvep(config, 8).phase)
            clock.now = 4.999
            self.assertEqual(Phase.COUNTDOWN, gateway.tick().phase)
            clock.now = 5.0
            running = gateway.tick()
            self.assertEqual(Phase.RUNNING, running.phase)
            self.assertEqual(0, running.elapsed)
            clock.now = 16.625
            completed = gateway.tick()
            self.assertEqual(Phase.COMPLETED, completed.phase)
            self.assertTrue(completed.result.persisted)
            self.assertTrue(completed.result.path.is_dir())
            session = json.loads(
                (completed.result.path / "session.json").read_text(encoding="utf-8")
            )
            self.assertTrue(session["simulated"])
            self.assertEqual(0, session["recorded_samples_per_channel"])
            self.assertEqual(1, len(gateway.datasets))

    def test_runtime_rest_parameter_matches_ui_estimate(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = MetadataSimulationGateway(
                protocol_path=PROTOCOL_PATH, dataset_root=root, clock=clock
            )
            config = CaptureConfig(
                repetitions=1,
                stimulus_seconds=5,
                rest_seconds=2,
                save_directory=root,
            )
            gateway.start_ssvep(config, 1)
            clock.now = 5
            gateway.tick()
            clock.now = 31
            completed = gateway.tick()
            self.assertEqual(26, config.recording_seconds)
            self.assertEqual(Phase.COMPLETED, completed.phase)
            self.assertEqual(26, completed.result.recording_seconds)

    def test_metadata_cancel_returns_saved_aborted_result(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "中止数据"
            gateway = MetadataSimulationGateway(
                protocol_path=PROTOCOL_PATH, dataset_root=root, clock=clock
            )
            gateway.start_ssvep(CaptureConfig(save_directory=root), 8)
            clock.now = 5
            gateway.tick()
            cancelled = gateway.cancel()
            self.assertEqual(Phase.CANCELLED, cancelled.phase)
            self.assertIsNotNone(cancelled.result)
            self.assertEqual("aborted", cancelled.result.status)
            self.assertTrue(cancelled.result.path.is_dir())


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None


@unittest.skipUnless(HAS_QT, "PySide6 not installed")
class IntegratedDesktopQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_window_completion_navigates_to_a_persisted_result(self) -> None:
        from apps.workstation_ui.app import MainWindow

        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "脑电数据"
            gateway = MetadataSimulationGateway(
                protocol_path=PROTOCOL_PATH, dataset_root=root, clock=clock
            )
            window = MainWindow(gateway=gateway, timer_enabled=False)
            window.start_ssvep(CaptureConfig(save_directory=root), 8)
            clock.now = 5
            window.poll()
            clock.now = 16.625
            window.poll()
            self.application.processEvents()
            self.assertEqual("result", window.current_page)
            self.assertTrue(window.result.persisted)
            self.assertTrue(window.result.path.is_dir())
            window.navigate("datasets")
            self.assertEqual(window.latest_dataset, gateway.datasets[-1].id)
            window.close()

    def test_full_ui_completes_synthetic_worker_and_opens_dataset(self) -> None:
        from apps.workstation_ui.app import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
            protocol_value["countdown_s"] = 1
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(protocol_value, ensure_ascii=False), encoding="utf-8"
            )
            dataset_root = root / "中文合成板数据"
            gateway = DesktopGateway(
                protocol_path=protocol_path,
                channel_config_path=ROOT / "configs" / "channel_config_v1_template.json",
                dataset_root=dataset_root,
            )
            gateway.add_user(UserProfile(
                user_id="U0001", name="Synthetic user", age=30,
                gender="unspecified", medical_conditions=("none",),
            ))
            window = MainWindow(gateway=gateway, timer_enabled=False)
            config = CaptureConfig(
                participant="UI-P001",
                user_id="U0001",
                user_name="Synthetic user",
                name="UI synthetic acceptance",
                stimulus_seconds=1,
                rest_seconds=0,
                repetitions=1,
                save_directory=dataset_root,
                mode=CaptureMode.SYNTHETIC,
                acknowledge_flicker_risk=True,
            )
            window.start_ssvep(config, 1)
            self.assertEqual("task", window.current_page)
            deadline = time.monotonic() + 12
            while gateway.snapshot.active and time.monotonic() < deadline:
                window.poll()
                self.application.processEvents()
                time.sleep(0.05)
            window.poll()
            self.application.processEvents()
            self.assertEqual(Phase.COMPLETED, gateway.snapshot.phase, gateway.snapshot.error)
            self.assertEqual("result", window.current_page)
            self.assertIs(window.result.source, CaptureMode.SYNTHETIC)
            self.assertGreater(window.result.samples_per_channel, 0)
            self.assertGreater((window.result.path / "raw_brainflow.tsv").stat().st_size, 0)
            self.assertGreater((window.result.path / "frame_timing.tsv").stat().st_size, 84)
            window.navigate("datasets")
            self.assertEqual(window.latest_dataset, window.result.id)
            window.close()


if __name__ == "__main__":
    unittest.main()
