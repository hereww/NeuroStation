import math
from pathlib import Path
import tempfile
import unittest

from apps.workstation_ui.gateway import CaptureConfig, MockGateway, Phase, format_duration


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gateway = MockGateway(self.clock)
        self.config = CaptureConfig()

    def test_default_production_timing_and_counts(self):
        self.assertEqual(self.config.trials, 12)
        self.assertEqual(self.config.recording_seconds, 93)
        self.assertEqual(self.config.total_seconds, 98)
        self.assertEqual(self.config.expected_events, 27)
        self.assertEqual(format_duration(98), "00:01:38")

    def test_no_rest_after_final_stimulus(self):
        config = CaptureConfig(stimulus_seconds=5, rest_seconds=3, repetitions=1)
        self.assertEqual(config.recording_seconds, 29)
        self.assertEqual(config.total_seconds, 34)
        self.assertEqual(CaptureConfig(rest_seconds=0, repetitions=1).recording_seconds, 20)

    def test_countdown_is_five_real_seconds_even_at_high_speed(self):
        self.gateway.start_ssvep(self.config, 16)
        self.clock.now = 4.999
        snapshot = self.gateway.tick()
        self.assertEqual(snapshot.phase, Phase.COUNTDOWN)
        self.assertEqual(snapshot.countdown, 1)
        self.clock.now = 5
        snapshot = self.gateway.tick()
        self.assertEqual(snapshot.phase, Phase.RUNNING)
        self.assertEqual(snapshot.elapsed, 0)
        self.assertEqual(snapshot.event_count, 3)

    def test_delayed_countdown_processing_does_not_skip_recording(self):
        self.gateway.start_ssvep(self.config)
        self.clock.now = 60
        snapshot = self.gateway.tick()
        self.assertEqual(snapshot.phase, Phase.RUNNING)
        self.assertEqual(snapshot.elapsed, 0)

    def test_trial_rest_and_frequency_boundaries(self):
        self.gateway.start_ssvep(self.config, 1)
        self.clock.now = 5
        self.gateway.tick()
        self.clock.now = 10
        first_rest = self.gateway.tick()
        self.assertTrue(first_rest.resting)
        self.assertEqual(first_rest.trial, 1)
        self.assertEqual(first_rest.event_count, 4)
        self.clock.now = 13
        second_stimulus = self.gateway.tick()
        self.assertFalse(second_stimulus.resting)
        self.assertEqual(second_stimulus.target, 2)
        self.assertEqual(second_stimulus.frequency, 12)
        self.assertEqual(second_stimulus.event_count, 5)

    def test_accelerated_completion_is_once_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created"
            config = CaptureConfig(save_directory=target)
            self.gateway.start_ssvep(config, 8)
            self.clock.now = 5
            self.gateway.tick()
            self.clock.now = 16.625
            completed = self.gateway.tick()
            self.assertEqual(completed.phase, Phase.COMPLETED)
            self.assertEqual(completed.progress, 100)
            self.assertEqual(completed.result.samples_per_channel, 23250)
            self.assertEqual(completed.result.sample_values, 186000)
            self.assertEqual(completed.result.event_count, 27)
            self.assertTrue(completed.result.path.is_absolute())
            self.assertFalse(completed.result.persisted)
            self.assertFalse(target.exists())
            self.clock.now = 100
            self.gateway.tick()
            self.assertEqual(len(self.gateway.datasets), 1)

    def test_dataset_delete_restore_and_purge_use_the_mock_recycle_bin(self):
        self.gateway.start_ssvep(self.config, 8)
        self.clock.now = 5
        self.gateway.tick()
        self.clock.now = 16.625
        result = self.gateway.tick().result

        deleted = self.gateway.delete_dataset(result.id)
        self.assertEqual((), self.gateway.datasets)
        self.assertEqual((result.id,), tuple(item.id for item in self.gateway.trashed_datasets))
        self.assertTrue(deleted.deleted_at)

        restored = self.gateway.restore_dataset(result.id)
        self.assertEqual((result.id,), tuple(item.id for item in self.gateway.datasets))
        self.assertEqual("", restored.deleted_at)
        self.assertEqual((), self.gateway.trashed_datasets)

        self.gateway.delete_dataset(result.id)
        self.gateway.purge_dataset(result.id)
        self.assertEqual((), self.gateway.datasets)
        self.assertEqual((), self.gateway.trashed_datasets)

    def test_duplicate_start_and_manual_device_conflict_rejected(self):
        self.gateway.start_ssvep(self.config)
        with self.assertRaisesRegex(RuntimeError, "validation.busy"):
            self.gateway.start_ssvep(self.config)
        with self.assertRaisesRegex(RuntimeError, "validation.busy"):
            self.gateway.start_manual(self.config)

    def test_cancel_at_countdown_and_run_produces_no_dataset(self):
        for during_run in (False, True):
            self.gateway.start_ssvep(self.config)
            if during_run:
                self.clock.now += 5
                self.gateway.tick()
            self.gateway.cancel()
            self.clock.now += 1000
            self.assertEqual(self.gateway.tick().phase, Phase.CANCELLED)
            self.assertEqual(self.gateway.datasets, ())

    def test_manual_recording_and_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "正式脑电数据库"
            config = CaptureConfig(save_directory=target)
            self.gateway.start_manual(config)
            self.clock.now = 2
            self.assertEqual(self.gateway.add_marker(), 2)
            self.clock.now = 3
            result = self.gateway.stop_manual()
            self.assertEqual(result.samples_per_channel, 750)
            self.assertEqual(result.event_count, 3)
            self.assertEqual(result.preparation_seconds, 0)
            self.assertEqual(result.origin, "capture_test")
            self.assertFalse(result.persisted)
            self.assertFalse(target.exists())
            self.assertEqual(result.path, target.resolve() / result.id)
        with self.assertRaises(RuntimeError):
            self.gateway.add_marker()

    def test_invalid_configuration_rejected(self):
        for config in (CaptureConfig(participant=" "), CaptureConfig(name=""),
                       CaptureConfig(stimulus_seconds=0), CaptureConfig(rest_seconds=-1),
                       CaptureConfig(repetitions=11), CaptureConfig(repetitions=True),
                       CaptureConfig(save_directory=Path("relative")), CaptureConfig(refresh_rate=59)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        for speed in (0, 33, math.nan, math.inf):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                self.gateway.start_ssvep(self.config, speed)

    def test_returned_datasets_are_immutable_collection(self):
        self.assertIsInstance(self.gateway.datasets, tuple)

    def test_unbuilt_openbci_workspace_is_not_launched_by_mock(self):
        self.assertFalse(self.gateway.openbci_status.executable_ready)
        with self.assertRaisesRegex(RuntimeError, "validation.openbci_unavailable"):
            self.gateway.launch_openbci_workspace("zh-CN")


if __name__ == "__main__":
    unittest.main()
