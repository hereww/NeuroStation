"""Pure, deterministic regression tests; no board, GUI, or model is started."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eeg_tools.realtime_commands import (
    CHANNEL_NAMES, CommandJournal, CommandWindowBuffer, PendingPredictionGate,
)


def batch(start=0, count=250, *, packet_offset=0, native=False):
    message = {
        "samples": [[float((i + start) % 101 + channel) for i in range(count)]
                    for channel in range(8)],
        "channel_names": list(CHANNEL_NAMES), "sample_rate_hz": 250,
        "sample_indices": list(range(start, start + count)),
        "package_numbers": [(start + i + packet_offset) % 256 for i in range(count)],
        "timestamp_source": "host_burst",
    }
    message["timestamps_s" if native else "timestamps"] = [100.0] * count
    message["received_monotonic_s" if native else "host_timestamp"] = 10.0
    return message


class CommandWindowBufferTests(unittest.TestCase):
    def test_five_seconds_then_one_second_stride(self):
        buffer = CommandWindowBuffer()
        for start in range(0, 1000, 250):
            self.assertIsNone(buffer.add_batch(batch(start)))
        window = buffer.add_batch(batch(1000))
        self.assertEqual((window["sample_start"], window["sample_end"]), (0, 1249))
        self.assertEqual((len(window["samples"]), len(window["samples"][0])), (8, 1250))
        self.assertEqual(window["channel_names"], list(CHANNEL_NAMES))
        self.assertEqual(window["sample_rate_hz"], 250)
        self.assertIsNone(buffer.add_batch(batch(1250, 249)))
        window = buffer.add_batch(batch(1499, 1))
        self.assertEqual((window["sample_start"], window["sample_end"]), (250, 1499))
        self.assertEqual(buffer.sample_count, 1250)
        self.assertEqual(buffer.reset_count, 0)

    def test_large_batch_emits_only_latest_and_restarts_stride(self):
        buffer = CommandWindowBuffer()
        window = buffer.add_batch(batch(0, 3000))
        self.assertEqual((window["sample_start"], window["sample_end"]), (1750, 2999))
        self.assertIsNone(buffer.add_batch(batch(3000, 249)))
        self.assertEqual(buffer.add_batch(batch(3249, 1))["sample_end"], 3249)
        self.assertEqual(buffer.sample_count, 1250)

    def test_packet_counter_wraps_255_to_zero(self):
        buffer = CommandWindowBuffer()
        window = buffer.add_batch(batch(0, 1250, packet_offset=253))
        self.assertEqual(window["package_numbers"][:4], [253, 254, 255, 0])
        self.assertEqual(buffer.reset_count, 0)

    def test_host_burst_timestamps_never_deduplicate_samples(self):
        buffer = CommandWindowBuffer()
        window = buffer.add_batch(batch(0, 1250))
        self.assertEqual(window["timestamps"], [100.] * 1250)
        message = batch(1250)
        message["timestamps"] = [99.] * 250
        message["host_timestamp"] = 9.
        self.assertIsNotNone(buffer.add_batch(message))
        self.assertEqual(buffer.reset_count, 0)

    def test_native_timestamp_aliases_and_consistent_dual_aliases(self):
        buffer = CommandWindowBuffer()
        message = batch(0, 1250, native=True)
        message["timestamps"] = message["timestamps_s"].copy()
        message["host_timestamp"] = message["received_monotonic_s"]
        window = buffer.add_batch(message)
        self.assertEqual(window["timestamp_start"], 100.)
        self.assertEqual(window["host_timestamp_end"], 10.)
        self.assertEqual(buffer.reset_count, 0)

    def test_conflicting_timestamp_aliases_clear(self):
        for field, value in (("timestamps_s", [101.] * 250), ("received_monotonic_s", 12.)):
            buffer = CommandWindowBuffer()
            buffer.add_batch(batch(0, 500))
            message = batch(500)
            message[field] = value
            self.assertIsNone(buffer.add_batch(message))
            self.assertEqual(buffer.sample_count, 0)
            self.assertEqual(buffer.last_reset_reason, "conflicting_timestamps")

    def test_sample_start_can_replace_indices(self):
        buffer = CommandWindowBuffer()
        message = batch(91, 1250)
        del message["sample_indices"]
        message.update(sample_start=91, sample_end=1340)
        window = buffer.add_batch(message)
        self.assertEqual(window["sample_indices"], list(range(91, 1341)))

    def test_missing_timestamps_remain_unknown_not_synthesized(self):
        buffer = CommandWindowBuffer()
        message = batch(0, 1250)
        del message["timestamps"]
        del message["host_timestamp"]
        window = buffer.add_batch(message)
        self.assertEqual(window["timestamps"], [None] * 1250)
        self.assertIsNone(window["timestamp_end"])
        self.assertIsNone(window["host_timestamp_end"])

    def test_index_gap_clears_and_keeps_only_new_suffix(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1250))
        self.assertIsNone(buffer.add_batch(batch(1300, 50)))
        self.assertEqual(buffer.sample_count, 50)
        self.assertEqual(buffer.reset_count, 1)
        self.assertEqual(buffer.last_reset_reason, "sample_gap")
        window = buffer.add_batch(batch(1350, 1200))
        self.assertEqual(window["sample_start"], 1300)

    def test_gap_inside_batch_detected_without_a_complete_window(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1200))
        message = batch(1200, 100)
        message["sample_indices"][50:] = list(range(1300, 1350))
        message["package_numbers"][50:] = [i % 256 for i in range(1300, 1350)]
        self.assertIsNone(buffer.add_batch(message))
        self.assertEqual(buffer.sample_count, 50)
        self.assertEqual(buffer.reset_count, 1)
        self.assertEqual(buffer.reset_serial, 1)

    def test_repeated_gap_reason_increments_reset_count(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 50))
        buffer.add_batch(batch(100, 50))
        self.assertEqual(buffer.reset_count, 1)
        buffer.add_batch(batch(200, 50))
        self.assertEqual(buffer.reset_count, 2)
        self.assertEqual(buffer.last_reset_reason, "sample_gap")

    def test_packet_gap_with_continuous_indices(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1200))
        self.assertIsNone(buffer.add_batch(batch(1200, 50, packet_offset=1)))
        self.assertEqual(buffer.sample_count, 50)
        self.assertEqual(buffer.last_reset_reason, "sample_gap")

    def test_duplicate_batch_clears_without_reusing_replayed_samples(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1250))
        self.assertIsNone(buffer.add_batch(batch(1000, 250)))
        self.assertEqual(buffer.sample_count, 0)
        self.assertGreater(buffer.reset_count, 0)
        self.assertIsNone(buffer.add_batch(batch(1250)))
        self.assertEqual(buffer.sample_count, 250)

    def test_duplicate_packet_drops_point_and_breaks_window(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1200))
        message = batch(1200, 50)
        message["package_numbers"][0] = 1199 % 256
        self.assertIsNone(buffer.add_batch(message))
        self.assertEqual(buffer.sample_count, 49)
        self.assertGreaterEqual(buffer.reset_count, 1)

    def test_nonfinite_batch_is_atomic_and_recovery_is_fresh(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                buffer = CommandWindowBuffer()
                buffer.add_batch(batch(0, 1200))
                message = batch(1200, 250)
                message["samples"][3][200] = value
                self.assertIsNone(buffer.add_batch(message))
                self.assertEqual(buffer.sample_count, 0)
                self.assertEqual(buffer.last_reset_reason, "nonfinite_samples")
                self.assertIsNone(buffer.add_batch(batch(1450, 1249)))
                self.assertEqual(buffer.add_batch(batch(2699, 1))["sample_start"], 1450)

    def test_saturation_in_either_direction_clears_and_requires_1250_fresh_points(self):
        for value in (187000., -187000., 187001., -187001.):
            for channel in range(8):
                with self.subTest(value=value, channel=channel):
                    buffer = CommandWindowBuffer()
                    self.assertIsNotNone(buffer.add_batch(batch(0, 1250)))
                    message = batch(1250, 1)
                    message["samples"][channel][0] = value
                    self.assertIsNone(buffer.add_batch(message))
                    self.assertEqual(buffer.sample_count, 0)
                    self.assertEqual(buffer.last_reset_reason, "saturated_sample")
                    self.assertEqual(buffer.reset_count, 1)
                    self.assertEqual(buffer.generation, 1)
                    self.assertIsNone(buffer.add_batch(batch(1251, 1249)))
                    window = buffer.add_batch(batch(2500, 1))
                    self.assertEqual((window["sample_start"], window["sample_end"]),
                                     (1251, 2500))
                    self.assertTrue(all(abs(value) < 187000 for row in window["samples"]
                                        for value in row))

    def test_saturation_inside_batch_retains_only_suffix_and_invalidates_pending(self):
        buffer, gate = CommandWindowBuffer(), PendingPredictionGate()
        buffer.add_batch(batch(0, 1250))
        ids = gate.begin(1, now=0.)
        serial = buffer.reset_count
        message = batch(1250, 100)
        message["samples"][7][49] = -187000.
        self.assertIsNone(buffer.add_batch(message))
        self.assertEqual(buffer.sample_count, 50)
        self.assertEqual(buffer.reset_count, serial + 1)
        gate.invalidate(buffer.last_reset_reason)
        self.assertFalse(gate.resolve(ids, now=0.1)["accepted"])
        self.assertIsNone(buffer.add_batch(batch(1350, 1199)))
        window = buffer.add_batch(batch(2549, 1))
        self.assertEqual((window["sample_start"], window["sample_end"]), (1300, 2549))

    def test_repeated_saturation_resets_each_time_without_latching_warning(self):
        buffer = CommandWindowBuffer()
        message = batch(0, 4)
        message["samples"][0][0] = 187000.
        message["samples"][1][2] = -187000.
        self.assertIsNone(buffer.add_batch(message))
        self.assertEqual(buffer.reset_count, 2)
        self.assertEqual(buffer.sample_count, 1)
        self.assertIsNone(buffer.add_batch(batch(4, 1248)))
        window = buffer.add_batch(batch(1252, 1))
        self.assertEqual(window["sample_start"], 3)
        self.assertEqual(buffer.reset_count, 2)

    def test_values_immediately_below_saturation_threshold_are_retained(self):
        buffer = CommandWindowBuffer()
        message = batch(0, 1250)
        message["samples"][0][3] = 186999.999
        message["samples"][1][4] = -186999.999
        original = deepcopy(message)
        window = buffer.add_batch(message)
        self.assertEqual(window["samples"], original["samples"])
        self.assertEqual(message, original)
        self.assertEqual(buffer.reset_count, 0)

    def test_flat_channel_across_batches_clears_buffer(self):
        buffer = CommandWindowBuffer()
        first, second = batch(0, 200), batch(200, 50)
        first["samples"][4] = [3.] * 200
        second["samples"][4] = [3.] * 50
        buffer.add_batch(first)
        self.assertIsNone(buffer.add_batch(second))
        self.assertEqual(buffer.sample_count, 0)
        self.assertEqual(buffer.last_reset_reason, "flat_channel")
        self.assertEqual(buffer.reset_count, 1)
        self.assertEqual(buffer.add_batch(batch(250, 1250))["sample_start"], 250)

    def test_flat_epsilon_uses_anchor_not_incremental_drift(self):
        buffer = CommandWindowBuffer(flat_epsilon=1e-3)
        message = batch(0, 1250)
        message["samples"][0] = [i * 0.0001 for i in range(1250)]
        self.assertIsNotNone(buffer.add_batch(message))
        self.assertEqual(buffer.reset_count, 0)

    def test_invalid_shapes_metadata_and_numbers_clear_buffer(self):
        invalid = []
        for key, value in (("samples", None), ("sample_rate_hz", 256),
                           ("channel_names", None), ("sample_indices", None),
                           ("package_numbers", [256] * 250), ("timestamps", [float("nan")] * 250),
                           ("host_timestamp", float("inf")), ("sample_start", 99),
                           ("sample_end", 99), ("timestamp_source", 0)):
            message = batch(250)
            message[key] = value
            invalid.append(message)
        message = batch(250); message["samples"][0].pop(); invalid.append(message)
        message = batch(250); message["samples"][0][0] = True; invalid.append(message)
        message = batch(250); message["samples"][0][0] = "1"; invalid.append(message)
        message = batch(250); message["sample_indices"][0] = True; invalid.append(message)
        invalid.extend((None, [], "bad"))
        for message in invalid:
            with self.subTest(message=repr(message)[:80]):
                buffer = CommandWindowBuffer()
                buffer.add_batch(batch(0))
                self.assertIsNone(buffer.add_batch(message))
                self.assertEqual(buffer.sample_count, 0)
                self.assertEqual(buffer.reset_count, 1)

    def test_metadata_transition_does_not_mix_timestamp_sources(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1200))
        message = batch(1200, 50)
        message["timestamp_source"] = "device"
        self.assertIsNone(buffer.add_batch(message))
        self.assertEqual(buffer.sample_count, 50)
        self.assertEqual(buffer.last_reset_reason, "timestamp_metadata_changed")

    def test_clear_always_counts_and_releases_continuity_anchor(self):
        buffer = CommandWindowBuffer()
        buffer.add_batch(batch(0, 1250))
        buffer.clear("stop")
        buffer.clear("stop")
        self.assertEqual((buffer.reset_count, buffer.reset_serial), (2, 2))
        self.assertEqual(buffer.last_reset_reason, "stop")
        self.assertEqual(buffer.sample_count, 0)
        self.assertIsNotNone(buffer.add_batch(batch(0, 1250)))

    def test_input_and_emitted_window_are_detached(self):
        buffer = CommandWindowBuffer()
        message = batch(0, 1250)
        original = deepcopy(message)
        window = buffer.add_batch(message)
        self.assertEqual(message, original)
        window["samples"][0][-1] = 99999.
        message["samples"][0][-1] = 77777.
        following = buffer.add_batch(batch(1250))
        self.assertEqual(following["samples"][0][999], original["samples"][0][-1])

    def test_constructor_validation(self):
        for kwargs in ({"channel_names": ["x"] * 8}, {"flat_samples": 1},
                       {"flat_samples": 1251}, {"flat_epsilon": float("nan")}):
            with self.assertRaises(ValueError):
                CommandWindowBuffer(**kwargs)


class PendingPredictionGateTests(unittest.TestCase):
    def test_one_pending_then_resolve(self):
        gate = PendingPredictionGate()
        ids = gate.begin(7, now=10.)
        self.assertEqual(ids, {"request_id": 7, "generation": 0})
        self.assertIsNone(gate.begin(8, now=10.5))
        self.assertEqual(set(gate.pending), {"request_id", "generation", "started_at", "deadline"})
        response = {**ids, "prediction": {"command": "MOVE_LEFT"}}
        self.assertTrue(gate.resolve(response, now=11.)["accepted"])
        self.assertIsNone(gate.pending)
        self.assertIsNotNone(gate.begin(8, now=11.))

    def test_wrong_association_does_not_consume_pending(self):
        gate = PendingPredictionGate()
        ids = gate.begin(1, now=0.)
        for response in ({**ids, "request_id": 2}, {**ids, "generation": 1},
                         {**ids, "request_id": True}, {**ids, "generation": 0.}, None):
            self.assertFalse(gate.resolve(response, now=1.)["accepted"])
            self.assertIsNotNone(gate.pending)
        self.assertTrue(gate.resolve(ids, now=1.)["accepted"])
        self.assertFalse(gate.resolve(ids, now=1.1)["accepted"])

    def test_timeout_expires_and_invalidates_late_reply(self):
        gate = PendingPredictionGate()
        ids = gate.begin(1, now=0.)
        self.assertFalse(gate.expire(now=1.999))
        self.assertTrue(gate.expire(now=2.))
        self.assertEqual(gate.generation, 1)
        self.assertEqual(gate.last_reset_reason, "prediction_timeout")
        self.assertFalse(gate.expire(now=3.))
        new_ids = gate.begin(2, now=3.)
        self.assertFalse(gate.resolve(ids, now=3.1)["accepted"])
        self.assertTrue(gate.resolve(new_ids, now=3.2)["accepted"])

    def test_resolve_checks_timeout_even_without_timer(self):
        gate = PendingPredictionGate()
        ids = gate.begin(1, now=0.)
        audit = gate.resolve(ids, now=2.1)
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["reason"], "prediction_timeout")
        self.assertEqual(audit["raw_prediction"], ids)
        self.assertIsNone(gate.pending)

    def test_stop_and_gap_advance_generation_even_if_ids_reused(self):
        for reason in ("stop", "sample_gap", "flat_channel"):
            gate = PendingPredictionGate()
            old = gate.begin(1, now=0.)
            gate.invalidate(reason)
            new = gate.begin(1, now=0.5)
            self.assertEqual(new["generation"], 1)
            self.assertFalse(gate.resolve(old, now=0.6)["accepted"])
            self.assertTrue(gate.resolve(new, now=0.7)["accepted"])

    def test_raw_audit_and_pending_are_detached_copies(self):
        gate = PendingPredictionGate()
        ids = gate.begin(1, now=0.)
        pending = gate.pending
        pending["generation"] = 999
        response = {**ids, "probabilities": [0.8, 0.1, 0.1]}
        audit = gate.resolve(response, now=1.)
        response["probabilities"][0] = 0.
        self.assertEqual(audit["raw_prediction"]["probabilities"][0], 0.8)
        self.assertTrue(audit["accepted"])

    def test_same_generation_cannot_reuse_request_id(self):
        gate = PendingPredictionGate()
        ids = gate.begin(1, now=0.)
        gate.resolve(ids, now=1.)
        with self.assertRaises(ValueError):
            gate.begin(1, now=1.)

    def test_clock_injection_and_validation(self):
        clock = [10.]
        gate = PendingPredictionGate(clock=lambda: clock[0])
        gate.begin(1)
        clock[0] = 12.
        self.assertTrue(gate.expire())
        for request_id in (-1, True, "1", 1.5):
            with self.assertRaises(ValueError):
                gate.begin(request_id)
        with self.assertRaises(ValueError):
            gate.begin(1, now=float("nan"))
        for timeout in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                PendingPredictionGate(timeout)

    def test_ui_can_invalidate_on_an_internal_batch_gap(self):
        buffer, gate = CommandWindowBuffer(), PendingPredictionGate()
        window = buffer.add_batch(batch(0, 1250))
        ids = gate.begin(1, now=0.)
        serial = buffer.reset_count
        self.assertIsNone(buffer.add_batch(batch(1300, 50)))
        if buffer.reset_count != serial:
            gate.invalidate(buffer.last_reset_reason)
        self.assertEqual(buffer.sample_count, 50)
        self.assertFalse(gate.resolve(ids, now=0.1)["accepted"])
        self.assertEqual(window["sample_end"], 1249)


class CommandJournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_flush_utf8_records_and_hashes_do_not_touch_raw_manifest(self):
        raw = self.directory / "manifest.json"
        raw_bytes = b'{"raw":"immutable"}\n'
        raw.write_bytes(raw_bytes)
        journal = CommandJournal(self.directory)
        self.addCleanup(journal.close)
        records = {"event": {"reason": "开始", "time": 1.},
                   "command": {"command": "UNKNOWN", "raw_prediction": {"request_id": 3}},
                   "frame": {"frame": 1, "sample_end": 1249}}
        for kind, record in records.items():
            getattr(journal, "append_" + kind)(record)
            path = self.directory / journal.FILENAMES[kind]
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
        manifest = journal.close()
        self.assertEqual(raw.read_bytes(), raw_bytes)
        self.assertEqual(json.loads(journal.manifest_path.read_text(encoding="utf-8")), manifest)
        self.assertEqual(manifest["status"], "formal")
        for kind, entry in manifest["files"].items():
            data = (self.directory / entry["path"]).read_bytes()
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(entry["size_bytes"], len(data))
            self.assertEqual(entry["records"], 1)

    def test_nonfinite_and_unserializable_record_never_partially_write(self):
        journal = CommandJournal(self.directory)
        self.addCleanup(journal.close)
        journal.append_event({"valid": 1})
        path = self.directory / journal.FILENAMES["event"]
        before = path.read_bytes()
        for record in ({"nested": [float("nan")]}, {"v": float("inf")},
                       {"v": -float("inf")}, {"v": object()}):
            with self.assertRaises((TypeError, ValueError)):
                journal.append_event(record)
            self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(TypeError):
            journal.append_event([1, 2])
        self.assertEqual(journal.close()["files"]["event"]["records"], 1)

    def test_close_is_idempotent_and_rejects_further_append(self):
        journal = CommandJournal(self.directory)
        manifest = journal.close()
        manifest["files"]["event"]["records"] = 99
        self.assertEqual(journal.close()["files"]["event"]["records"], 0)
        with self.assertRaises(ValueError):
            journal.append_command({})

    def test_context_manager_closes_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "application stopped"):
            with CommandJournal(self.directory) as journal:
                journal.append_event({"reason": "stop"})
                raise RuntimeError("application stopped")
        self.assertTrue(journal.manifest_path.is_file())

    def test_existing_logs_and_manifest_are_never_truncated(self):
        with CommandJournal(self.directory) as journal:
            journal.append_command({"command": "UNKNOWN"})
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        with self.assertRaises(FileExistsError):
            CommandJournal(self.directory)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_unknown_kind_does_not_write(self):
        journal = CommandJournal(self.directory)
        self.addCleanup(journal.close)
        with self.assertRaises(ValueError):
            journal.append("raw", {"value": 1})
        self.assertTrue(all(path.stat().st_size == 0 for path in self.directory.iterdir()))

    def test_module_imports_only_standard_library(self):
        source = Path(__file__).resolve().parents[1] / "eeg_tools/realtime_commands.py"
        forbidden = {"torch", "numpy", "PySide6", "PyQt6", "brainflow", "eeg_tools"}
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            self.assertFalse({name.split(".")[0] for name in names} & forbidden)


if __name__ == "__main__":
    unittest.main()
