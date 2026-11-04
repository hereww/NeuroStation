import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.verify_denoiser_v4_artifacts import checked_path, verify_files, verify_lock


class VerifyV4ArtifactsTests(unittest.TestCase):
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for p in ("..\\outside", "../outside", str(root.parent / "outside"), "."):
                with self.subTest(path=p), self.assertRaises(ValueError):
                    checked_path(root, p)

    def test_sha_mismatch_detected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root / "data.bin"; p.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            self.assertEqual(verify_files(root, {"data.bin": expected}), [])
            p.write_bytes(b"changed")
            diff = verify_files(root, {"data.bin": expected})
            self.assertEqual(diff[0]["path"], "data.bin")
            self.assertEqual(p.read_bytes(), b"changed")

    def test_strict_digest_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for digest in (None, "a" * 63, "G" * 64):
                with self.subTest(digest=digest), self.assertRaises(ValueError):
                    verify_files(root, {"x": digest})
            self.assertIsNone(verify_files(root, {"x": "a" * 64})[0]["actual"])

    def test_lock_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "src").write_bytes(b"code")
            lock = {"files": {"src": hashlib.sha256(b"code").hexdigest()}}
            (root / "lock.json").write_text(json.dumps(lock), encoding="utf-8")
            self.assertTrue(verify_lock(root, "lock.json")["passed"])
            (root / "src").write_bytes(b"modified")
            with self.assertRaises(ValueError): verify_lock(root, "lock.json")


if __name__ == "__main__":
    unittest.main()
