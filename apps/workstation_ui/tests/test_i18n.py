import ast
from pathlib import Path
import unittest

from apps.workstation_ui.i18n import Translator


class TranslationTests(unittest.TestCase):
    def test_chinese_default_and_english_fallback(self):
        tr = Translator()
        self.assertEqual(tr("nav.apps"), "采集应用")
        del tr.catalogs["zh-CN"]["nav.apps"]
        self.assertEqual(tr("nav.apps"), "Acquisition apps")
        self.assertEqual(Translator("unknown")("nav.apps"), "Acquisition apps")

    def test_catalogs_have_matching_keys_and_valid_format_fields(self):
        tr = Translator()
        self.assertEqual(set(tr.catalogs["zh-CN"]), set(tr.catalogs["en-US"]))
        self.assertIn("00:01:38", tr("ssvep.estimate", total="00:01:38", recording="00:01:33", trials=12))

    def test_ui_has_no_acquisition_core_imports(self):
        directory = Path(__file__).resolve().parents[1]
        forbidden = {"brainflow", "eeg_tools", "pylsl", "serial", "run_ssvep_session"}
        for file in directory.glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                names = [name.name for name in node.names] if isinstance(node, ast.Import) else (
                    [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, str(file))


if __name__ == "__main__":
    unittest.main()
