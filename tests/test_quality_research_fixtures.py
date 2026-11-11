"""Synthetic fixtures cannot silently become evidence or bless changed hashes."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tests import synthetic_research_fixtures as fixtures


class ResearchFixtureTests(unittest.TestCase):
    def test_contracts_are_explicitly_synthetic_and_independent_objects(self):
        for name in ('v3-preregistration.json', 'v3-dong-metadata.json', 'v4-preregistration.json',
                     'v4-interface.json', 'v13-plan.json', 'v5-parser-texts.json'):
            with self.subTest(name=name):
                first = fixtures.research_contract(name)
                self.assertEqual(first['fixture_scope'], fixtures.SCOPE)
                first['fixture_scope'] = 'modified'
                self.assertEqual(fixtures.research_contract(name)['fixture_scope'], fixtures.SCOPE)

    def test_unlisted_file_cannot_be_read(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError), self.assertRaises(ValueError):
            fixtures.research_contract('../history.json')

    def test_missing_synthetic_disclaimer_is_rejected(self):
        with patch.object(Path, 'read_text', return_value='{}'), self.assertRaisesRegex(ValueError, 'disclaim'):
            fixtures.research_contract('v13-plan.json')

    def test_changed_embedded_preregistration_does_not_change_frozen_pin(self):
        module = SimpleNamespace(_EXPECTED_PREREG_JSON='changed', PREREG_SHA256='0'*64)
        with self.assertRaisesRegex(ValueError, 'original frozen SHA256'):
            fixtures.embedded_preregistration_bytes(module)
        self.assertEqual(module.PREREG_SHA256, '0'*64)

    def test_nonempty_fixture_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); guard = root / 'keep'; guard.write_bytes(b'untouched')
            with self.assertRaisesRegex(ValueError, 'empty'):
                fixtures.failed_v14_fixture(root)
            self.assertEqual(guard.read_bytes(), b'untouched')
            self.assertEqual(list(root.iterdir()), [guard])

    def test_fixture_writer_requires_temporary_area(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('tempfile.gettempdir', return_value=str(root / 'not-this-root')):
                with self.assertRaisesRegex(ValueError, 'system temporary root'):
                    fixtures.failed_v14_fixture(root)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
