"""Acceptance selection and fixture identity, not a historical-validation waiver."""
from pathlib import Path
import tempfile
import unittest

from scripts import verify_quality_upgrade as runner


class AcceptanceRunnerTests(unittest.TestCase):
    def test_fixture_json_and_build_inputs_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['tests/fixtures/research_contracts/test.json', 'entry.py', 'build.bat',
                     'quality-output/ignored.json', 'docs/note.md']
            for name in names:
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('first')
            before = runner.source_manifest(root, names)
            self.assertEqual(set(before), set(names[:3]))
            (root / names[0]).write_text('modified fixture')
            self.assertEqual(runner.manifest_changes(before, runner.source_manifest(root, names)), [names[0]])

    def test_source_additions_and_removals_are_changes(self):
        self.assertEqual(runner.manifest_changes({'a':'1', 'b':'2'}, {'b':'2', 'c':'3'}), ['a','c'])

    def test_contract_exclusion_is_explicit_not_a_pass_or_skip(self):
        excluded_id = next(iter(runner.CONTRACT_EXCLUSIONS))
        class Probe(unittest.TestCase):
            def __init__(self, name): super().__init__(); self.name = name
            def id(self): return self.name
            def runTest(self): raise AssertionError('selection must not execute tests')
        selected, excluded = runner.select_contract_tests(unittest.TestSuite([
            Probe('test_synthetic.Suite.test_contract'), unittest.TestSuite([Probe('tests.' + excluded_id)])]))
        self.assertEqual(list(runner.collect_ids(selected)), ['test_synthetic.Suite.test_contract'])
        self.assertEqual(excluded, [{'id':'tests.'+excluded_id,
                                    'reason':runner.CONTRACT_EXCLUSIONS[excluded_id],
                                    'status':'not_run_in_contracts'}])

    def test_full_selection_is_not_the_portable_subset(self):
        # The historical method is retained in the source and is only filtered
        # at contracts selection. The full discover call does not use this filter.
        root = Path(__file__).resolve().parents[1]
        self.assertTrue(all((root/'tests'/f'{name}.py').is_file() for name in runner.CONTRACT_MODULES))
        self.assertNotIn('test_verify_ssvep_v14_final', runner.CONTRACT_MODULES)
        self.assertIn('test_personal_sessions_are_fixed_and_present',
                      (root/'tests/test_evaluate_ssvep_denoiser_v14.py').read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
