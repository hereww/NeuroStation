"""Offscreen tests for selecting motion in the existing acquisition page."""
import os
import unittest
from pathlib import Path
import tempfile

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from apps.workstation_ui.pages import SSVEPPage
from apps.workstation_ui.i18n import Translator
from neurostation_contract import CaptureConfig, StimulusCondition, UserProfile


class SinglePassUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.user = UserProfile(user_id='U001',name='Operator test',age=25)
        self.page = SSVEPPage(Translator('zh-CN'), CaptureConfig(save_directory=Path(self.temp.name)),
                             lambda page: None, users=(self.user,))
        self.addCleanup(self.page.close)

    def test_all_four_modes_emit_correct_config_and_preview_without_running_hardware(self):
        requests = []
        self.page.start_requested.connect(lambda config, speed: requests.append(config))
        self.page.acknowledge.setChecked(True)
        for mode in StimulusCondition:
            index = self.page.eye_side.findData(mode.value)
            self.assertGreater(index,0)
            self.page.eye_side.setCurrentIndex(index)
            self.assertEqual(mode.value,self.page.stimulus_preview.eye_side)
            self.assertIn('_'+mode.dataset_suffix,self.page.dataset_preview.text())
            self.page._start()
            config = requests[-1]
            config.validate()
            self.assertEqual(mode.start_side,config.eye_side)
            self.assertEqual(mode.value,config.stimulus_condition)
            self.assertEqual(40,config.trials)  # existing four-frequency default, not four spatial groups
        self.assertEqual(4,len(requests))

    def test_motion_does_not_bypass_risk_ack(self):
        requests = []
        self.page.start_requested.connect(lambda *args: requests.append(args))
        self.page.eye_side.setCurrentIndex(self.page.eye_side.findData('left_to_right'))
        self.page._start()
        self.assertEqual([],requests)
        self.assertTrue(self.page.error.text())

    def test_language_rebuild_preserves_motion_and_parameters(self):
        self.page.eye_side.setCurrentIndex(self.page.eye_side.findData('right_to_left'))
        self.page.stimulus.setValue(7)
        self.page.repetitions.setValue(3)
        config = self.page.config()
        restored = SSVEPPage(Translator('en-US'),config,lambda page:None,users=(self.user,))
        self.addCleanup(restored.close)
        self.assertEqual('right_to_left',restored.eye_side.currentData())
        self.assertEqual(7,restored.config().stimulus_seconds)
        self.assertEqual(3,restored.config().repetitions)
        self.assertEqual('right_to_left',restored.stimulus_preview.eye_side)
        self.assertIn('_右到左',restored.dataset_preview.text())

    def test_motion_preview_is_a_static_image(self):
        self.page.eye_side.setCurrentIndex(self.page.eye_side.findData('left_to_right'))
        preview = self.page.stimulus_preview.display
        preview.resize(400,160)
        first = preview.grab().toImage()
        self.app.processEvents()
        second = preview.grab().toImage()
        self.assertFalse(first.isNull())
        self.assertEqual(first,second)


if __name__=='__main__':
    unittest.main()
