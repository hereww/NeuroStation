"""Frequency display regression tests independent of hardware and acceptance gates."""
import math
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from apps.workstation_ui.quality_panel import QualityPanel
from apps.workstation_ui.quality_spectrum import BANDS, band_shares, visible_peaks


class SpectrumTests(unittest.TestCase):
    def test_nonuniform_grid_integrates_at_exact_edges(self):
        hz = [0, 1, 3, 7, 11, 18, 28, 42, 50, 60]
        # PSD = 2f gives the analytic integral hi² - lo² even between samples.
        shares = band_shares(hz, [2*f for f in hz])
        for share, (_, lo, hi, _) in zip(shares, BANDS):
            self.assertAlmostEqual(share, (hi**2 - lo**2) / (45**2 - .5**2))

    def test_missing_band_or_zero_denominator_has_no_percent(self):
        hz = list(range(51))
        self.assertIsNone(band_shares(hz, [0] * len(hz)))
        self.assertIsNone(band_shares(hz[:30], [1] * 30))
        self.assertIsNone(band_shares(hz, [1]*12 + [None] + [1]*38))
        self.assertAlmostEqual(sum(band_shares(hz, [1]*46 + [None]*5)), 1)

    def test_hz_viewport_and_real_peaks(self):
        hz = [0, 2, 4, 6, 9, 10, 13, 40, 50, 60, 82, 125]
        power = [0, 20, 0, .0001, 0, 45, 0, 0, 2, 0, 60, 0]
        self.assertEqual([hz[i] for i in visible_peaks(hz, power, 65)], [10, 2, 50])
        self.assertEqual([hz[i] for i in visible_peaks(hz, power, 125)], [82, 10, 2, 50])

    def test_channel_range_and_clear_update_together(self):
        app = QApplication.instance() or QApplication([])
        panel = QualityPanel(spectrum_first=True)
        self.addCleanup(panel.close)
        hz = list(range(126))
        rows = [[float(i == peak) for i in hz] for peak in (6, 38)]
        rows[1][82] = 2.
        record = {'schema_version': 2, 'start': 0, 'stop': 2000, 'dimensions': {},
                  'channels': [{'channel':'CH1'}, {'channel':'CH2'}],
                  'psd': {'frequency_hz':hz, 'power_uv2_hz':rows, 'sample_stop':2000}}
        panel.update_quality(record)
        view = panel.spectrum
        self.assertIn('6 Hz', view.peak_readout.text())
        self.assertIn('100.0%', view.band_widgets[1][1].text())
        panel.channel.setCurrentIndex(1)
        self.assertIn('CH2', view.title.text())
        self.assertIn('38 Hz', view.peak_readout.text())
        self.assertNotIn('82 Hz', view.peak_readout.text())
        self.assertIn('100.0%', view.band_widgets[4][1].text())
        view.range.setCurrentIndex(1)
        self.assertIn('82 Hz', view.peak_readout.text())
        self.assertEqual(view.chart.hz_limit(), 125.)
        self.assertIn('2000', view.note.text())
        panel.update_quality({**record, 'psd': None})
        self.assertEqual(view.chart.frequency, [])
        self.assertIn('—', view.peak_readout.text())
        self.assertNotIn('2000', view.note.text())
        self.assertTrue(all('—' in label.text() for _, label, _ in view.band_widgets))
        for axis in ([2, 1], [1, 1], [math.nan, 2]):
            panel.update_quality({**record, 'psd': {'frequency_hz':axis, 'power_uv2_hz':[[1, 1]]*2}})
            self.assertIsNone(panel.latest)
        app.processEvents()
