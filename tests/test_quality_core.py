import tempfile
import unittest
from pathlib import Path
import numpy as np
from eeg_tools.quality.filtering import CausalProcessor, ProcessingConfig
from eeg_tools.quality.artifacts import ArtifactConfig, ArtifactDetector, merge_intervals
from eeg_tools.quality.engine import QualityEngine
from eeg_tools.quality.cache import writer_lock, verified_revision

class QualityCoreTests(unittest.TestCase):
    def test_random_chunk_equivalence_and_nonfinite_reset(self):
        rng=np.random.default_rng(16); x=rng.normal(size=(2,2300));x[0,800:803]=np.nan
        cfg=ProcessingConfig(channels=('a','b'))
        whole=CausalProcessor(cfg).process(x,stream_id='s',start=0)
        proc=CausalProcessor(cfg); ys=[];masks=[];start=0
        while start<x.shape[1]:
            stop=min(x.shape[1],start+int(rng.integers(1,240)))
            result=proc.process(x[:,start:stop],stream_id='s',start=start)
            ys.append(result['samples']);masks.append(result['valid']);start=stop
        np.testing.assert_allclose(np.concatenate(ys,axis=1),whole['samples'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(np.concatenate(masks,axis=1),whole['valid'])
        self.assertTrue(np.isnan(whole['samples'][0,801]))
        self.assertFalse(whole['valid'][0,900])
    def test_stream_or_index_change_resets(self):
        cfg=ProcessingConfig(channels=('a',));x=np.ones((1,400));p=CausalProcessor(cfg)
        p.process(x,stream_id='one',start=0)
        changed=p.process(x,stream_id='two',start=400)
        fresh=CausalProcessor(cfg).process(x,stream_id='two',start=400)
        np.testing.assert_allclose(changed['samples'],fresh['samples'])
        self.assertNotEqual(cfg.processing_id,ProcessingConfig(channels=('a',),highpass_hz=1).processing_id)
    def test_every_spike_position_and_equal_threshold(self):
        cfg=ArtifactConfig(sample_rate_hz=10,amplitude_window_s=.3,absolute_uv=10,peak_to_peak_uv=10,step_uv=10,hardware_authority='test-fixture-not-hardware-validation')
        for position in range(24):
            x=np.zeros((1,24));x[0,position]=10
            whole=ArtifactDetector(['a'],cfg).process(x,stream_id='s',start=0)
            split=ArtifactDetector(['a'],cfg);intervals=[]
            for start in range(0,24,4):intervals.extend(split.process(x[:,start:start+4],stream_id='s',start=start)['intervals'])
            self.assertEqual(merge_intervals(intervals),whole['intervals'])
            self.assertTrue(any(r['rule']=='absolute' and r['start']<=position<r['stop'] for r in whole['intervals']))
    def test_unvalidated_thresholds_and_short_coverage(self):
        r=ArtifactDetector(['a'],ArtifactConfig(absolute_uv=1)).process(np.array([[9.]]),stream_id='s',start=0)
        self.assertEqual(r['rules']['absolute']['status'],'unknown')
        self.assertEqual(r['rules']['peak_to_peak']['unassessable'],1)
    def test_quality_dimensions_do_not_invent_good(self):
        p=QualityEngine(['a'])
        x=np.sin(2*np.pi*10*np.arange(2500)/250)[None,:]
        r=p.process(x,{'stream_id':'s','start':0,'stop':2500,'generation':0})
        self.assertEqual(r['dimensions']['signal_quality'],'unknown')
        self.assertEqual(r['dimensions']['coverage'],'pass')
        self.assertIsNotNone(r['psd']);self.assertEqual(r['dimensions']['stimulus_timing'],'unknown')
    def test_writer_lock_excludes_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'writer.lock'
            with writer_lock(path):
                with self.assertRaises(ValueError):
                    with writer_lock(path):pass
            with writer_lock(path):pass
