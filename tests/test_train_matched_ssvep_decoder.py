"""CPU-only contract tests for the bounded matched Wang decoder."""
from __future__ import annotations
import copy, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from scripts import train_matched_ssvep_decoder as train
from eeg_tools import matched_ssvep_decoder as decoder
from scripts import benchmark_matched_public_ssvep as baseline


def trial(subject, block, frequency, source="synthetic.mat"):
    t=np.arange(1250)/250
    x=np.stack([np.sin(2*np.pi*frequency*t+c*.07).astype(np.float32) for c in range(8)])
    return baseline.Trial(subject,block,"fixed","session",(decoder.FREQUENCIES.index(frequency) if frequency in decoder.FREQUENCIES else 99),float(frequency),baseline.canonical([subject,block,"fixed","session"]),x,source,((decoder.FREQUENCIES.index(frequency) if frequency in decoder.FREQUENCIES else 99)*6+block))

class MatchedDecoderTests(unittest.TestCase):
    def test_actual_sample_lengths_are_supported_without_padding(self):
        for n in (1088,1139,1250):
            features=decoder.spectral_features(np.ones((8,n),dtype=np.float32))
            self.assertEqual(features.shape,(72,)); self.assertTrue(np.isfinite(features).all())

    def test_exact_unknown_command_gate(self):
        accepted=decoder.command_prediction(np.asarray([.8,.1,.1]))
        self.assertEqual(accepted["command"],"MOVE_LEFT"); self.assertFalse(accepted["executed"])
        unknown=decoder.command_prediction(np.asarray([.45,.35,.20]))
        self.assertEqual(unknown["command"],"UNKNOWN"); self.assertFalse(unknown["accepted"])
        with self.assertRaises(ValueError): decoder.command_prediction(np.asarray([.8,.1,.1]), {"min_probability":.1})

    def test_checkpoint_selection_is_validation_only_and_deterministic(self):
        history=[{"epoch":1,"train_cross_entropy":.7,"validation_accuracy":.8,"validation_cross_entropy":.4},
                 {"epoch":2,"train_cross_entropy":.5,"validation_accuracy":.8,"validation_cross_entropy":.3},
                 {"epoch":3,"train_cross_entropy":.2,"validation_accuracy":.9,"validation_cross_entropy":.9},
                 {"epoch":4,"train_cross_entropy":.1,"validation_accuracy":.9,"validation_cross_entropy":.9}]
        self.assertEqual(train.select_epoch(history)["epoch"],3)
        bad=copy.deepcopy(history); bad[1]["validation_accuracy"]=float("nan")
        with self.assertRaises(ValueError): train.select_epoch(bad)

    def test_fit_saves_state_dict_contract_on_cpu_synthetic_only(self):
        train_trials=[trial(1,b,f) for b in range(2) for f in decoder.FREQUENCIES]
        val_trials=[trial(2,b,f) for b in range(1) for f in decoder.FREQUENCIES]
        model,state,history=train.fit_model(train_trials,val_trials,{"epochs":2,"batch_size":3,"learning_rate":.001,"weight_decay":.001},torch.device("cpu"))
        self.assertEqual(len(history),2); self.assertIsInstance(state,dict)
        self.assertEqual(set(state),set(model.state_dict()))
        self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
        self.assertEqual(model(torch.zeros((1,72))).shape,(1,3))

    def test_split_leakage_and_exact_frequency_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"S1.mat"; source.write_bytes(b"synthetic")
            spec_path=Path(tmp)/"spec.json"; spec_path.write_text("{}",encoding="utf-8")
            spec={"schema_version":1,"status":"formal"}
            prereg={"source_spec":spec,"source_spec_path":str(spec_path.resolve()),"source_hashes":{str(spec_path.resolve()):baseline.sha256_file(spec_path),str(source.resolve()):baseline.sha256_file(source)},"subjects":{"train":[1],"validation":[2],"test":[3]},"expected_trials":{"train":[],"validation":[],"test":[]}}
            # The frozen manifest deliberately requests 10/12/15 for one trial.
            row={"subject":1,"block":0,"frequency_hz":10,"condition_index":0,"source_trial_index":0,"source_file":str(source.resolve()),"group_id":baseline.canonical([1,0,"fixed","session"])}
            prereg["expected_trials"]["train"]=[row]
            dataset=baseline.Dataset([trial(1,0,20,str(source))],{"source_spec":spec,"source_spec_file_sha256":prereg["source_hashes"][str(spec_path.resolve())]}, {source.resolve():prereg["source_hashes"][str(source.resolve())]},"wang2016")
            with self.assertRaises(ValueError): train.validate_dataset(dataset,prereg,("train",))
            leak=copy.deepcopy(prereg); leak["subjects"]["validation"]=[1]
            with self.assertRaises(ValueError): train.validate_dataset(dataset,leak,("train","validation"))

    def test_require_cuda_never_silently_falls_back(self):
        with self.assertRaises(ValueError): train.select_device("cpu",require_cuda=True)

if __name__=="__main__": unittest.main()

