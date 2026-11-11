"""Synthetic MAT/NPZ tests only; never open the research source corpus."""
from __future__ import annotations
from tests.synthetic_research_fixtures import research_contract
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly
from eeg_tools import learned_dataset_v4 as data

ROOT = Path(__file__).resolve().parents[1]


def fixture(root: Path, samples=801):
    prereg = research_contract('v4-preregistration.json')
    interface = research_contract('v4-interface.json')
    d = prereg["data"]
    d.update(source_segments=3, group_assignments={"1": 3, "2": 1, "3": 2},
             train_groups=[3], validation_groups=[1], test_groups=[2])
    sources = {k: {} for k in ("clean", "noisy", "heog", "veog")}
    for segment in range(1, 4):
        t = np.arange(samples, dtype=np.float64) / 200
        pure = np.stack([.2*segment + np.sin(2*np.pi*(8+c/10)*t+.07*c) + .3*np.cos(2*np.pi*19*t) for c in range(19)])
        eog = .7*np.sin(2*np.pi*2*t+.1*segment)
        sources["clean"][f"sim{segment}_resampled"] = pure
        sources["noisy"][f"sim{segment}_con"] = pure + .15 + eog
        sources["heog"][f"heog_{segment}"] = eog[None]
        sources["veog"][f"veog_{segment}"] = (eog*.6)[None]
    d["sources"] = {}
    for role, values in sources.items():
        path = root / f"{role}.mat"
        savemat(path, values)
        d["sources"][role] = {"relative_path": path.name, "sha256": data.sha256(path)}
    for rel, value in ((data.DEFAULT_PREREGISTRATION_RELATIVE, prereg), (data.DEFAULT_INTERFACE_RELATIVE, interface)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return prereg, sources


def rewrite_archive(path, mutate):
    with np.load(path, allow_pickle=False) as archive:
        members = {k: np.array(archive[k]) for k in archive.files}
    manifest = json.loads(Path(str(path)+".manifest.json").read_text(encoding="utf-8"))
    mutate(members, manifest)
    base = {k:v for k,v in manifest.items() if k != "dataset_file_sha256"}
    members["manifest_json"] = np.asarray(json.dumps(base))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **members)
    manifest["dataset_file_sha256"] = data.sha256(path)
    Path(str(path)+".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class DatasetV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prereg, self.sources = fixture(self.root)
        self.path = self.root / data.DEFAULT_DATASET_RELATIVE

    def build(self):
        return data.build_dataset(self.root)

    def test_exact_resample_all_channels_full_windows_identity_and_original_gt_dc(self):
        result = self.build()
        self.assertEqual(result["manifest"]["split_counts"], dict(train=76, validation=76, test=76))
        pure = resample_poly(self.sources["clean"]["sim1_resampled"],5,4,axis=1)
        noisy = resample_poly(self.sources["noisy"]["sim1_con"],5,4,axis=1)
        loaded = data.load_split(self.path,"train")
        np.testing.assert_array_equal(loaded["clean"][0], pure[0,:500])
        np.testing.assert_array_equal(loaded["noisy"][0], noisy[0,:500])
        np.testing.assert_array_equal(loaded["noisy"][1], pure[0,:500])
        mean, rms = noisy[0,:500].mean(), np.sqrt(np.mean((noisy[0,:500]-noisy[0,:500].mean())**2))
        np.testing.assert_array_equal(loaded["target"][0,0], ((pure[0,:500]-mean)/rms).astype(np.float32))
        self.assertGreater(abs(float(loaded["target"][0].mean())), .05)
        self.assertEqual(loaded["group_id"].tolist(), [3]*76)
        self.assertEqual(result["manifest"]["tails"][0]["tail_samples"], 2)
        self.assertTrue(result["manifest"]["no_subject_or_electrode_inference"])
        self.assertIsNone(result["manifest"]["physical_units"])
        self.assertEqual(loaded["noisy"].dtype, np.float64)
        self.assertEqual(loaded["normalized"].dtype, np.float32)
        with np.load(self.path,allow_pickle=False) as npz:
            self.assertEqual(npz["manifest_json"].dtype.kind,"U")
            self.assertTrue(all(npz[k].dtype.kind != "O" for k in npz.files))
        self.assertEqual(len(data.load_evaluation_split(self.path)["clean"]),76)

    def test_training_loader_refuses_test_and_does_not_get_test_members(self):
        self.build()
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"only train"):
            data.load_split(self.path,"test")
        with np.load(self.path) as archive:
            klass = type(archive)
        original, seen = klass.__getitem__, []
        def audited(archive,key):
            seen.append(key)
            if key.startswith("test_"):
                raise AssertionError("training selected a test member")
            return original(archive,key)
        with patch.object(klass,"__getitem__",audited):
            data.load_split(self.path,"train")
            data.load_split(self.path,"validation")
        self.assertNotIn("test_clean",seen)

    def test_source_hash_change_rejected(self):
        with (self.root/"clean.mat").open("ab") as handle: handle.write(b"changed")
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"SHA256"):
            self.build()

    def test_group_overlap_rejected_no_random_fallback(self):
        self.prereg["data"]["validation_groups"]=[3]
        (self.root/data.DEFAULT_PREREGISTRATION_RELATIVE).write_text(json.dumps(self.prereg),encoding="utf-8")
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"overlap"):
            self.build()

    def test_check_only_no_publication_and_refuse_overwrite(self):
        result = data.build_dataset(self.root, check_only=True)
        self.assertTrue(result["check_only"])
        self.assertFalse(self.path.exists())
        self.build()
        digest=data.sha256(self.path)
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"overwrite"): self.build()
        self.assertEqual(data.sha256(self.path),digest)

    def test_bad_source_shape_and_nonfinite_and_constant_not_silently_removed(self):
        for mode in ("shape","nonfinite","constant"):
            with self.subTest(mode=mode):
                values=deepcopy(self.sources["noisy"])
                if mode=="shape": values["sim1_con"]=values["sim1_con"][:18]
                elif mode=="nonfinite": values["sim1_con"][0,0]=np.nan
                else: values["sim1_con"][:]=0
                savemat(self.root/"noisy.mat",values)
                self.prereg["data"]["sources"]["noisy"]["sha256"]=data.sha256(self.root/"noisy.mat")
                (self.root/data.DEFAULT_PREREGISTRATION_RELATIVE).write_text(json.dumps(self.prereg),encoding="utf-8")
                with self.assertRaises(data.LearnedDatasetV4Error): self.build()
                self.assertFalse(self.path.exists())

    def test_npz_hash_and_normalization_tampering_rejected(self):
        self.build()
        with self.path.open("ab") as f:f.write(b"changed")
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"SHA256"): data.load_split(self.path,"train")
        rewrite_archive(self.path,lambda arrays,manifest:arrays["train_normalized"].__setitem__((0,0,0),123))
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"normalization"): data.load_split(self.path,"train")

    def test_manifest_rows_and_counts_tampering_rejected(self):
        self.build()
        rewrite_archive(self.path,lambda a,m:m["window_hashes"]["train"]["rows"][0].__setitem__("channel_index",8))
        with self.assertRaisesRegex(data.LearnedDatasetV4Error,"hash"): data.load_split(self.path,"train")

    def test_atomic_pair_publication_rolls_back_only_own_files(self):
        import os
        original=os.link
        calls=[]
        def fail_second(src,dst):
            calls.append(dst)
            if len(calls)==2: raise OSError("synthetic publication failure")
            return original(src,dst)
        with patch("os.link",side_effect=fail_second):
            with self.assertRaisesRegex(OSError,"synthetic"): self.build()
        self.assertFalse(self.path.exists())
        self.assertFalse(Path(str(self.path)+".manifest.json").exists())
        self.assertEqual(list(self.path.parent.glob("*.pending")),[])

if __name__ == "__main__":unittest.main()
