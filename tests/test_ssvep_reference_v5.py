"""Synthetic subject references only; no real D:/SSVEP waveforms/holdout metrics."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat,whosmat
from eeg_tools import ssvep_reference_v5 as reference
from scripts import prepare_ssvep_reference_v5 as cli
from tests.synthetic_research_fixtures import research_contract

ROOT=Path(__file__).resolve().parents[1]


class SSVEPReferenceV5Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.workspace=self.base/"workspace";self.workspace.mkdir()
        self.source=self.base/"source";self.source.mkdir()
        self.audit_path=self.workspace/"audit.json";self.lock_path=self.workspace/"subject_lock.json"
        self.output=self.workspace/"native_references"
        specs=deepcopy(reference.SPECS)
        specs["Nakanishi2015"]["shape"]=[12,8,47,2]
        specs["Wearable"]["shape"]=[8,128,2,2,12]
        self.spec_patch=patch.object(reference,"SPECS",specs);self.spec_patch.start();self.addCleanup(self.spec_patch.stop)
        texts = research_contract("v5-parser-texts.json")["texts"]
        for role,relative in reference.EVIDENCE_PATHS.items():
            target=self.workspace/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(texts[role], encoding="utf-8")
        inventory=[];self.originals={}
        for name,spec in reference.SPECS.items():
            for subject in range(1,11):
                path=self.source/spec["relative_template"].format(subject_id=subject)
                path.parent.mkdir(parents=True,exist_ok=True)
                array=np.arange(np.prod(spec["shape"]),dtype=np.float64).reshape(spec["shape"])+subject*1e6
                self.originals[(name,subject)]=array
                payload={spec["variable"]:array}
                if name=="Nakanishi2015":
                    payload["train"]=np.array([["CELL: NEVER NUMERICALLY SELECT"]],dtype=object)
                    payload["ans"]=np.array([[123.]],dtype=np.float64)
                savemat(path,payload,do_compression=True)
                stamp=reference.safety._stamp(path)
                inventory.append({"path":str(path),"relative_path":path.relative_to(self.source).as_posix(),"dataset":name,
                    "type":"matlab_v5","magic":{"type":"matlab_v5"},"status":"header_only","changed":False,
                    "errors":[],"sha256":reference.sha256(path),"stat_initial":stamp,"stat_before":stamp,"stat_after":stamp,"stat_final":stamp,
                    "mat":{"header_reader":"scipy.io.whosmat_metadata_only","variables":[{"name":key,"shape":list(shape),"matlab_class":kind} for key,shape,kind in whosmat(path)]}})
        self.audit={"schema_version":1,"audit_type":"read_only_local_ssvep_v5_point_in_time","root":str(self.source),
            "inventory_complete":True,"options":{"full_integrity":True},"inventory":inventory,
            "changes":{"new_files":[],"missing_files":[],"changed_files":[]}}
        self.write_audit()

    def write_audit(self):self.audit_path.write_text(json.dumps(self.audit,allow_nan=False),encoding="utf-8")
    def freeze(self):return reference.freeze_reference(self.audit_path,self.lock_path,workspace=self.workspace)
    def build(self,**kwargs):return reference.build_reference(self.lock_path,self.output,workspace=self.workspace,**kwargs)
    def write_lock(self,value):
        value["payload_sha256"]=reference._digest({k:v for k,v in value.items() if k!="payload_sha256"})
        self.lock_path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
    def path(self,name,subject):return self.source/reference.SPECS[name]["relative_template"].format(subject_id=subject)

    def test_freeze_only_audit_evidence_stat_no_source_bytes_or_waveform(self):
        original=reference.sha256;seen=[]
        def guarded(path):
            path=Path(path);seen.append(path)
            if path.is_relative_to(self.source):raise AssertionError("freeze source bytes read")
            return original(path)
        with patch.object(reference,"sha256",side_effect=guarded),patch.object(reference,"loadmat",side_effect=AssertionError("no waveforms")),patch.object(reference,"whosmat",side_effect=AssertionError("use audited header")):
            lock=self.freeze()
        self.assertEqual(len(lock["sources"]),20)
        self.assertEqual(lock["split_expected_trials"],{"train":432,"validation":144,"holdout":144})
        self.assertEqual(lock["total_expected_trials"],720)
        self.assertEqual(lock["partitions"],{"train":[1,2,3,4,5,6],"validation":[7,8],"holdout":[9,10]})
        self.assertTrue(lock["within_dataset_subject_disjoint"])
        self.assertFalse(lock["cross_dataset_participant_identity_verified"])
        self.assertEqual(len({row["group_key"] for row in lock["sources"]}),20)
        self.assertEqual(lock["status"],"formal")
        self.assertFalse(lock["raw_reference_is_clean_ground_truth"])
        self.assertFalse(lock["runtime"]["enabled"])
        self.assertIsNone(lock["dataset_specs"]["Nakanishi2015"]["channel_order"])
        self.assertIsNone(lock["dataset_specs"]["Nakanishi2015"]["units"])
        self.assertIsNone(lock["dataset_specs"]["Wearable"]["units"])
        self.assertEqual(reference.validate_lock(lock,self.workspace),self.source)
        self.assertNotIn("reference",lock)

    def test_default_shape_counts_without_any_real_source_read(self):
        # Shapes/constants only; no freeze/build or D file access.
        specs=deepcopy(reference.SPECS);specs["Nakanishi2015"]["shape"]=[12,8,1114,15];specs["Wearable"]["shape"]=[8,710,2,10,12]
        self.assertEqual(reference._trial_count("Nakanishi2015",specs["Nakanishi2015"]),180)
        self.assertEqual(reference._trial_count("Wearable",specs["Wearable"]),240)
        total=180+240
        self.assertEqual([total*n for n in (10,6,2,2,8)],[4200,2520,840,840,3360])
        self.assertEqual((180*8,240*8),(1440,1920))

    def test_4d_conversion_every_value_frequency_order_and_native_samples(self):
        original=self.originals[("Nakanishi2015",1)]
        members=reference.subject_to_trials("Nakanishi2015",original)
        self.assertEqual(members["reference"].shape,(24,47,8))
        self.assertEqual(members["reference"].dtype,np.float64)
        for trial in range(24):
            target,block=divmod(trial,2)
            np.testing.assert_array_equal(members["reference"][trial],original[target,:,:,block].T)
            self.assertEqual(members["target_id"][trial],target)
            self.assertEqual(members["block_id"][trial],block)
            self.assertEqual(members["frequency_hz"][trial],reference.FREQUENCIES[target])
            self.assertEqual(members["electrode_id"][trial],-1)
        self.assertNotIn("phase_pi_multiple",members)
        self.assertNotIn("clean",members)
        np.testing.assert_array_equal(original,self.originals[("Nakanishi2015",1)])

    def test_5d_conversion_every_value_dry_wet_order_phase_not_ground_truth_pair(self):
        original=self.originals[("Wearable",1)]
        members=reference.subject_to_trials("Wearable",original)
        self.assertEqual(members["reference"].shape,(48,128,8))
        for trial in range(48):
            target,remainder=divmod(trial,4);block,electrode=divmod(remainder,2)
            np.testing.assert_array_equal(members["reference"][trial],original[:,:,electrode,block,target].T)
            self.assertEqual(members["target_id"][trial],target)
            self.assertEqual(members["block_id"][trial],block)
            self.assertEqual(members["electrode_id"][trial],electrode)
            self.assertEqual(members["frequency_hz"][trial],reference.FREQUENCIES[target])
            self.assertEqual(members["phase_pi_multiple"][trial],reference.PHASES[target])
        self.assertEqual(reference.SPECS["Wearable"]["electrode_order"],["dry","wet"])
        self.assertEqual(reference.SPECS["Wearable"]["channel_order"],["POz","PO3","PO4","PO5","PO6","Oz","O1","O2"])

    def test_build_only_16_subject_variables_holdout_source_bytes_and_loads_forbidden(self):
        self.freeze();before={path:reference.sha256(path) for path in self.source.rglob("*.mat")}
        real_loader=reference.loadmat;real_header=reference.whosmat;real_sha=reference.sha256
        heldout={self.path(name,sid) for name in reference.SPECS for sid in (9,10)}
        reads=[]
        def guarded_load(path,**kwargs):
            self.assertNotIn(Path(path),heldout)
            name=next(name for name in reference.SPECS if name in str(path))
            self.assertEqual(kwargs["variable_names"],[reference.SPECS[name]["variable"]])
            reads.append(Path(path));return real_loader(path,**kwargs)
        def guarded_header(path):
            self.assertNotIn(Path(path),heldout);return real_header(path)
        def guarded_sha(path):
            self.assertNotIn(Path(path),heldout);return real_sha(path)
        with patch.object(reference,"loadmat",side_effect=guarded_load),patch.object(reference,"whosmat",side_effect=guarded_header),patch.object(reference,"sha256",side_effect=guarded_sha):
            manifest=self.build()
        self.assertEqual(len(reads),16)
        self.assertEqual(manifest["materialized_trial_counts"],{"train":432,"validation":144})
        self.assertEqual(manifest["total_materialized_trials"],576)
        self.assertEqual(len(manifest["holdout_identities"]),4)
        self.assertFalse(manifest["holdout_waveform_loaded"]);self.assertFalse(manifest["holdout_materialized"])
        self.assertFalse(manifest["model_metrics_computed"])
        for name in reference.SPECS:
            for sid in range(1,9):
                file=self.output/name/f"subject_{sid:03d}.npz"
                with np.load(file,allow_pickle=False) as saved:
                    self.assertNotIn("clean",saved.files)
                    self.assertTrue(all(saved[key].dtype.kind!="O" for key in saved.files))
                    expected=reference.subject_to_trials(name,self.originals[(name,sid)])
                    for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
                    metadata=json.loads(str(saved["manifest_json"].item()))
                    self.assertEqual(metadata["native_sample_rate_hz"],256 if name=="Nakanishi2015" else 250)
                    self.assertIsNone(metadata["native_units"])
                    self.assertTrue(all(v is False for v in metadata["signal_processing"].values()))
                    self.assertFalse(metadata["raw_reference_is_clean_ground_truth"])
            self.assertFalse((self.output/name/"subject_009.npz").exists())
        self.assertEqual({path:reference.sha256(path) for path in before},before)
        self.assertTrue((self.output/"manifest.json").exists())

    def test_wrong_audit_magic_shape_sha_and_download_changed_rejected(self):
        original=deepcopy(self.audit)
        for mode in ("magic","shape","sha","changed","error","incomplete_header","missing_stat","unknown_class"):
            with self.subTest(mode=mode):
                self.audit=deepcopy(original);row=self.audit["inventory"][0]
                if mode=="magic":row["magic"]["type"]="zip"
                elif mode=="shape":next(v for v in row["mat"]["variables"] if v["name"]=="eeg")["shape"]=[12,8,1114,15]
                elif mode=="sha":row["sha256"]="not a digest"
                elif mode=="changed":row["changed"]=True
                elif mode=="error":row["errors"]=[{"error":"in progress"}]
                elif mode=="incomplete_header":row["mat"]["header_reader"]="extension_only"
                elif mode=="missing_stat":del row["stat_after"]
                else:next(v for v in row["mat"]["variables"] if v["name"]=="eeg")["matlab_class"]="cell"
                self.write_audit()
                with self.assertRaises(reference.SSVEPReferenceV5Error):self.freeze()
                self.assertFalse(self.lock_path.exists())

    def test_lock_group_overlap_or_native_metadata_tampering_refused_even_rehashed(self):
        original=self.freeze()
        for mode in ("groups","source_group","channel_names","units","rate","ground_truth","runtime"):
            with self.subTest(mode=mode):
                changed=deepcopy(original)
                if mode=="groups":changed["partitions"]["validation"]=[6,8]
                elif mode=="source_group":changed["sources"][6]["split"]="train"
                elif mode=="channel_names":changed["dataset_specs"]["Nakanishi2015"]["channel_order"]=["Oz"]*8
                elif mode=="units":changed["dataset_specs"]["Nakanishi2015"]["units"]="microvolts"
                elif mode=="rate":changed["dataset_specs"]["Wearable"]["sample_rate_hz"]=256
                elif mode=="ground_truth":changed["raw_reference_is_clean_ground_truth"]=True
                else:changed["runtime"]["enabled"]=True
                self.write_lock(changed)
                with self.assertRaises(reference.SSVEPReferenceV5Error):self.build()
                self.assertFalse(self.output.exists())

    def test_payload_hash_tamper_not_recomputed_refused(self):
        lock=self.freeze();lock["created_at_utc"]="modified"
        self.lock_path.write_text(json.dumps(lock),encoding="utf-8")
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"payload SHA256"):self.build()
        self.assertFalse(self.output.exists())

    def test_source_changed_since_audit_prevents_freeze_without_waveform_read(self):
        path=self.path("Nakanishi2015",1)
        with path.open("ab") as handle:handle.write(b"download continued")
        with patch.object(reference,"loadmat",side_effect=AssertionError("must not read waveform")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"changed since audit"):self.freeze()
        self.assertFalse(self.lock_path.exists())

    def test_source_changed_after_freeze_refused_before_loading_and_output_creation(self):
        self.freeze();path=self.path("Nakanishi2015",1)
        with path.open("ab") as handle:handle.write(b"download continued")
        with patch.object(reference,"loadmat",side_effect=AssertionError("must refuse before loading")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"stat changed"):self.build()
        self.assertFalse(self.output.exists())

    def test_source_sha_mismatch_even_with_mocked_unchanged_stamp(self):
        lock=self.freeze();row=lock["sources"][0];path=Path(row["path"])
        original=reference.safety._stamp
        with path.open("ab") as handle:handle.write(b"changed bytes")
        def stamp(current):return deepcopy(row["audit_stat"]) if Path(current)==path else original(current)
        with patch.object(reference.safety,"_stamp",side_effect=stamp),patch.object(reference,"loadmat",side_effect=AssertionError("must refuse before loading")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"SHA256 mismatch"):self.build()
        self.assertFalse(self.output.exists())

    def test_evidence_or_bound_audit_changed_refused_before_subject_loading(self):
        self.freeze()
        evidence=self.workspace/reference.EVIDENCE_PATHS["nakanishi_author"]
        original=evidence.read_bytes();evidence.write_bytes(original+b"different evidence")
        with patch.object(reference,"loadmat",side_effect=AssertionError("no waveform")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"evidence SHA256 changed"):self.build()
        self.assertFalse(self.output.exists())
        evidence.write_bytes(original)
        self.audit_path.write_text(self.audit_path.read_text(encoding="utf-8")+" ",encoding="utf-8")
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"audit/evidence SHA256 changed"):self.build()
        self.assertFalse(self.output.exists())

    def test_author_evidence_inconsistent_facts_refused(self):
        path=self.workspace/reference.EVIDENCE_PATHS["wearable_stimulation"]
        path.write_text(path.read_text(encoding="utf-8").replace("9.25","8.25",1),encoding="utf-8")
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"stimulation evidence"):self.freeze()
        self.assertFalse(self.lock_path.exists())

    def test_output_source_or_outside_workspace_and_existing_refused(self):
        self.freeze()
        with patch.object(reference,"loadmat",side_effect=AssertionError("must not load")):
            for destination in (self.source/"output",self.base/"outside"):
                with self.subTest(destination=destination):
                    with self.assertRaises(reference.SSVEPReferenceV5Error):reference.build_reference(self.lock_path,destination,workspace=self.workspace)
            self.output.mkdir();(self.output/"keep.txt").write_text("keep",encoding="utf-8")
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"overwrite"):self.build()
        self.assertEqual((self.output/"keep.txt").read_text(),"keep")
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"overwrite"):self.freeze()
        with self.assertRaises(reference.SSVEPReferenceV5Error):reference.freeze_reference(self.audit_path,self.source/"lock.json",workspace=self.workspace)

    def test_mocked_junction_source_and_output_rejected(self):
        self.freeze();original=reference.safety._stamp
        path=self.workspace/"junction_output";path.mkdir()
        def stamp(current):
            value=original(current)
            if Path(current)==path:value={**value,"file_attributes":value["file_attributes"]|0x400}
            return value
        with patch.object(reference.safety,"_stamp",side_effect=stamp):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"junction"):reference.build_reference(self.lock_path,path/"child",workspace=self.workspace)
        source_parent=self.source/"Nakanishi2015"
        def source_stamp(current):
            value=original(current)
            if Path(current)==source_parent:value={**value,"file_attributes":value["file_attributes"]|0x400}
            return value
        with patch.object(reference.safety,"_stamp",side_effect=source_stamp),patch.object(reference,"loadmat",side_effect=AssertionError("no follows")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"junction"):self.build()

    def test_nonfinite_object_complex_wrong_shape_no_interpolation_or_partial_success(self):
        good=self.originals[("Nakanishi2015",1)]
        cases=[good.astype(complex),good.astype(object),good[:,:,:-1,:]]
        bad=good.copy();bad[0,0,0,0]=np.nan;cases.append(bad)
        for array in cases:
            with self.subTest(dtype=array.dtype,shape=array.shape):
                with self.assertRaises(reference.SSVEPReferenceV5Error):reference.subject_to_trials("Nakanishi2015",array)
        self.freeze();real=reference.loadmat
        def bad_result(path,**kwargs):
            result=real(path,**kwargs)
            if Path(path)==self.path("Nakanishi2015",1):result["eeg"][0,0,0,0]=np.nan
            return result
        with patch.object(reference,"loadmat",side_effect=bad_result):
            with self.assertRaises(reference.SSVEPReferenceV5Error):self.build()
        self.assertFalse((self.output/"manifest.json").exists())
        failure=json.loads((self.output/"build_failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["build_completed"])
        self.assertFalse(failure["holdout_waveform_loaded"])
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"overwrite"):self.build()

    def test_working_memory_budget_rejects_before_loadmat(self):
        lock=self.freeze();row=lock["sources"][0]
        with patch.object(reference,"loadmat",side_effect=AssertionError("over-budget load")):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"working-data budget"):
                reference._materialize_subject(self.source,row,1*reference.safety.CHUNK)
        for size in (0,257):
            with self.assertRaises(reference.SSVEPReferenceV5Error):self.build(memory_budget_mb=size)

    def test_path_traversal_and_duplicate_subject_audit_rejected(self):
        self.audit["inventory"][0]["path"]=str(self.base/"wrong.mat");self.write_audit()
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"namespace mismatch"):self.freeze()
        self.audit["inventory"][0]["path"]=str(self.path("Nakanishi2015",1))
        self.audit["inventory"].append(deepcopy(self.audit["inventory"][0]));self.write_audit()
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"duplicate audit"):self.freeze()
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"safe POSIX"):
            reference._relative_source(self.source,"../outside.mat")

    def test_change_during_selected_load_rejected_and_not_completed(self):
        self.freeze();real=reference.loadmat
        def growing(path,**kwargs):
            result=real(path,**kwargs)
            with Path(path).open("ab") as handle:handle.write(b"concurrent download")
            return result
        with patch.object(reference,"loadmat",side_effect=growing):
            with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"changed/inprogress"):self.build()
        self.assertFalse((self.output/"manifest.json").exists())
        self.assertTrue((self.output/"build_failure.json").exists())

    def test_relative_metadata_outputs_resolve_within_explicit_workspace(self):
        self.assertEqual(reference._inside(self.workspace,"relative/output.json"),self.workspace/"relative/output.json")
        with self.assertRaises(reference.SSVEPReferenceV5Error):reference._inside(self.workspace,"../escape.json")
        self.audit["root"]="relative/source";self.write_audit()
        with self.assertRaisesRegex(reference.SSVEPReferenceV5Error,"explicitly absolute"):self.freeze()
        self.assertFalse(self.lock_path.exists())

    def test_cli_two_stages_defaults_no_train_or_model_option(self):
        freeze=cli.make_parser().parse_args(["freeze","--audit","audit.json","--output-lock","lock.json"])
        build=cli.make_parser().parse_args(["build","--lock","lock.json","--output-dir","refs"])
        self.assertEqual(freeze.command,"freeze");self.assertEqual(build.command,"build")
        self.assertEqual(build.memory_budget_mb,256)
        self.assertFalse(hasattr(build,"model"))

if __name__=="__main__":unittest.main()
