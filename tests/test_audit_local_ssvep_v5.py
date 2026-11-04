"""Synthetic local audit fixtures only; never inspect real D:/SSVEP."""
from __future__ import annotations
from copy import deepcopy
import builtins
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from scipy.io import savemat
from scripts import audit_local_ssvep_v5 as audit


class LocalSSVEPAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name);self.root=self.base/"synthetic_source";self.root.mkdir()
        for i in range(9):(self.root/f"dataset{i+1}").mkdir()
        self.folder=self.root/"dataset1";self.output=self.base/"reports"/"synthetic_run"

    def mat(self,name="a.mat",array=None,folder=None,compress=True):
        path=(folder or self.folder)/name
        if array is None:array=np.arange(1500,dtype=np.float64).reshape(3,500)/10
        savemat(path,{"unknown_axes":array},do_compression=compress);return path

    def zip(self,name="data.zip",unsafe=False):
        path=self.folder/name
        with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED) as handle:
            handle.writestr("signal.bin",b"safe synthetic signal payload")
            if unsafe:handle.writestr("../../escaped.txt",b"NEVER EXTRACT")
        return path

    def tgz(self,name="data.tar.gz",unsafe=False,tail=b""):
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode="w") as archive:
            content=b"synthetic signal";entry=tarfile.TarInfo("signal.bin");entry.size=len(content);archive.addfile(entry,io.BytesIO(content))
            if unsafe:
                entry=tarfile.TarInfo("../escaped.txt");entry.size=4;archive.addfile(entry,io.BytesIO(b"oops"))
                entry=tarfile.TarInfo("absolute-link");entry.type=tarfile.SYMTYPE;entry.linkname="C:/outside";archive.addfile(entry)
        path=self.folder/name;path.write_bytes(gzip.compress(payload.getvalue()+tail));return path

    def scan(self,**options):
        settings=dict(root=self.root,report_prefix=self.output,settle_seconds=0)
        settings.update(options)
        with patch.object(audit,"find_seven_zip",return_value=None):return audit.audit_local(**settings)

    def row(self,report,name):return next(r for r in report["inventory"] if r["relative_path"]=="dataset1/"+name)

    def test_mat_sha_header_probe_inventory_and_no_scientific_assumptions(self):
        path=self.mat()
        before=path.read_bytes()
        report=self.scan()
        row=self.row(report,"a.mat")
        self.assertEqual(row["type"],"matlab_v5")
        self.assertEqual(row["sha256"],hashlib.sha256(before).hexdigest())
        self.assertTrue(row["ready_stable"])
        self.assertEqual(row["mat"]["arrays"][0]["shape"],[3,500])
        self.assertEqual(row["mat"]["arrays"][0]["real_range"],[0.,149.9])
        self.assertEqual(row["label_status"],"unknown")
        self.assertEqual(row["clean_target_status"],"not_defined")
        self.assertIn("unknown",row["mat"]["arrays"][0]["axis_interpretation"])
        self.assertEqual(report["observed_dataset_count"],9)
        self.assertTrue(report["dataset_count_matches_expected"])
        self.assertEqual(report["datasets"][0]["ready_stable_count"],1)
        self.assertEqual(report["datasets"][0]["signal_bytes"],path.stat().st_size)
        self.assertFalse(report["train_eligibility_assessed"])
        self.assertTrue(report["point_in_time_only"]);self.assertFalse(report["future_stability_guaranteed"])
        self.assertEqual(path.read_bytes(),before)
        saved=json.loads(Path(str(self.output)+".json").read_text(encoding="utf-8"))
        self.assertEqual(saved,report)
        self.assertIn("NOT scientific training approval",Path(str(self.output)+".md").read_text(encoding="utf-8"))

    def test_probe_quota_per_dataset_unprobed_are_header_only(self):
        self.mat("a.mat");self.mat("b.mat");self.mat("c.mat",folder=self.root/"dataset2")
        report=self.scan()
        self.assertTrue(self.row(report,"a.mat")["ready_stable"])
        self.assertFalse(self.row(report,"b.mat")["ready_stable"])
        self.assertEqual(self.row(report,"b.mat")["mat"]["probe_status"],"not_selected")
        self.assertTrue(next(r for r in report["inventory"] if r["dataset"]=="dataset2")["ready_stable"])

    def test_zero_bytes_pending_and_download_suffix_not_ready(self):
        (self.folder/"download.mat").touch();self.mat("active.part")
        report=self.scan()
        self.assertEqual(self.row(report,"download.mat")["type"],"empty")
        self.assertEqual(self.row(report,"download.mat")["status"],"pending_zero_bytes")
        self.assertEqual(self.row(report,"active.part")["type"],"matlab_v5")
        self.assertEqual(self.row(report,"active.part")["status"],"pending_download_name")
        self.assertTrue(all(not r["ready_stable"] for r in report["inventory"]))
        self.assertEqual(report["datasets"][0]["pending"],2)

    def test_actual_magic_over_misnamed_extensions_all_types(self):
        self.mat("numeric.zip");self.zip("archive.mat")
        fixtures={"seven.txt":b"7z\xbc\xaf\x27\x1c"+b"\x00"*100,
                  "rar.mat":b"Rar!\x1a\x07\x01\x00"+b"\x00"*100,
                  "document.bin":b"%PDF-1.7\nsynthetic",
                  "plain.mat":b"plain readable text\n",
                  "hdf5.txt":audit.HDF5_MAGIC+b"\x00"*100,
                  "execute.ps1":b"MZ"+b"\x00"*100,
                  "pickle.mat":b"\x80\x05unsafe pickle"}
        for name,content in fixtures.items():(self.folder/name).write_bytes(content)
        report=self.scan()
        expected={"numeric.zip":"matlab_v5","archive.mat":"zip","seven.txt":"7z","rar.mat":"rar","document.bin":"pdf",
                  "plain.mat":"text","hdf5.txt":"hdf5","execute.ps1":"executable_mz_never_executed","pickle.mat":"pickle_never_loaded"}
        for name,kind in expected.items():self.assertEqual(self.row(report,name)["type"],kind)
        self.assertFalse(report["safety"]["dataset_code_executed"])
        self.assertFalse(report["safety"]["pickle_loaded"])

    def test_zip_crc_all_members_no_extract_and_path_traversal_flagged(self):
        path=self.zip(unsafe=True);before=path.read_bytes()
        with patch.object(zipfile.ZipFile,"extract",side_effect=AssertionError("extract prohibited")),patch.object(zipfile.ZipFile,"extractall",side_effect=AssertionError("extractall prohibited")):
            report=self.scan()
        archive=self.row(report,"data.zip")["archive"]
        self.assertEqual(archive["integrity"],"verified_zip_member_crc")
        self.assertEqual(archive["member_count"],2)
        self.assertEqual(archive["unsafe_member_count"],1)
        self.assertFalse(self.row(report,"data.zip")["ready_stable"])
        self.assertFalse((self.base/"escaped.txt").exists())
        self.assertEqual(path.read_bytes(),before)

    def test_zip_corrupted_crc_and_incomplete_zip_not_falsely_verified(self):
        path=self.zip()
        with zipfile.ZipFile(path) as archive:
            item=archive.getinfo("signal.bin");offset=item.header_offset+30+len(item.filename.encode())+len(item.extra)
        corrupted=bytearray(path.read_bytes());corrupted[offset]^=1;path.write_bytes(corrupted)
        (self.folder/"incomplete.zip").write_bytes(b"PK\x03\x04"+b"\x00"*80)
        report=self.scan()
        self.assertEqual(self.row(report,"data.zip")["status"],"corrupt")
        incomplete=self.row(report,"incomplete.zip")
        self.assertEqual(incomplete["status"],"pending_incomplete_archive")
        self.assertNotIn("verified",incomplete["archive"].get("integrity","").replace("not_verified",""))
        self.assertFalse(incomplete["archive"].get("central_directory_read"))

    def test_tgz_stream_members_no_extract_and_gzip_eof_crc_isize(self):
        self.tgz(unsafe=True,tail=b"tail after tar zero blocks"*100000)
        with patch.object(tarfile.TarFile,"extract",side_effect=AssertionError("extract prohibited")),patch.object(tarfile.TarFile,"extractall",side_effect=AssertionError("extractall prohibited")):
            report=self.scan()
        archive=self.row(report,"data.tar.gz")["archive"]
        self.assertEqual(archive["type"],"tar_gzip")
        self.assertTrue(archive["gzip_crc_isize_verified"])
        self.assertEqual(archive["member_count"],3)
        self.assertEqual(archive["unsafe_member_count"],2)
        self.assertFalse((self.base/"escaped.txt").exists())

    def test_tgz_corrupt_crc_or_isize_after_tar_end_detected(self):
        for name,offset in (("bad_crc.tgz",-8),("bad_size.tgz",-4)):
            path=self.tgz(name,tail=b"extra after tar"*20000)
            value=bytearray(path.read_bytes());value[offset]^=1;path.write_bytes(value)
        report=self.scan()
        for name in ("bad_crc.tgz","bad_size.tgz"):
            row=self.row(report,name);self.assertEqual(row["status"],"corrupt");self.assertFalse(row["archive"]["gzip_crc_isize_verified"])

    def test_regular_gzip_not_tar_still_crc_checked(self):
        (self.folder/"plain.gz").write_bytes(gzip.compress(b"not a tar archive"))
        archive=self.row(self.scan(),"plain.gz")["archive"]
        self.assertEqual(archive["type"],"gzip")
        self.assertTrue(archive["gzip_crc_isize_verified"])

    def test_inventory_only_does_not_claim_archive_crc(self):
        self.zip();self.tgz()
        with patch.object(zipfile.ZipFile,"testzip",side_effect=AssertionError("CRC forbidden inventory-only")):
            report=self.scan(full_integrity=False,mat_probe_per_dataset=0)
        self.assertEqual(self.row(report,"data.zip")["archive"]["integrity"],"metadata_only_no_member_crc")
        self.assertEqual(self.row(report,"data.tar.gz")["archive"]["integrity"],"magic_only_no_gzip_crc")
        self.assertTrue(all(r["sha256"] is None for r in report["inventory"]))

    def test_nonfinite_mat_error_is_per_file_not_whole_scan(self):
        self.mat("bad.mat",np.array([[1.,np.nan],[np.inf,2.]]));self.mat("good.mat",folder=self.root/"dataset2")
        report=self.scan()
        self.assertEqual(self.row(report,"bad.mat")["status"],"corrupt")
        self.assertEqual(report["datasets"][1]["ready_stable_count"],1)
        json.dumps(report,allow_nan=False)

    def test_4d_and_5d_finite_tensor_ready_without_axes_or_eeg_claim(self):
        self.mat("4d.mat",np.arange(240,dtype=np.float32).reshape(2,3,4,10))
        self.mat("5d.mat",np.arange(240,dtype=np.float64).reshape(2,3,2,2,10),folder=self.root/"dataset2")
        report=self.scan()
        for row,shape in zip([self.row(report,"4d.mat"),next(r for r in report["inventory"] if r["dataset"]=="dataset2")],[[2,3,4,10],[2,3,2,2,10]]):
            self.assertTrue(row["ready_stable"])
            self.assertTrue(row["mat"]["locally_finite_numeric_tensor_or_matrix"])
            self.assertEqual(row["mat"]["arrays"][0]["shape"],shape)
            self.assertIn("not automatically EEG",row["mat"]["scientific_signal_identity"])
            self.assertIn("unknown",row["mat"]["arrays"][0]["axis_interpretation"])
        self.assertFalse(report["train_eligibility_assessed"])

    def test_mat_memory_budget_prevents_loadmat_and_unknown_struct_not_loaded(self):
        from scipy import io as scipy_io
        path=self.mat()
        header=[("unknown_axes",(100000,100000),"double")]
        with patch.object(scipy_io,"whosmat",return_value=header),patch.object(scipy_io,"loadmat",side_effect=AssertionError("load prohibited")):
            result=audit.inspect_mat(path,audit.ReadBudget(),True,256*audit.CHUNK)
        self.assertEqual(result["probe_status"],"skipped_memory_budget")
        with patch.object(scipy_io,"whosmat",return_value=[("nested",(1,1),"struct")]),patch.object(scipy_io,"loadmat",side_effect=AssertionError("struct load prohibited")):
            result=audit.inspect_mat(path,audit.ReadBudget(),True,256*audit.CHUNK)
        self.assertIn("nested_struct",result["probe_status"])
        self.assertFalse(result.get("finite_matrix",False))

    def test_finite_io_budget_enforced_not_corruption_not_ready(self):
        self.mat();self.zip()
        report=self.scan(max_bytes=32)
        self.assertLessEqual(report["read_budget"]["bytes_returned_by_python_source_readers"],32)
        self.assertTrue(all(not r["ready_stable"] for r in report["inventory"]))
        self.assertTrue(all(r["status"]=="not_verified_budget" for r in report["inventory"]))
        self.assertEqual(report["datasets"][0]["corrupt"],0)
        self.assertEqual(report["datasets"][0]["changed"],0)
        budget=audit.ReadBudget(0)
        path=self.folder/"large.bin";path.write_bytes(b"X"*(2*audit.CHUNK+7))
        digest=audit.stream_sha256(path,budget)
        self.assertEqual(digest,hashlib.sha256(b"X"*(2*audit.CHUNK+7)).hexdigest())
        self.assertEqual(budget.used,path.stat().st_size)

    def test_changed_stat_before_after_never_ready_or_corrupt(self):
        path=self.mat()
        original=audit._stamp;calls=0
        def changing_stamp(current):
            nonlocal calls
            value=original(current)
            if Path(current)==path:
                calls+=1
                if calls>=3:value={**value,"size":value["size"]+100,"mtime_ns":value["mtime_ns"]+1}
            return value
        with patch.object(audit,"_stamp",side_effect=changing_stamp):report=self.scan()
        row=self.row(report,"a.mat")
        self.assertEqual(row["status"],"changed_inprogress")
        self.assertFalse(row["ready_stable"])
        self.assertTrue(row["changed"])
        self.assertIn("dataset1/a.mat",report["changes"]["changed_files"])
        self.assertEqual(report["datasets"][0]["corrupt"],0)

    def test_actual_download_growth_during_settle_new_missing_and_changes(self):
        growing=self.folder/"growing.mat";growing.touch()
        removed=self.folder/"gone.bin";removed.write_bytes(b"old")
        def downloads_continue(seconds):
            self.assertEqual(seconds,2.)
            savemat(growing,{"unknown_axes":np.arange(40).reshape(4,10)})
            removed.unlink();self.mat("new.mat")
        with patch.object(audit.time,"sleep",side_effect=downloads_continue):report=self.scan(settle_seconds=2.)
        self.assertIn("dataset1/growing.mat",report["changes"]["changed_files"])
        self.assertIn("dataset1/gone.bin",report["changes"]["missing_files"])
        self.assertIn("dataset1/new.mat",report["changes"]["new_files"])
        self.assertEqual(self.row(report,"growing.mat")["status"],"changed_inprogress")
        self.assertEqual(self.row(report,"new.mat")["status"],"pending_new_file")
        self.assertFalse(self.row(report,"new.mat")["ready_stable"])
        self.assertIsNotNone(self.row(report,"new.mat")["stat_before"])
        self.assertIsNotNone(self.row(report,"new.mat")["stat_after"])
        self.assertEqual(report["datasets"][0]["ready_stable_count"],0)

    def test_symlink_and_git_excluded_no_follow_or_execute(self):
        self.mat()
        git=self.folder/".git";git.mkdir();(git/"hooks.ps1").write_text("never execute",encoding="utf-8")
        outside=self.base/"outside";outside.mkdir();self.mat("external.mat",folder=outside)
        link=self.folder/"external_link"
        try:link.symlink_to(outside,target_is_directory=True)
        except OSError:self.skipTest("host does not permit symlinks")
        with patch.object(audit.subprocess,"Popen",side_effect=AssertionError("unexpected execute")):
            report=self.scan()
        self.assertEqual(len(report["inventory"]),1)
        reasons={r["reason"] for r in report["exclusions"]["initial"]}
        self.assertIn("git_metadata",reasons);self.assertIn("symlink_junction_or_reparse_point",reasons)
        self.assertTrue((outside/"external.mat").is_file())

    def test_git_and_mocked_windows_reparse_exclusion_always_covered(self):
        self.mat()
        git=self.folder/".git";git.mkdir();(git/"hooks.ps1").write_text("never execute",encoding="utf-8")
        junction=self.folder/"mock_junction";junction.mkdir();self.mat("hidden.mat",folder=junction)
        original=audit._stamp
        def reparse_stamp(path):
            value=original(path)
            if Path(path)==junction:value={**value,"file_attributes":value["file_attributes"]|0x400}
            return value
        with patch.object(audit,"_stamp",side_effect=reparse_stamp),patch.object(audit.subprocess,"Popen",side_effect=AssertionError("execute prohibited")):
            report=self.scan()
        self.assertEqual(len(report["inventory"]),1)
        excluded={row["reason"] for row in report["exclusions"]["initial"]}
        self.assertIn("git_metadata",excluded)
        self.assertIn("symlink_junction_or_reparse_point",excluded)

    def test_finite_budget_exhausted_after_magic_does_not_become_corrupt_or_ready(self):
        path=self.mat()
        limit=path.stat().st_size+5
        report=self.scan(max_bytes=limit)
        row=self.row(report,"a.mat")
        self.assertEqual(row["type"],"matlab_v5")
        self.assertEqual(row["status"],"not_verified_budget")
        self.assertFalse(row["ready_stable"])
        self.assertIsNone(row["sha256"])
        self.assertEqual(report["datasets"][0]["corrupt"],0)
        self.assertLessEqual(report["read_budget"]["bytes_returned_by_python_source_readers"],limit)

    def test_root_symlink_and_report_under_source_rejected_before_scan(self):
        with patch.object(audit,"enumerate_inventory",side_effect=AssertionError("must not scan")):
            with self.assertRaisesRegex(audit.AuditError,"read-only"):
                audit.audit_local(self.root,self.folder/"report",settle_seconds=0)
        self.assertFalse((self.folder/"report.json").exists())
        link=self.base/"linkedroot"
        try:link.symlink_to(self.root,target_is_directory=True)
        except OSError:return
        with self.assertRaisesRegex(audit.AuditError,"symlink"):
            audit.enumerate_inventory(link)
        outlink=self.base/"output_link"
        try:outlink.symlink_to(self.folder,target_is_directory=True)
        except OSError:return
        with self.assertRaisesRegex(audit.AuditError,"read-only"):
            audit.audit_local(self.root,outlink/"report",settle_seconds=0)

    def test_hdf5_reader_missing_no_install_or_signal_read(self):
        path=self.folder/"hdf5.mat";path.write_bytes(audit.HDF5_MAGIC+b"\x00"*100)
        original=builtins.__import__
        def no_h5py(name,*args,**kwargs):
            if name=="h5py":raise ImportError("not installed")
            return original(name,*args,**kwargs)
        with patch("builtins.__import__",side_effect=no_h5py),patch.object(audit.subprocess,"Popen",side_effect=AssertionError("no installation")):
            report=self.scan()
        row=self.row(report,"hdf5.mat")
        self.assertEqual(row["hdf5"]["status"],"needs_hdf5reader")
        self.assertFalse(row["hdf5"]["signal_loaded"]);self.assertFalse(row["ready_stable"])

    def test_rar_7z_missing_tool_and_finite_budget_not_verified(self):
        for name,magic in (("data.rar",b"Rar!\x1a\x07\x00"),("data.7z",b"7z\xbc\xaf\x27\x1c")):
            (self.folder/name).write_bytes(magic+b"\x00"*100)
        report=self.scan()
        for name in ("data.rar","data.7z"):
            row=self.row(report,name);self.assertEqual(row["archive"]["integrity"],"not_verified");self.assertFalse(row["ready_stable"])
        with patch.object(audit,"_run_7zip",side_effect=AssertionError("unmetered external IO")):
            result=audit.inspect_seven_zip(self.folder/"data.7z","7z",audit.ReadBudget(1024),True,"installed7z.exe",self.base,1)
        self.assertEqual(result["integrity"],"not_verified")
        self.assertIn("cannot meter",result["reason"])

    def test_installed_7zip_only_list_and_test_argv_never_extract(self):
        path=self.folder/"data.7z";path.write_bytes(b"7z\xbc\xaf\x27\x1c")
        process=unittest.mock.Mock();process.wait.return_value=0
        commands=[]
        def popen(command,**kwargs):
            commands.append(command)
            self.assertFalse(kwargs["shell"])
            return process
        with patch.object(audit.subprocess,"Popen",side_effect=popen):
            audit._run_7zip("installed7z.exe","l",path,self.base,1)
            audit._run_7zip("installed7z.exe","t",path,self.base,1)
        self.assertEqual([c[1] for c in commands],["l","t"])
        self.assertTrue(all("--" in c for c in commands))
        for action in ("x","e"):
            with self.assertRaises(audit.AuditError):audit._run_7zip("installed7z.exe",action,path,self.base,1)

    def test_output_refuse_overwrite_before_scan_and_atomic_rollback(self):
        self.output.parent.mkdir(parents=True)
        existing=Path(str(self.output)+".md");existing.write_text("keep",encoding="utf-8")
        with patch.object(audit,"enumerate_inventory",side_effect=AssertionError("scan should not run")):
            with self.assertRaisesRegex(audit.AuditError,"overwrite"):self.scan()
        self.assertEqual(existing.read_text(encoding="utf-8"),"keep")
        existing.unlink()
        original=os.link;calls=[]
        def second_fails(source,target):
            calls.append(target)
            if len(calls)==2:raise OSError("synthetic publication error")
            return original(source,target)
        with patch.object(audit.os,"link",side_effect=second_fails):
            with self.assertRaisesRegex(OSError,"synthetic"):self.scan()
        self.assertFalse(Path(str(self.output)+".json").exists())
        self.assertFalse(Path(str(self.output)+".md").exists())
        self.assertEqual(list(self.output.parent.glob("*.pending")),[])

    def test_byte_budget_direct_reader_and_member_memory_bound(self):
        path=self.folder/"stream.bin";path.write_bytes(b"abcde")
        budget=audit.ReadBudget(4)
        with audit.SourceReader(path,budget) as reader:
            self.assertEqual(reader.read(3),b"abc")
            with self.assertRaises(audit.BudgetExceeded):reader.read(2)
        self.assertEqual(budget.used,3)
        result=audit._archive_base("tar")
        with patch.object(audit,"MEMBER_LIMIT",1):
            audit._retain_member(result,audit._member("a",1));audit._retain_member(result,audit._member("../b",2))
        self.assertEqual(result["member_count"],2);self.assertEqual(result["members_omitted"],1);self.assertEqual(result["unsafe_member_count"],1)

    def test_root_metadata_scripts_inventory_not_counted_as_tenth_dataset_or_run(self):
        (self.root/"download.ps1").write_text("Write-Output 'NEVER EXECUTE'",encoding="utf-8")
        with patch.object(audit.subprocess,"Popen",side_effect=AssertionError("source script must not run")):
            report=self.scan()
        self.assertEqual(report["observed_dataset_count"],9)
        self.assertTrue(report["dataset_count_matches_expected"])
        self.assertEqual(report["root_level_file_count"],1)
        row=report["inventory"][0]
        self.assertEqual(row["dataset"],"_root")
        self.assertFalse(report["datasets"][0]["is_dataset_directory"])
        self.assertFalse(report["safety"]["dataset_code_executed"])

    def test_cli_defaults_and_argument_budget_validation(self):
        args=audit.make_parser().parse_args([])
        self.assertEqual(args.root,Path("D:/SSVEP"));self.assertTrue(args.full_integrity)
        self.assertEqual((args.settle_seconds,args.mat_probe_per_dataset,args.max_bytes,args.memory_budget_mb),(2.,1,0,256))
        for kwargs in ({"memory_budget_mb":257},{"max_bytes":-1},{"settle_seconds":float("nan")},{"mat_probe_per_dataset":-1}):
            with self.assertRaises(audit.AuditError):self.scan(**kwargs)

if __name__=="__main__":unittest.main()
