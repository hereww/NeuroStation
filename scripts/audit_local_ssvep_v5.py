"""Read-only local SSVEP inventory; point-in-time, NOT scientific eligibility.

Never extract archives, execute dataset code/Git/hooks or unpickle objects.
Only an installed 7-Zip outside the source root may be invoked with l / t.
Importing this module does not inspect D:/SSVEP or publish a report.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
import time
import uuid
import zipfile

WORKSPACE = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
EXPECTED_DATASET_COUNT = 9
CHUNK = 1024 * 1024
MAX_MEMORY_MB = 256
MEMBER_LIMIT = 20000
INVENTORY_LIMIT = 100000
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
NUMERIC_MAT_CLASSES = {"double":8,"single":4,"int8":1,"uint8":1,"int16":2,
    "uint16":2,"int32":4,"uint32":4,"int64":8,"uint64":8,"logical":1}
DOWNLOAD_SUFFIXES = (".part", ".partial", ".download", ".crdownload", ".tmp", ".aria2")

class AuditError(ValueError):
    """Invocation/report publication error."""

class BudgetExceeded(AuditError):
    """Budget exhausted, not evidence of corruption."""

class ReadBudget:
    """Zero means unlimited streaming IO, never unlimited RAM.

    Counts bytes returned by Python source readers, including repeated reads.
    External 7zip is skipped with a finite budget because its IO is unmetered.
    """
    def __init__(self, maximum=0): self.maximum,self.used=maximum,0
    def allow(self, requested):
        if self.maximum and self.used+requested>self.maximum:
            raise BudgetExceeded("source read byte budget exhausted")

class SourceReader(io.BufferedIOBase):
    def __init__(self, path, budget):
        self.handle=open(path,"rb")
        self.size=os.fstat(self.handle.fileno()).st_size
        self.budget=budget
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.handle.tell()
    def seek(self,offset,whence=os.SEEK_SET):return self.handle.seek(offset,whence)
    def fileno(self):return self.handle.fileno()
    def read(self,size=-1):
        remaining=max(0,self.size-self.tell())
        if size is None or size<0:
            if remaining>CHUNK:raise BudgetExceeded("unbounded source read refused; stream chunks")
            size=remaining
        size=min(size,remaining)
        self.budget.allow(size)
        value=self.handle.read(size);self.budget.used+=len(value)
        return value
    def readinto(self,buffer):
        value=self.read(len(buffer));buffer[:len(value)]=value;return len(value)
    def close(self):
        if not self.closed:self.handle.close()
        super().close()

def _stamp(path):
    value=os.stat(path,follow_symlinks=False)
    return {"size":int(value.st_size),"mtime_ns":int(value.st_mtime_ns),"ctime_ns":int(value.st_ctime_ns),
        "device":int(value.st_dev),"inode":int(value.st_ino),"mode":int(value.st_mode),
        "file_attributes":int(getattr(value,"st_file_attributes",0))}

def _is_link(path,info=None):
    info=info or _stamp(path)
    return (stat.S_ISLNK(info["mode"]) or bool(info["file_attributes"]&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400))
        or (hasattr(os.path,"isjunction") and os.path.isjunction(path)))

def _same(a,b):return a is not None and b is not None and a==b

def _iso(ns):return datetime.fromtimestamp(ns/1e9,timezone.utc).isoformat()

def enumerate_inventory(root):
    """Do not follow symlinks/reparse points, including a linked root."""
    root=Path(os.path.abspath(root))
    if _is_link(root) or not root.is_dir():raise AuditError("root must be an existing real directory, not symlink/junction")
    files,excluded,errors,datasets={},[],[],set()
    pending=[root]
    while pending:
        parent=pending.pop()
        try:
            if _is_link(parent):
                excluded.append({"path":str(parent),"relative_path":parent.relative_to(root).as_posix(),"reason":"directory_became_symlink_junction"})
                continue
            with os.scandir(parent) as entries:children=sorted(entries,key=lambda e:e.name.casefold())
        except OSError as error:
            errors.append({"path":str(parent),"error":str(error)});continue
        for entry in children:
            path=Path(entry.path);relative=path.relative_to(root).as_posix()
            try:
                info=_stamp(path)
                if entry.name.casefold()==".git" or _is_link(path,info):
                    excluded.append({"path":str(path),"relative_path":relative,"reason":"git_metadata" if entry.name.casefold()==".git" else "symlink_junction_or_reparse_point"})
                    continue
                if stat.S_ISDIR(info["mode"]):
                    if parent==root:datasets.add(entry.name)
                    pending.append(path)
                elif stat.S_ISREG(info["mode"]):
                    if len(files)>=INVENTORY_LIMIT:raise BudgetExceeded("inventory limit reached; cannot claim completeness")
                    dataset=relative.split("/",1)[0] if "/" in relative else "_root"
                    datasets.add(dataset)
                    files[relative]={"path":str(path),"relative_path":relative,"dataset":dataset,
                        "filesystem_type":"regular_file","size":info["size"],"mtime":_iso(info["mtime_ns"]),"mtime_ns":info["mtime_ns"],"stat":info}
                else:excluded.append({"path":str(path),"relative_path":relative,"reason":"not_regular_file"})
            except BudgetExceeded:raise
            except OSError as error:errors.append({"path":str(path),"relative_path":relative,"error":str(error)})
    return {"files":files,"datasets":sorted(datasets),"excluded":excluded,"errors":errors}

def identify_magic(path,size,budget):
    if size==0:return {"type":"empty","signature_offset":None,"prefix_hex":"","basis":"zero_bytes"}
    with SourceReader(path,budget) as handle:head=handle.read(min(size,65544))
    kind,offset="unknown_binary",0
    if head.startswith((b"PK\x03\x04",b"PK\x05\x06",b"PK\x07\x08")):kind="zip"
    elif head.startswith(b"\x1f\x8b"):kind="gzip"
    elif head.startswith(b"7z\xbc\xaf\x27\x1c"):kind="7z"
    elif head.startswith((b"Rar!\x1a\x07\x00",b"Rar!\x1a\x07\x01\x00")):kind="rar"
    elif head.startswith(b"%PDF-"):kind="pdf"
    elif head.startswith(b"MATLAB 5.0 MAT-file"):kind="matlab_v5"
    elif head.startswith(b"MATLAB 7.3 MAT-file"):kind="matlab_v7_3_hdf5"
    elif head.startswith(b"MZ"):kind="executable_mz_never_executed"
    elif len(head)>1 and head[0]==0x80 and head[1]<=5:kind="pickle_never_loaded"
    else:
        found=next((p for p in [0]+[2**n for n in range(9,17)] if head[p:p+8]==HDF5_MAGIC),None)
        if found is not None:kind,offset="hdf5",found
        elif len(head)>=512 and head[257:262]==b"ustar":kind="tar"
        else:
            sample=head[:4096]
            try:
                encoding="utf-16" if sample.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig"
                text=sample.decode(encoding,errors="strict")
                if text and "\x00" not in text and sum(c.isprintable() or c in "\r\n\t" for c in text)/len(text)>=.95:kind="text"
            except UnicodeError:pass
    return {"type":kind,"signature_offset":offset,"prefix_hex":head[:32].hex(),"basis":"read_bytes_not_extension","extension":Path(path).suffix.lower()}

def stream_sha256(path,budget):
    digest=hashlib.sha256()
    with SourceReader(path,budget) as handle:
        while block:=handle.read(CHUNK):digest.update(block)
    return digest.hexdigest()

def _member(name,size,kind="file",**extras):
    normalized=name.replace("\\","/");reasons=[]
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:",normalized):reasons.append("absolute_or_drive_path")
    if ".." in PurePosixPath(normalized).parts:reasons.append("parent_traversal")
    if "\x00" in normalized:reasons.append("nul_in_name")
    if kind in ("symlink","hardlink"):reasons.append("archive_link_not_followed")
    return {"name":name,"size":int(size),"type":kind,"unsafe_path":bool(reasons),"safety_flags":reasons,**extras}

def _retain_member(result,member):
    result["member_count"]+=1;result["unsafe_member_count"]+=int(member["unsafe_path"])
    if len(result["members"])<MEMBER_LIMIT:result["members"].append(member)
    else:result["members_omitted"]+=1

def _archive_base(kind):
    return {"type":kind,"member_count":0,"members":[],"members_omitted":0,"unsafe_member_count":0,"extracted":False,"integrity":"not_verified"}

def _zip_preflight(handle,memory_bytes):
    """Bound central-directory allocation before stdlib ZipFile reads it."""
    handle.seek(max(0,handle.size-65557));tail=handle.read(65557);index=tail.rfind(b"PK\x05\x06")
    if index<0 or len(tail)-index<22:raise zipfile.BadZipFile("central directory/end record missing; may be in progress")
    _,disk,cd_disk,on_disk,count,size,offset,comment=struct.unpack("<4s4H2LH",tail[index:index+22])
    if disk or cd_disk or on_disk!=count:raise BudgetExceeded("multivolume ZIP not fully verifiable by this reader")
    if count==65535 or size==0xffffffff or offset==0xffffffff:raise BudgetExceeded("ZIP64 allocation not bounded; not verified")
    if size*5+count*2048>memory_bytes//2:raise BudgetExceeded("ZIP central-directory memory budget exceeded")
    if len(tail)-index<22+comment:raise zipfile.BadZipFile("truncated ZIP end record")
    handle.seek(0)

def inspect_zip(path,budget,full,memory_bytes):
    result=_archive_base("zip")
    try:
        with SourceReader(path,budget) as handle:
            _zip_preflight(handle,memory_bytes)
            with zipfile.ZipFile(handle,"r") as archive:
                for entry in archive.infolist():
                    mode=entry.external_attr>>16
                    kind="symlink" if stat.S_ISLNK(mode) else "directory" if entry.is_dir() else "file"
                    _retain_member(result,_member(entry.filename,entry.file_size,kind,compressed_size=int(entry.compress_size),crc32=f"{entry.CRC:08x}"))
                result["central_directory_read"]=True
                if full:
                    bad=archive.testzip()  # Streams all member CRCs; NEVER extract.
                    if bad is not None:result.update(integrity="corrupt",bad_crc_member=bad)
                    else:result.update(integrity="verified_zip_member_crc",crc_verified=True)
                else:result["integrity"]="metadata_only_no_member_crc"
    except BudgetExceeded as error:result.update(integrity="not_verified",error=str(error),budget_limited=True)
    except zipfile.BadZipFile as error:
        reason=str(error)
        result.update(integrity="corrupt" if "CRC" in reason.upper() else "not_verified_incomplete_archive",error=reason,central_directory_read=result.get("central_directory_read",False))
    except (RuntimeError,NotImplementedError,OSError,EOFError,ValueError) as error:result.update(integrity="not_verified",error=str(error))
    return result

def inspect_gzip(path,budget,full):
    result=_archive_base("gzip")
    if not full:
        result["integrity"]="magic_only_no_gzip_crc";return result
    gzip_error=None
    with SourceReader(path,budget) as reader:
        with gzip.GzipFile(fileobj=reader,mode="rb") as compressed:
            try:
                with tarfile.open(fileobj=compressed,mode="r|") as archive:
                    result["type"]="tar_gzip"
                    for entry in archive:
                        kind="symlink" if entry.issym() else "hardlink" if entry.islnk() else "directory" if entry.isdir() else "file" if entry.isfile() else "special"
                        _retain_member(result,_member(entry.name,entry.size,kind,link_name=entry.linkname or None))
                    result["tar_metadata_iteration_completed"]=True
            except tarfile.TarError as error:result["tar_metadata_error"]=str(error)
            except (gzip.BadGzipFile,EOFError,OSError) as error:gzip_error=error
            # Tar can stop at its zero blocks before the gzip trailer. Drain to
            # EOF regardless: that is necessary for BOTH gzip CRC and ISIZE.
            if gzip_error is None:
                try:
                    while compressed.read(CHUNK):pass
                except (gzip.BadGzipFile,EOFError,OSError) as error:gzip_error=error
    if gzip_error is not None:result.update(integrity="corrupt",error=str(gzip_error),gzip_crc_isize_verified=False)
    else:
        result.update(integrity="verified_gzip_crc_isize",gzip_crc_isize_verified=True)
        if result["type"]=="tar_gzip" and "tar_metadata_error" in result:result["integrity"]="corrupt_tar_metadata"
    return result

def inspect_tar(path,budget,full):
    result=_archive_base("tar")
    if not full:return result
    with SourceReader(path,budget) as reader:
        with tarfile.open(fileobj=reader,mode="r|") as archive:
            for entry in archive:
                kind="symlink" if entry.issym() else "hardlink" if entry.islnk() else "directory" if entry.isdir() else "file" if entry.isfile() else "special"
                _retain_member(result,_member(entry.name,entry.size,kind,link_name=entry.linkname or None))
    result["integrity"]="tar_metadata_streamed_no_payload_checksum"
    return result

def find_seven_zip(root):
    candidates=[shutil.which(name) for name in ("7z","7zz","7za")]
    if os.name=="nt":
        candidates += [str(Path(os.environ.get("ProgramFiles","C:/Program Files"))/"7-Zip/7z.exe"),
                       str(Path(os.environ.get("ProgramFiles(x86)","C:/Program Files (x86)"))/"7-Zip/7z.exe")]
    for value in candidates:
        if not value:continue
        candidate=Path(value).absolute()
        try:
            if not candidate.resolve().is_relative_to(root.resolve()) and candidate.is_file() and not _is_link(candidate):return str(candidate)
        except OSError:pass
    return None

def _run_7zip(executable,action,path,output_dir,timeout):
    if action not in ("l","t"):raise AuditError("7-Zip may only list/test")
    command=[executable,action,"-bd","-bb0","-y"]
    if action=="l":command += ["-slt","-ba"]
    command += ["--",str(path)]
    with tempfile.TemporaryFile(mode="w+b",dir=output_dir) as output:
        options={"stdout":output,"stderr":subprocess.STDOUT,"stdin":subprocess.DEVNULL,"shell":False}
        if os.name=="nt":options["creationflags"]=subprocess.CREATE_NO_WINDOW
        process=subprocess.Popen(command,**options)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()
            return {"exit_code":None,"timeout":True,"command_action":action,"text":""}
        length=output.tell();output.seek(0);text=output.read(2*CHUNK).decode("utf-8",errors="replace")
    return {"exit_code":int(code),"timeout":False,"command_action":action,"text":text,"output_truncated":length>2*CHUNK}

def inspect_seven_zip(path,kind,budget,full,executable,output_dir,timeout):
    result=_archive_base(kind)
    if not executable:
        result["reason"]="installed_7zip_not_found; no installation attempted";return result
    if budget.maximum:
        result["reason"]="finite source IO budget cannot meter external 7zip; skipped";return result
    result["tool"]={"executable":executable,"allowed_actions":["l","t"]}
    listed=_run_7zip(executable,"l",path,output_dir,timeout)
    result["list_exit_code"]=listed["exit_code"]
    if listed["exit_code"]==0 and not listed.get("output_truncated"):
        for block in re.split(r"\r?\n\s*\r?\n",listed["text"]):
            attrs=dict(line.split(" = ",1) for line in block.splitlines() if " = " in line)
            if "Path" in attrs and "Size" in attrs:
                try:_retain_member(result,_member(attrs["Path"],int(attrs["Size"]),"directory" if attrs.get("Folder")=="+" else "file"))
                except ValueError:result["list_parse_warning"]="invalid member size"
    else:result["list_warning"]="list failed/timed out/truncated; membership incomplete"
    if full:
        tested=_run_7zip(executable,"t",path,output_dir,timeout);result["test_exit_code"]=tested["exit_code"]
        if tested["exit_code"]==0:result["integrity"]="verified_7zip_test"
        elif tested["exit_code"] is not None and any(s in tested["text"].lower() for s in ("crc failed","crc error","data error")):
            result.update(integrity="corrupt",error=tested["text"][:2000])
        else:result.update(integrity="not_verified",error=tested["text"][:2000] or "test timeout/unavailable")
    return result

def _numeric_summary(array):
    import numpy as np
    finite=True;lo=None;hi=None;ilo=None;ihi=None
    flat=array.ravel(order="K")
    for start in range(0,flat.size,65536):
        part=flat[start:start+65536];good=np.isfinite(part)
        if not bool(good.all()):finite=False
        part=part[good]
        if not part.size:continue
        a,b=float(part.real.min()),float(part.real.max());lo=a if lo is None else min(lo,a);hi=b if hi is None else max(hi,b)
        if np.iscomplexobj(part):
            a,b=float(part.imag.min()),float(part.imag.max());ilo=a if ilo is None else min(ilo,a);ihi=b if ihi is None else max(ihi,b)
    return {"shape":[int(d) for d in array.shape],"ndim":int(array.ndim),"dtype":str(array.dtype),
        "nbytes":int(array.nbytes),"elements":int(array.size),"finite":bool(finite),"real_range":[lo,hi],
        "imaginary_range":[ilo,ihi] if np.iscomplexobj(array) else None,
        "axis_interpretation":"unknown; no channel/time/subject/stimulus inference"}

def inspect_mat(path,budget,probe,memory_bytes):
    from scipy.io import loadmat,whosmat
    import numpy as np
    result={"header_reader":"scipy.io.whosmat_metadata_only","variables":[],"probe_status":"not_selected",
        "scientific_axes":"unknown","labels_status":"unknown","clean_target_status":"not_defined"}
    with SourceReader(path,budget) as reader:header=whosmat(reader)
    numeric=[];estimated=0
    for name,shape,kind in header:
        shape=[int(d) for d in shape];amount=math.prod(shape)*NUMERIC_MAT_CLASSES[kind] if kind in NUMERIC_MAT_CLASSES else None
        result["variables"].append({"name":name,"shape":shape,"matlab_class":kind,"numeric_bytes_lower_bound":int(amount) if amount is not None else None})
        if amount is not None:
            numeric.append(name)
            # whosmat does not expose complex storage. Reserve twice real size,
            # three live copies and a baseline allowance before loadmat.
            estimated+=int(amount)*2
    result["estimated_numeric_storage_upper_bytes"]=int(estimated)
    if not probe:return result
    if not numeric:
        result["probe_status"]="header_only_nested_struct_cell_or_unknown; not loaded";return result
    reserve=min(64*CHUNK,memory_bytes//2)
    if 3*estimated+reserve>memory_bytes:
        result.update(probe_status="skipped_memory_budget",probe_estimated_peak_bytes=int(3*estimated+reserve));return result
    with SourceReader(path,budget) as reader:
        values=loadmat(reader,variable_names=numeric,verify_compressed_data_integrity=True)
    summaries=[]
    for name in numeric:
        value=values.get(name)
        if not isinstance(value,np.ndarray) or value.dtype.kind not in "biufc":
            result["probe_status"]="unexpected_numeric_representation_not_verified";return result
        if value.nbytes>estimated or sum(row["nbytes"] for row in summaries)+value.nbytes>estimated:
            result["probe_status"]="unexpected_storage_budget_not_verified";return result
        summaries.append({"name":name,**_numeric_summary(value)})
    finite=all(row["finite"] for row in summaries);nonempty=all(row["elements"]>0 for row in summaries)
    real_matrix=any(row["ndim"]>=2 and row["elements"]>0 and row["imaginary_range"] is None for row in summaries)
    result.update(probe_status="finite_numeric_arrays" if finite and nonempty else "nonfinite_or_empty_numeric_arrays",
        arrays=summaries,finite=bool(finite),nonempty=bool(nonempty),finite_matrix=bool(finite and nonempty and real_matrix),
        locally_finite_numeric_tensor_or_matrix=bool(finite and nonempty and real_matrix),
        scientific_signal_identity="unknown; numeric metadata arrays are not automatically EEG",
        probe_scope="named top-level numeric variables only; structs/cells/other variables not loaded")
    return result

def inspect_hdf5(path,budget):
    if budget.maximum:return {"status":"needs_hdf5_reader_budgeted_metadata","signal_loaded":False}
    try:import h5py
    except ImportError:return {"status":"needs_hdf5reader","reader_installed":False,"signal_loaded":False}
    result={"status":"header_only_hdf5_metadata","reader_installed":True,"signal_loaded":False,"arrays":[]};visited=0
    with h5py.File(path,"r") as archive:
        def visit(name,value):
            nonlocal visited
            visited+=1
            if visited>MEMBER_LIMIT:raise BudgetExceeded("HDF5 metadata entry budget exceeded")
            if isinstance(value,h5py.Dataset):result["arrays"].append({"name":name,"shape":[int(d) for d in value.shape],"dtype":str(value.dtype),"axes":"unknown"})
        archive.visititems(visit)  # metadata only; no tensor reads or external-link dereference.
    return result

def _error_record(error,stage):return {"stage":stage,"type":type(error).__name__,"message":str(error)}

def audit_file(entry,options,budget,probe,seven_zip,output_dir):
    result={k:v for k,v in entry.items() if k!="stat"}
    result.update(status="header_only",ready_stable=False,changed=False,errors=[],sha256=None,
        magic={"type":"not_inspected"},type="not_inspected",stat_initial=entry["stat"],stat_before=None,stat_after=None,
        label_status="unknown",clean_target_status="not_defined",train_eligibility_assessed=False)
    path=Path(entry["path"])
    try:
        before=_stamp(path);result["stat_before"]=before
        if _is_link(path,before) or not stat.S_ISREG(before["mode"]):
            result.update(status="excluded",exclusion_reason="became_symlink_junction_or_nonregular");return result
        result["magic"]=identify_magic(path,before["size"],budget);result["type"]=result["magic"]["type"]
        if not _same(before,entry["stat"]):result.update(status="changed_inprogress",changed=True);return result
        if before["size"]==0:result.update(status="pending_zero_bytes",reason="missing_or_download_in_progress_not_ready");return result
        if path.name.lower().endswith(DOWNLOAD_SUFFIXES):result.update(status="pending_download_name",reason="download_hint; no signal probe");return result
        if options["full_integrity"]:result["sha256"]=stream_sha256(path,budget)
        kind=result["type"]
        if kind=="matlab_v5":
            result["probe_attempted"]=bool(probe)
            result["mat"]=inspect_mat(path,budget,probe,options["memory_bytes"])
            if result["mat"].get("finite_matrix"):result.update(status="ready_stable_matrix",ready_stable=True)
            elif result["mat"].get("finite") is False:result.update(status="corrupt",reason="nonfinite_numeric_signal; not a clean-target inference")
        elif kind in ("hdf5","matlab_v7_3_hdf5"):result["hdf5"]=inspect_hdf5(path,budget)
        elif kind=="zip":result["archive"]=inspect_zip(path,budget,options["full_integrity"],options["memory_bytes"])
        elif kind=="gzip":result["archive"]=inspect_gzip(path,budget,options["full_integrity"])
        elif kind=="tar":result["archive"]=inspect_tar(path,budget,options["full_integrity"])
        elif kind in ("rar","7z"):result["archive"]=inspect_seven_zip(path,kind,budget,options["full_integrity"],seven_zip,output_dir,options["archive_timeout_seconds"])
        else:result["verification_scope"]="magic/file identity only; not a finite signal matrix"
        integrity=result.get("archive",{}).get("integrity","")
        if integrity.startswith("corrupt"):result["status"]="corrupt"
        elif integrity=="not_verified_incomplete_archive":result["status"]="pending_incomplete_archive"
    except BudgetExceeded as error:
        result.update(status="not_verified_budget",ready_stable=False);result["errors"].append(_error_record(error,"read_or_memory_budget"))
    except FileNotFoundError as error:
        result.update(status="missing",ready_stable=False);result["errors"].append(_error_record(error,"file_inspection"))
    except Exception as error:  # isolate reader errors; other datasets still audited.
        result.update(status="not_verified_error",ready_stable=False)
        if result["type"]=="matlab_v5" or isinstance(error,gzip.BadGzipFile):result["status"]="corrupt"
        result["errors"].append(_error_record(error,"file_inspection"))
    finally:
        try:
            after=_stamp(path);result["stat_after"]=after
            if result["stat_before"] is not None and not _same(result["stat_before"],after):result.update(status="changed_inprogress",changed=True,ready_stable=False)
        except OSError as error:
            result.update(status="missing",ready_stable=False);result["errors"].append(_error_record(error,"stat_after"))
    return result

def _summaries(records,datasets):
    summaries={name:{"dataset":name,"is_dataset_directory":name!="_root","file_count":0,"ready_stable_count":0,"changed":0,"corrupt":0,"pending":0,
        "header_only":0,"not_verified":0,"signal_bytes":0,"archives_member_count":0,"archive_members_omitted":0,
        "verified_archive_count":0,"train_eligibility_assessed":False} for name in datasets}
    for row in records:
        summary=summaries[row["dataset"]];summary["file_count"]+=1;summary["ready_stable_count"]+=int(row["ready_stable"])
        status=row["status"];summary["changed"]+=int(row["changed"]);summary["corrupt"]+=int(status=="corrupt")
        summary["pending"]+=int(status.startswith("pending") or status in ("missing","changed_inprogress"))
        summary["header_only"]+=int(status=="header_only")
        summary["not_verified"]+=int(status.startswith("not_verified") or row.get("archive",{}).get("integrity","").startswith("not_verified"))
        if row.get("type") in ("matlab_v5","matlab_v7_3_hdf5","hdf5") and row.get("stat_final"):summary["signal_bytes"]+=int(row["stat_final"]["size"])
        archive=row.get("archive",{});summary["archives_member_count"]+=int(archive.get("member_count",0));summary["archive_members_omitted"]+=int(archive.get("members_omitted",0))
        summary["verified_archive_count"]+=int(archive.get("integrity","").startswith("verified") and not row["changed"] and row.get("stat_final") is not None)
    return list(summaries.values())

def markdown_report(report):
    esc=lambda v:str(v).replace("|","\\|").replace("\n"," ").replace("\r"," ")
    lines=["# Local SSVEP read-only audit (point-in-time)","",f"Root: `{esc(report['root'])}`",f"Run: `{report['run_id']}`","",
        "**NOT scientific training approval.** `train_eligibility_assessed=false`; labels, units, axes, subjects and clean targets remain unknown.",
        "`ready_stable_count` only means a locally stable, probed finite real numeric tensor/matrix during this audit; not dataset scientific readiness or future download completion.",
        "`signal_bytes` is final standalone MAT/HDF5 container file size, not decoded signal bytes. Archives are never extracted or counted as ready signals.","",
        "| Dataset | Files | Ready stable | Changed | Corrupt | Pending | Header only | Not verified | Signal bytes | Archive members |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["datasets"]:
        keys=("dataset","file_count","ready_stable_count","changed","corrupt","pending","header_only","not_verified","signal_bytes","archives_member_count")
        lines.append("| "+" | ".join(esc(row[k]) for k in keys)+" |")
    lines += ["",f"Expected top-level dataset count: {EXPECTED_DATASET_COUNT}; observed: {report['observed_dataset_count']}.",
        f"Excluded initial entries: {len(report['exclusions']['initial'])}; excluded final entries: {len(report['exclusions']['final'])}.",
        f"New: {len(report['changes']['new_files'])}; missing: {len(report['changes']['missing_files'])}; changed: {len(report['changes']['changed_files'])}.","","## Limits",""]
    lines += ["- "+esc(line) for line in report["limitations"]];lines += ["","## Files requiring attention",""]
    for row in report["inventory"]:
        if row["status"] not in ("ready_stable_matrix","header_only") or row.get("hdf5") or row.get("archive",{}).get("integrity","").startswith("not_verified"):
            lines.append(f"- `{esc(row['relative_path'])}`: {esc(row['status'])}; magic={esc(row.get('type','unknown'))}; {esc(row.get('reason',''))}")
    return "\n".join(lines)+"\n"

def publish_report(prefix,report):
    targets=[Path(str(prefix)+".json"),Path(str(prefix)+".md")]
    if any(p.exists() or p.is_symlink() for p in targets):raise AuditError("report exists; refuse overwrite")
    staged=[];published=[]
    try:
        for target in targets:
            fd,name=tempfile.mkstemp(prefix=target.name+".",suffix=".pending",dir=target.parent);os.close(fd);staged.append((Path(name),target))
        with staged[0][0].open("w",encoding="utf-8") as handle:
            json.dump(report,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write("\n")
        staged[1][0].write_text(markdown_report(report),encoding="utf-8")
        for temporary,target in staged:os.link(temporary,target);published.append((temporary,target))
    except Exception:
        for temporary,target in published:
            if target.exists() and temporary.samefile(target):target.unlink()
        raise
    finally:
        for temporary,_ in staged:temporary.unlink(missing_ok=True)
    return {"json":str(targets[0]),"markdown":str(targets[1])}

def audit_local(root="D:/SSVEP",report_prefix=None,*,full_integrity=True,settle_seconds=2.,
                mat_probe_per_dataset=1,max_bytes=0,memory_budget_mb=256,archive_timeout_seconds=120.):
    """Reusable read-only scan; only requested JSON/Markdown are published."""
    root=Path(os.path.abspath(root))
    if (type(mat_probe_per_dataset) is not int or mat_probe_per_dataset<0 or type(max_bytes) is not int or max_bytes<0
            or type(memory_budget_mb) is not int or not 1<=memory_budget_mb<=MAX_MEMORY_MB
            or not math.isfinite(settle_seconds) or settle_seconds<0
            or not math.isfinite(archive_timeout_seconds) or archive_timeout_seconds<=0):raise AuditError("invalid settle/probe/read/memory/time budget")
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    prefix=Path(os.path.abspath(report_prefix or WORKSPACE/"analysis_outputs"/f"ssvep_local_v5_{run_id}"))
    # Resolve the parent before mkdir: do not write through a report junction into
    # the source directory. No file under the external source tree is created.
    prefix=prefix.parent.resolve()/prefix.name
    if prefix.is_relative_to(root.resolve()):raise AuditError("reports must be outside the read-only source root")
    if any(Path(str(prefix)+s).exists() or Path(str(prefix)+s).is_symlink() for s in (".json",".md")):
        raise AuditError("report exists; refuse overwrite before scanning")
    initial=enumerate_inventory(root);prefix.parent.mkdir(parents=True,exist_ok=True)
    began=datetime.now(timezone.utc).isoformat()
    if settle_seconds:time.sleep(settle_seconds)
    budget=ReadBudget(max_bytes)
    options={"full_integrity":bool(full_integrity),"memory_bytes":int(memory_budget_mb*CHUNK),"archive_timeout_seconds":float(archive_timeout_seconds)}
    executable=find_seven_zip(root);counts={name:0 for name in initial["datasets"]};records=[]
    for relative,entry in sorted(initial["files"].items()):
        may_probe=counts[entry["dataset"]]<mat_probe_per_dataset
        row=audit_file(entry,options,budget,may_probe,executable,prefix.parent)
        if row.get("probe_attempted"):counts[entry["dataset"]]+=1
        records.append(row)
    final=enumerate_inventory(root)
    new=sorted(set(final["files"])-set(initial["files"]))
    missing=sorted(set(initial["files"])-set(final["files"]))
    changed=sorted(rel for rel in set(initial["files"])&set(final["files"]) if not _same(initial["files"][rel]["stat"],final["files"][rel]["stat"]))
    changed=sorted(set(changed)|{row["relative_path"] for row in records if row["changed"]})
    for row in records:
        end=final["files"].get(row["relative_path"]);row["stat_final"]=end["stat"] if end else None
        if end is None:row.update(status="missing",ready_stable=False)
        elif row["relative_path"] in changed or not _same(row["stat_after"],end["stat"]):row.update(status="changed_inprogress",changed=True,ready_stable=False)
        row["integrity_result_usable_for_stable_file"]=bool(end and not row["changed"] and _same(row["stat_initial"],row["stat_before"]) and _same(row["stat_before"],row["stat_after"]) and _same(row["stat_after"],end["stat"]))
    for relative in new:
        entry=final["files"][relative];row={k:v for k,v in entry.items() if k!="stat"}
        row.update(status="pending_new_file",ready_stable=False,changed=True,errors=[],sha256=None,
            stat_initial=None,stat_before=None,stat_after=None,stat_final=entry["stat"],integrity_result_usable_for_stable_file=False,
            train_eligibility_assessed=False,label_status="unknown",clean_target_status="not_defined")
        try:
            row["stat_before"]=_stamp(Path(entry["path"]))
            if _is_link(Path(entry["path"]),row["stat_before"]):raise AuditError("new file became symlink/junction")
            row["magic"]=identify_magic(Path(entry["path"]),row["stat_before"]["size"],budget)
            row["stat_after"]=_stamp(Path(entry["path"]))
        except Exception as error:row["magic"]={"type":"not_inspected"};row["errors"].append(_error_record(error,"new_file_magic"))
        row["type"]=row["magic"]["type"];records.append(row)
    records.sort(key=lambda row:row["relative_path"])
    names=sorted(set(initial["datasets"])|set(final["datasets"]))
    report={"schema_version":SCHEMA_VERSION,"audit_type":"read_only_local_ssvep_v5_point_in_time",
        "run_id":run_id,"root":str(root),"started_at_utc":began,"finished_at_utc":datetime.now(timezone.utc).isoformat(),
        "point_in_time_only":True,"future_stability_guaranteed":False,"train_eligibility_assessed":False,
        "labels_status":"unknown; external metadata/scientific audit required later","raw_signal_is_clean_target":False,
        "expected_dataset_count":EXPECTED_DATASET_COUNT,"observed_dataset_count":len([n for n in names if n!="_root"]),
        "dataset_count_matches_expected":len([n for n in names if n!="_root"])==EXPECTED_DATASET_COUNT,
        "root_level_file_count":sum(row["dataset"]=="_root" for row in records),
        "options":{"full_integrity":bool(full_integrity),"settle_seconds":float(settle_seconds),"mat_probe_per_dataset":mat_probe_per_dataset,
            "max_bytes":max_bytes,"memory_budget_mb":memory_budget_mb,"archive_timeout_seconds":float(archive_timeout_seconds)},
        "read_budget":{"bytes_returned_by_python_source_readers":int(budget.used),"maximum":max_bytes,
            "definition":"raw input reads; repeated reads included, decompressed bytes not counted twice; external 7zip skipped when finite"},
        "safety":{"source_write_performed":False,"archives_extracted":False,"dataset_code_executed":False,"pickle_loaded":False,"git_or_hooks_executed":False,"installed_7zip":executable},
        "changes":{"new_files":new,"missing_files":missing,"changed_files":changed},
        "exclusions":{"initial":initial["excluded"],"final":final["excluded"]},
        "enumeration_errors":{"initial":initial["errors"],"final":final["errors"]},
        "inventory_complete":not initial["errors"] and not final["errors"],
        "datasets":_summaries(records,names),"inventory":records,
        "limitations":[
            "Point-in-time observations only; downloads can resume/change immediately afterwards. Initial, per-file before/after and final stat comparisons are not a filesystem snapshot or lock.",
            "Ready only means a locally stable probed finite real numeric tensor/matrix. Header-only/unprobed data, archives and zero/new/missing/changed files are never scientifically ready.",
            "No subject/channel/time/stimulus axes or units inferred; no raw signal defined as a clean target. No train/test/model/holdout analysis performed.",
            "SHA256 is streamed. MATLAB v5 uses whosmat first; only budgeted top-level numeric variables are loaded. Nested structs/cells/sparse/unknown representations are not probed.",
            "256 MiB is the maximum configured working-data/reader allocation budget, conservatively enforced for numeric loads and ZIP metadata, not an OS-level hard RSS cap for Python/SciPy or external tools.",
            "HDF5/v7.3 signal tensors are never loaded. Absent h5py is needs_hdf5reader; optional metadata does not validate signal finiteness or axes.",
            f"Archive traversal and links are flagged only, never extracted. Gzip drains to EOF for CRC/ISIZE. ZIP64/multivolume/encrypted/unsupported data may remain not_verified. At most {MEMBER_LIMIT} member records per archive retained; total count/omissions reported.",
            "ZIP CRC, gzip CRC/ISIZE or 7zip success certifies that container check only, not finite EEG or training eligibility."]}
    report["report_paths"]={"json":str(prefix)+".json","markdown":str(prefix)+".md"}
    publish_report(prefix,report)
    return report

def make_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("D:/SSVEP"));parser.add_argument("--report-prefix",type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--full-integrity",dest="full_integrity",action="store_true",default=True)
    mode.add_argument("--inventory-only",dest="full_integrity",action="store_false")
    parser.add_argument("--settle-seconds",type=float,default=2.)
    parser.add_argument("--mat-probe-per-dataset",type=int,default=1)
    parser.add_argument("--max-bytes",type=int,default=0)
    parser.add_argument("--memory-budget-mb",type=int,default=256)
    parser.add_argument("--archive-timeout-seconds",type=float,default=120.)
    return parser

def main(argv=None):
    args=make_parser().parse_args(argv)
    try:
        report=audit_local(**vars(args))
        print(json.dumps({"status":"point_in_time_audit_completed_not_training_approval","report_paths":report["report_paths"],
            "datasets":report["datasets"],"changes":report["changes"],"inventory_complete":report["inventory_complete"],
            "train_eligibility_assessed":False},ensure_ascii=True,allow_nan=False,indent=2));return 0
    except (AuditError,OSError,ValueError) as error:
        print(json.dumps({"status":"error","error":str(error)},ensure_ascii=True,allow_nan=False));return 2

if __name__=="__main__":raise SystemExit(main())
