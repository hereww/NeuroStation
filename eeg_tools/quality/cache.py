"""Immutable cache revisions, content verification and an OS-released writer lock."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import uuid
import numpy as np
import scipy
from eeg_tools.runtime_bindings import sha256
from eeg_tools.quality.contracts import identity


@contextmanager
def writer_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+b')
    locked=False
    try:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=True
        yield
    except OSError as exc:
        if not locked:raise ValueError('cache writer is already active') from exc
        raise
    finally:
        if locked:
            handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_UN)
        handle.close()


def implementation_identity():
    root=Path(__file__).parents[1]
    names=['denoise.py','offline_analysis.py','quality/cache.py','quality/statistics.py','quality/artifacts.py']
    return {'sources':{n:sha256(root/n) for n in names},'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__}


def verified_revision(path, key):
    try:
        metadata=json.loads((path/'complete_v2.json').read_text(encoding='utf-8'))
        if metadata['cache_key']!=key or metadata['schema_version']!=2 or not metadata['outputs']:return False
        for name,digest in metadata['outputs'].items():
            if Path(name).name!=name or (path/name).is_symlink() or sha256(path/name)!=digest:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def run_revision(base, key, producer):
    base=Path(base);base.parent.mkdir(parents=True,exist_ok=True)
    with writer_lock(base.with_name(base.name+'.lock')):
        candidates=[base,*sorted(base.parent.glob(base.name+'__*'))]
        for candidate in candidates:
            if candidate.is_dir() and verified_revision(candidate,key):
                report=json.loads((candidate/'denoising_metrics.json').read_text(encoding='utf-8'))
                return {**report,'status':'up_to_date','output_dir':str(candidate),'cache_key':key}
        final=base if not base.exists() else base.with_name(base.name+'__'+key[:12]+'_'+uuid.uuid4().hex[:8])
        stage=base.with_name('.'+base.name+'.working-'+uuid.uuid4().hex)
        stage.mkdir()
        # Failed stages are deliberately retained, never mistaken for completed cache entries.
        result=producer(stage)
        for path in stage.glob('*.json'):
            value=path.read_text(encoding='utf-8')
            value=value.replace(json.dumps(str(stage))[1:-1],json.dumps(str(final))[1:-1])
            path.write_text(value,encoding='utf-8')
        # The legacy manifest is regenerated after fixing output paths.
        from eeg_tools.denoise import _manifest
        prep=json.loads((stage/'preprocessing.json').read_text(encoding='utf-8'))
        files=[p for p in sorted(stage.iterdir()) if p.is_file() and p.name!='derived_manifest.csv']
        _manifest(stage/'derived_manifest.csv',str(prep['session_id']),prep['input_hashes'],files)
        outputs={p.name:sha256(p) for p in sorted(stage.iterdir()) if p.is_file()}
        for name in outputs:
            with (stage/name).open('r+b') as f:os.fsync(f.fileno())
        with (stage/'complete_v2.json').open('x',encoding='utf-8') as f:
            json.dump({'schema_version':2,'cache_key':key,'outputs':outputs},f,sort_keys=True)
            f.flush();os.fsync(f.fileno())
        os.replace(stage,final)
        return {**result,'output_dir':str(final),'cache_key':key}
