"""Read-only evidence snapshots and all-or-nothing derived bundle publication.

Hashes detect changed local evidence, not authenticity against a malicious author.
Only a private sibling staging directory is written; source inputs are never edited.
"""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import tempfile

from eeg_tools.runtime_bindings import sha256


def check_cancelled(cancelled=None):
    if cancelled and cancelled():
        raise InterruptedError('quality.cancelled')


def capture_snapshot(root, names, *, cancelled=None):
    root = Path(root)
    result = {}
    for name in sorted(names):
        check_cancelled(cancelled)
        result[name] = sha256(root / name)
    return result


def snapshot_matches(root, expected, *, names=None, cancelled=None):
    current_names = names() if callable(names) else names
    if current_names is not None and set(current_names) != set(expected):
        return False
    for name, digest in expected.items():
        check_cancelled(cancelled)
        try:
            if sha256(Path(root) / name) != digest:
                return False
        except OSError:
            return False
    return not callable(names) or set(names()) == set(expected)


def read_snapshot(root, name, expected):
    try:
        data = (Path(root) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() == expected[name]:
            return data
    except (OSError, KeyError):
        pass
    raise ValueError('quality.error.source_changed: ' + name)


def snapshot_lines(root, name, expected, *, cancelled=None):
    """Hash the bytes actually consumed, not a second read of a changing file."""
    digest = hashlib.sha256()
    with (Path(root) / name).open('rb') as handle:
        for line in handle:
            check_cancelled(cancelled)
            digest.update(line)
            yield line.decode('utf-8')
    if digest.hexdigest() != expected.get(name):
        raise ValueError('quality.error.source_changed: ' + name)


def publish_bundle(output, files, *, cancelled=None, validate=None):
    """Flush every file before publishing a fresh directory by sibling rename.

    A process crash can leave an explicitly temporary sibling, never a successful
    final bundle. This is not a claim of power-loss durability on every filesystem.
    """
    output = Path(output).resolve()
    check_cancelled(cancelled)
    if output.exists():
        raise FileExistsError(str(output))
    if any(Path(name).name != name or name in ('', '.', '..') for name in files):
        raise ValueError('bundle entries must be plain file names')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', suffix='.tmp', dir=output.parent)).resolve()
    if stage.parent != output.parent:
        raise ValueError('staging directory escaped output parent')
    try:
        for name, data in files.items():
            check_cancelled(cancelled)
            with (stage / name).open('xb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        check_cancelled(cancelled)
        if validate:
            validate()
        check_cancelled(cancelled)
        # Windows rename refuses an existing target; never replace a delivered report.
        if output.exists():
            raise FileExistsError(str(output))
        try:
            stage.rename(output)
        except OSError as exc:
            # Another publisher may win after exists(). POSIX reports ENOTEMPTY
            # for its completed directory, Windows reports EEXIST. Expose the
            # same conflict without retrying or overwriting either result.
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise FileExistsError(errno.EEXIST, 'bundle already published', str(output)) from exc
            raise
    finally:
        # Remove only this invocation's exact staging files, never user/source trees.
        if stage.is_dir():
            for name in files:
                (stage / name).unlink(missing_ok=True)
            stage.rmdir()
    return output
