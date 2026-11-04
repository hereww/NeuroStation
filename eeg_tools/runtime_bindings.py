"""Explicit, read-only runtime bindings. No environment discovery or hash rewriting."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_config():
    path = os.environ.get("NEUROSTATION_RUNTIME_CONFIG")
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported runtime binding schema")
    return value


def bound_file(name, fallback):
    entry = runtime_config().get("files", {}).get(name)
    if entry is None:
        return Path(fallback)
    p = Path(entry["path"])
    if not p.is_absolute() or not p.is_file() or p.is_symlink() or sha256(p) != entry["sha256"]:
        raise ValueError(f"runtime binding missing or changed: {name}")
    return p


def model_run(fallback):
    value = runtime_config().get("model_run")
    if not value:
        return Path(fallback)
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise ValueError("model run binding is unavailable")
    return path


class RootResolver:
    """Map exact declared roots; never basename-search, guess, or edit manifests."""
    def __init__(self, mappings):
        self.mappings = [(Path(row["from"]), Path(row["to"])) for row in mappings]
        for source, dest in self.mappings:
            if not source.is_absolute() or not dest.is_absolute():
                raise ValueError("root mappings must be absolute")
        for i, (source, _) in enumerate(self.mappings):
            for other, _ in self.mappings[i + 1:]:
                if source.is_relative_to(other) or other.is_relative_to(source):
                    raise ValueError("ambiguous overlapping root mappings")

    def resolve(self, recorded):
        source = Path(recorded)
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("unsafe recorded path")
        matches = [(a, b) for a, b in self.mappings if source.is_relative_to(a)]
        candidate = source
        if matches:
            a, b = matches[0]
            candidate = b / source.relative_to(a)
            if not candidate.resolve().is_relative_to(b.resolve()):
                raise ValueError("mapped path escapes root")
        if candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents):
            raise ValueError("symlink paths are not admissible")
        if not candidate.is_file():
            raise ValueError(f"missing bound file: {candidate}")
        return candidate.resolve()

    def verify(self, hashes):
        resolved = {}
        for recorded, expected in hashes.items():
            p = self.resolve(recorded)
            if p in resolved or sha256(p) != expected:
                raise ValueError(f"duplicate binding or content mismatch: {recorded}")
            resolved[p] = expected
        return resolved
