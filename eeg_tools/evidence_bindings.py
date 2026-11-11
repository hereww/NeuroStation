"""Read-only, explicitly pinned historical evidence; never a model-load fallback.

A matching local inventory hash verifies bytes, NOT historical authenticity or
validation success. Missing expected identities are unknown, never auto-pinned.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eeg_tools.runtime_bindings import RootResolver


class EvidenceBindingError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceBindingError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _nonfinite(value):
    raise EvidenceBindingError("nonfinite JSON: " + value)


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


class EvidenceBindings:
    """Pinned file allowlist plus explicit RootResolver root mappings.

    No recursive search, basename guessing, copying, training, or lock repair.
    """
    def __init__(self, manifest: dict):
        if type(manifest) is not dict or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise EvidenceBindingError("unsupported evidence binding schema")
        mappings = manifest.get("root_mappings")
        entries = manifest.get("files")
        if type(mappings) is not list or type(entries) is not list or not entries:
            raise EvidenceBindingError("explicit root_mappings and nonempty files required")
        for mapping in mappings:
            if type(mapping) is not dict:
                raise EvidenceBindingError("root mapping must be an object")
            for key in ("from", "to"):
                value = mapping.get(key)
                if type(value) is not str or not Path(value).is_absolute() or ".." in Path(value).parts:
                    raise EvidenceBindingError("absolute traversal-free mapping roots required")
        try:
            self.resolver = RootResolver(mappings)
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceBindingError("invalid root mappings: " + str(exc)) from exc
        self.entries = []
        identifiers, paths = set(), set()
        for entry in entries:
            if type(entry) is not dict:
                raise EvidenceBindingError("evidence entry must be an object")
            name, path = entry.get("id"), entry.get("recorded_path")
            if type(name) is not str or not name or name in identifiers:
                raise EvidenceBindingError("nonempty unique evidence id required")
            if type(path) is not str or not Path(path).is_absolute() or ".." in Path(path).parts:
                raise EvidenceBindingError("absolute traversal-free recorded path required")
            path_key = os.path.normcase(str(Path(path)))
            if path_key in paths:
                raise EvidenceBindingError("duplicate recorded path")
            if "expected_sha256" not in entry:
                raise EvidenceBindingError("expected_sha256 is required (null means unknown)")
            expected = entry["expected_sha256"]
            if expected is not None and not _digest(expected):
                raise EvidenceBindingError("expected_sha256 must be lowercase SHA256 or null")
            authority = entry.get("identity_authority")
            if authority not in ("frozen_contract", "local_inventory", "unknown"):
                raise EvidenceBindingError("explicit identity authority required")
            if (expected is None) != (authority == "unknown"):
                raise EvidenceBindingError("unknown authority and absent identity must agree")
            identifiers.add(name); paths.add(path_key)
            self.entries.append(dict(entry))

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        manifest = json.loads(raw, object_pairs_hook=_object,
                              parse_constant=_nonfinite)
        result = cls(manifest)
        result.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        return result

    def _candidate(self, recorded, *, check_aliases=True):
        candidate = Path(recorded)
        for source, destination in self.resolver.mappings:
            if candidate.is_relative_to(source):
                candidate = destination / candidate.relative_to(source)
                break
        if check_aliases and any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                                 for p in (candidate, *candidate.parents)):
            raise EvidenceBindingError("symlink/junction paths are not admissible")
        return candidate

    def validate_output(self, output):
        """Refuse writes inside mapped evidence roots or to any declared input."""
        output = Path(output).absolute()
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
               for p in (output, *output.parents)):
            raise EvidenceBindingError("unsafe output path")
        output = output.resolve()
        for _, destination in self.resolver.mappings:
            if output.is_relative_to(destination.resolve()):
                raise EvidenceBindingError("output must be outside mapped evidence roots")
        for entry in self.entries:
            if output in (Path(entry["recorded_path"]).resolve(),
                          self._candidate(entry["recorded_path"], check_aliases=False).resolve()):
                raise EvidenceBindingError("output cannot replace a declared input, even when missing")
        return output

    def audit(self) -> dict:
        rows, resolved = [], set()
        for entry in self.entries:
            row = {key: entry[key] for key in ("id", "recorded_path", "expected_sha256", "identity_authority")}
            row.update(status="blocked", observed_sha256=None, resolved_path=None,
                       historical_conclusions_verified=False)
            try:
                self._candidate(entry["recorded_path"])
                path = self.resolver.resolve(entry["recorded_path"])
                key = os.path.normcase(str(path))
                if key in resolved:
                    raise EvidenceBindingError("multiple entries resolve to the same file")
                resolved.add(key)
                row["resolved_path"] = str(path)
                if entry["expected_sha256"] is None:
                    row["reason"] = "expected_sha256_unknown; obtain an independent pin before verification"
                else:
                    before = path.stat()
                    h = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(chunk)
                    self._candidate(entry["recorded_path"])
                    after = path.stat()
                    row["observed_sha256"] = h.hexdigest()
                    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or self.resolver.resolve(entry["recorded_path"]) != path:
                        row["reason"] = "evidence changed during read"
                    elif h.hexdigest() != entry["expected_sha256"]:
                        row["reason"] = "content_sha256_mismatch"
                    else:
                        row.update(status="verified_bytes", reason=None, byte_count=after.st_size)
            except (OSError, ValueError) as exc:
                row["reason"] = str(exc)
            rows.append(row)
        return {"schema_version": 1,
                "manifest_sha256": getattr(self, "manifest_sha256", None),
                "scope": "declared_file_bytes_only_not_historical_verdicts",
                "historical_conclusions_verified": False,
                "all_declared_bytes_verified": all(row["status"] == "verified_bytes" for row in rows),
                "verified_files": sum(row["status"] == "verified_bytes" for row in rows),
                "blocked_files": sum(row["status"] != "verified_bytes" for row in rows),
                "files": rows}
