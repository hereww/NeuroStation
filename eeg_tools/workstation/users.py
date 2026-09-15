"""Local user profiles and recoverable user/session trash handling."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any
from neurostation_contract import GENDERS, MEDICAL_OPTIONS, UserProfile
from .privacy import write_public_user_snapshot
from .secure_storage import protect, unprotect, SecureStorageError


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class UserRegistry:
    """A small JSON-backed registry with a separate recoverable trash state."""

    def __init__(self, dataset_root: Path | None = None):
        self.root = Path(dataset_root).expanduser().resolve() if dataset_root else None
        self.users_dir = self.root / "Users" if self.root else None
        self.registry_path = self.users_dir / "registry.json" if self.users_dir else None
        self.secure_registry_path = self.users_dir / "registry.secure" if self.users_dir else None
        self.trash_root = self.root / "Trash" if self.root else None
        self._active: dict[str, UserProfile] = {}
        self._trash: dict[str, UserProfile] = {}
        self._trashed_sessions: dict[str, list[dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_path is None:
            return
        if not self.registry_path.is_file() and (self.secure_registry_path is None or not self.secure_registry_path.is_file()):
            return
        source_path = self.registry_path
        raw: bytes
        if self.secure_registry_path is not None and self.secure_registry_path.is_file():
            try:
                raw, _storage_mode = unprotect(self.secure_registry_path.read_bytes())
                source_path = self.secure_registry_path
            except (OSError, SecureStorageError):
                raw = self.registry_path.read_bytes()
        else:
            try:
                raw = self.registry_path.read_bytes()
            except OSError:
                return
        try:
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            try:
                backup = self.registry_path.with_suffix(".json.corrupt")
                if backup.exists():
                    index = 2
                    while backup.exists():
                        backup = self.registry_path.with_suffix(f".json.corrupt.{index}")
                        index += 1
                shutil.copy2(source_path, backup)
            except OSError:
                pass
            self._save()
            return
        if not isinstance(value, dict):
            return
        for item in value.get("users", ()):
            if isinstance(item, dict):
                try:
                    profile = UserProfile.from_dict(item)
                except (TypeError, ValueError):
                    continue
                if profile.user_id:
                    self._active[profile.user_id] = profile
        for item in value.get("trash", ()):
            if isinstance(item, dict):
                try:
                    profile = UserProfile.from_dict(item)
                except (TypeError, ValueError):
                    continue
                if profile.user_id:
                    self._trash[profile.user_id] = profile
                    moved = item.get("trashed_sessions", ())
                    if isinstance(moved, list):
                        self._trashed_sessions[profile.user_id] = [
                            dict(entry) for entry in moved if isinstance(entry, dict)
                        ]

    def _save(self) -> None:
        if self.registry_path is None or self.secure_registry_path is None:
            return
        assert self.users_dir is not None
        self.users_dir.mkdir(parents=True, exist_ok=True)
        trash = []
        for profile in self._trash.values():
            item = profile.to_dict()
            moved = self._trashed_sessions.get(profile.user_id, [])
            if moved:
                item["trashed_sessions"] = moved
            trash.append(item)
        value = {
            "schema_version": 1,
            "updated_at": _now(),
            "users": [profile.to_dict() for profile in self._active.values()],
            "trash": trash,
        }
        payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        protected, storage_mode = protect(payload)
        secure_pending = self.secure_registry_path.with_suffix(".secure.pending")
        secure_pending.write_bytes(protected)
        secure_pending.replace(self.secure_registry_path)
        # Keep a readable, privacy-minimized index for diagnostics and migration.
        public = {
            "schema_version": 1,
            "storage": storage_mode,
            "updated_at": value["updated_at"],
            "users": [
                {"user_id": profile.user_id, "status": profile.status, "is_demo": profile.is_demo,
                 "created_at": profile.created_at, "updated_at": profile.updated_at}
                for profile in self._active.values()
            ],
            "trash": [
                {"user_id": profile.user_id, "status": profile.status, "is_demo": profile.is_demo,
                 "created_at": profile.created_at, "updated_at": profile.updated_at, "deleted_at": profile.deleted_at}
                for profile in self._trash.values()
            ],
        }
        pending = self.registry_path.with_suffix(".json.pending")
        pending.write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(self.registry_path)

    def export_public_snapshot(self, path: Path) -> Path:
        """Export a redacted user index suitable for sharing or diagnostics."""
        destination = Path(path).expanduser().resolve()
        write_public_user_snapshot(destination, self.users, self.trash)
        return destination

    @property
    def users(self) -> tuple[UserProfile, ...]:
        return tuple(sorted(self._active.values(), key=lambda item: item.user_id))

    @property
    def trash(self) -> tuple[UserProfile, ...]:
        return tuple(sorted(self._trash.values(), key=lambda item: item.user_id))

    def get(self, user_id: str, *, include_trash: bool = False) -> UserProfile | None:
        profile = self._active.get(str(user_id))
        if profile is None and include_trash:
            profile = self._trash.get(str(user_id))
        return profile

    def next_user_id(self) -> str:
        values = []
        for user_id in (*self._active, *self._trash):
            if user_id.startswith("U") and user_id[1:].isdigit():
                values.append(int(user_id[1:]))
        return f"U{max(values or [0]) + 1:04d}"

    def add(self, profile: UserProfile) -> UserProfile:
        profile = replace(
            profile,
            user_id=profile.user_id.strip() or self.next_user_id(),
            name=profile.name.strip(),
            status="active",
            created_at=profile.created_at or _now(),
            updated_at=_now(),
            deleted_at="",
        )
        profile.validate()
        if profile.user_id in self._active or profile.user_id in self._trash:
            raise ValueError("validation.user_id_duplicate")
        self._active[profile.user_id] = profile
        self._save()
        return profile

    def update(self, original_id: str, profile: UserProfile) -> UserProfile:
        if original_id not in self._active:
            raise ValueError("validation.user_not_found")
        profile = replace(profile, user_id=profile.user_id.strip(), name=profile.name.strip(), updated_at=_now())
        profile.validate()
        if profile.user_id != original_id and (
            profile.user_id in self._active or profile.user_id in self._trash
        ):
            raise ValueError("validation.user_id_duplicate")
        del self._active[original_id]
        self._active[profile.user_id] = profile
        self._save()
        if profile.user_id != original_id:
            self._rewrite_session_user_id(original_id, profile.user_id, profile.name)
        return profile

    def delete(self, user_id: str, *, move_data: bool) -> UserProfile:
        profile = self._active.pop(user_id, None)
        if profile is None:
            raise ValueError("validation.user_not_found")
        moved: list[dict[str, str]] = []
        if move_data and self.root is not None:
            moved = self._move_sessions_to_trash(user_id)
        else:
            self._mark_sessions_orphaned(user_id, profile.name)
        profile = replace(profile, status="trash", deleted_at=_now(), updated_at=_now())
        self._trash[user_id] = profile
        self._trashed_sessions[user_id] = moved
        self._save()
        return profile

    def restore(self, user_id: str) -> UserProfile:
        profile = self._trash.pop(user_id, None)
        if profile is None:
            raise ValueError("validation.user_not_found")
        if user_id in self._active:
            raise ValueError("validation.user_id_duplicate")
        profile = replace(profile, status="active", deleted_at="", updated_at=_now())
        self._active[user_id] = profile
        self._restore_sessions(user_id)
        self._save()
        return profile

    def purge(self, user_id: str) -> None:
        if user_id not in self._trash:
            raise ValueError("validation.user_not_found")
        for item in self._trashed_sessions.pop(user_id, []):
            path = Path(item.get("trash_path", ""))
            if self.trash_root and path.is_dir() and path.resolve().is_relative_to(self.trash_root.resolve()):
                shutil.rmtree(path)
        if self.trash_root:
            parent = self.trash_root / "Datasets" / user_id
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        self._trash.pop(user_id, None)
        self._save()

    def _session_paths(self) -> list[Path]:
        if self.root is None or not self.root.exists():
            return []
        paths = list(self.root.glob("session_*/session.json"))
        paths.extend((self.root / "imports").glob("*/session.json"))
        return paths

    def _read_session(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write_session(self, path: Path, value: dict[str, Any]) -> None:
        pending = path.with_suffix(".json.pending")
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(path)

    def _mark_sessions_orphaned(self, user_id: str, user_name: str) -> None:
        for path in self._session_paths():
            value = self._read_session(path)
            if not value or str(value.get("user_id") or value.get("participant_id") or "") != user_id:
                continue
            value["user_id"] = user_id
            value["user_name"] = str(value.get("user_name") or user_name)
            value["user_link_status"] = "orphaned"
            self._write_session(path, value)

    def _rewrite_session_user_id(self, old_id: str, new_id: str, name: str) -> None:
        for path in self._session_paths():
            value = self._read_session(path)
            if not value or str(value.get("user_id") or value.get("participant_id") or "") != old_id:
                continue
            value["user_id"] = new_id
            value["participant_id"] = new_id
            value["user_name"] = name
            self._write_session(path, value)

    def _move_sessions_to_trash(self, user_id: str) -> list[dict[str, str]]:
        assert self.root is not None and self.trash_root is not None
        destination_root = self.trash_root / "Datasets" / user_id
        destination_root.mkdir(parents=True, exist_ok=True)
        moved: list[dict[str, str]] = []
        for session_path in self._session_paths():
            value = self._read_session(session_path)
            if not value or str(value.get("user_id") or value.get("participant_id") or "") != user_id:
                continue
            source_dir = session_path.parent
            target = destination_root / source_dir.name
            suffix = 2
            while target.exists():
                target = destination_root / f"{source_dir.name}-{suffix}"
                suffix += 1
            shutil.move(str(source_dir), str(target))
            moved.append({"trash_path": str(target), "original_path": str(source_dir)})
        return moved

    def _restore_sessions(self, user_id: str) -> None:
        for item in self._trashed_sessions.get(user_id, []):
            source = Path(item.get("trash_path", ""))
            target = Path(item.get("original_path", ""))
            if not source.is_dir() or not self.root or not source.resolve().is_relative_to(self.trash_root.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            shutil.move(str(source), str(target))
        self._trashed_sessions.pop(user_id, None)
        profile = self._active.get(user_id)
        if profile:
            for path in self._session_paths():
                value = self._read_session(path)
                if not value or str(value.get("user_id") or value.get("participant_id") or "") != user_id:
                    continue
                value["user_link_status"] = "active"
                value["user_name"] = profile.name
                self._write_session(path, value)
