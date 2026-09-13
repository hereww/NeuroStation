"""Privacy-aware projections for sharing NeuroStation metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from neurostation_contract import UserProfile


def public_user_profile(profile: UserProfile) -> dict[str, Any]:
    """Return a pseudonymous profile projection without identity or medical details."""
    return {
        "user_id": profile.user_id,
        "status": profile.status,
        "is_demo": profile.is_demo,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "screening_recorded": bool(profile.medical_conditions),
    }


def write_public_user_snapshot(path: Path, users: tuple[UserProfile, ...], trash: tuple[UserProfile, ...]) -> None:
    """Write a shareable user index with names, ages and medical fields removed."""
    payload = {
        "schema_version": 1,
        "users": [public_user_profile(item) for item in users],
        "trash": [public_user_profile(item) for item in trash],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
