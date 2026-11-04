"""Canonical identities and immutable block metadata. Intervals are [start, stop)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math

SCHEMA_VERSION = 2
IMPLEMENTATION_VERSION = "nsx-quality-1"


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class BlockMetadata:
    session_id: str
    stream_id: str
    generation: int
    start: int
    stop: int
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    unit: str = "uV"
    timestamp_source: str = "unknown"
    gaps: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.session_id or not self.stream_id or self.generation < 0:
            raise ValueError("invalid stream identity")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("invalid half-open interval")
        if not math.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("invalid sampling rate")
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("invalid channel order")

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "interval": "zero_based_half_open", **asdict(self)}
