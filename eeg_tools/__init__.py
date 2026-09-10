"""Shared helpers for the local EEG acquisition tools."""

from .config import ConfigError, load_and_validate_configs

__all__ = ["ConfigError", "load_and_validate_configs"]
