"""Configuration loading and validation for EEG acquisition sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an acquisition configuration is incomplete or inconsistent."""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Invalid JSON in {path} (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a JSON object: {path}")
    return value


def _required_mapping(config: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be an object in {source}")
    return value


def _positive_number(value: Any, key: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive number in {source}")
    return float(value)


def _nonnegative_number(value: Any, key: str, source: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"'{key}' must be a non-negative number in {source}")
    return float(value)


def _positive_integer(value: Any, key: str, source: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer in {source}")
    return value


def _nonempty_string(value: Any, key: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string in {source}")
    return value


def validate_stimulus_config(config: dict[str, Any], source: Path) -> list[str]:
    warnings: list[str] = []
    display = _required_mapping(config, "display", source)
    layout = _required_mapping(config, "layout", source)
    protocol = _required_mapping(config, "trial_protocol", source)
    hardware = _required_mapping(config, "hardware", source)

    refresh_rate = _positive_integer(
        display.get("refresh_rate_hz"), "display.refresh_rate_hz", source
    )
    frequencies = config.get("frequencies_hz")
    if not isinstance(frequencies, list) or not frequencies:
        raise ConfigError(f"'frequencies_hz' must be a non-empty array in {source}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in frequencies
    ):
        raise ConfigError(f"Every frequency must be a positive integer in {source}")
    if len(set(frequencies)) != len(frequencies):
        raise ConfigError(f"'frequencies_hz' contains duplicate values in {source}")
    for frequency in frequencies:
        if refresh_rate % frequency != 0:
            raise ConfigError(
                f"Frequency {frequency} Hz does not divide the configured {refresh_rate} Hz refresh rate"
            )

    target_count = _positive_integer(
        layout.get("target_count"), "layout.target_count", source
    )
    positions = layout.get("positions")
    if not isinstance(positions, list) or len(positions) != target_count:
        raise ConfigError(
            f"'layout.positions' must contain exactly {target_count} entries in {source}"
        )
    supported_positions = {"top_left", "top_right", "bottom_left", "bottom_right"}
    if any(not isinstance(position, str) for position in positions) or (
        len(set(positions)) != len(positions)
        or any(position not in supported_positions for position in positions)
    ):
        raise ConfigError(
            "'layout.positions' must contain unique values chosen from "
            f"{sorted(supported_positions)} in {source}"
        )
    for color_key in ("background_rgb", "target_rgb"):
        color = layout.get(color_key)
        if (
            not isinstance(color, list)
            or len(color) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255 for value in color)
        ):
            raise ConfigError(f"'layout.{color_key}' must be three integers from 0 to 255 in {source}")
    size_fraction = _positive_number(
        layout.get("target_size_fraction_of_short_side"),
        "layout.target_size_fraction_of_short_side",
        source,
    )
    if size_fraction > 0.5:
        warnings.append("Target size is over half of the display's short side; check the layout.")

    for key in ("pre_session_rest_s", "pre_trial_rest_s", "post_trial_rest_s"):
        _nonnegative_number(protocol.get(key), f"trial_protocol.{key}", source)
    _positive_number(protocol.get("stimulus_s"), "trial_protocol.stimulus_s", source)
    _positive_integer(
        protocol.get("repetitions_per_frequency"),
        "trial_protocol.repetitions_per_frequency",
        source,
    )
    random_seed = protocol.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ConfigError(f"'trial_protocol.random_seed' must be an integer in {source}")

    _nonempty_string(hardware.get("board"), "hardware.board", source)
    _nonempty_string(hardware.get("serial_port"), "hardware.serial_port", source)
    _positive_integer(hardware.get("channels"), "hardware.channels", source)
    _positive_integer(
        hardware.get("sampling_rate_hz"), "hardware.sampling_rate_hz", source
    )

    if str(config.get("status", "")).startswith("draft"):
        warnings.append("The stimulus configuration is marked as a draft.")
    return warnings


def validate_channel_config(config: dict[str, Any], source: Path) -> list[str]:
    warnings: list[str] = []
    _nonempty_string(config.get("board"), "board", source)
    _nonempty_string(config.get("serial_port"), "serial_port", source)
    _positive_integer(config.get("sampling_rate_hz"), "sampling_rate_hz", source)
    channels = config.get("channels")
    order = config.get("channel_order")
    if not isinstance(channels, list) or not channels:
        raise ConfigError(f"'channels' must be a non-empty array in {source}")
    if not isinstance(order, list) or len(order) != len(channels):
        raise ConfigError(f"'channel_order' must match the channel count in {source}")

    names: list[str] = []
    indexes: list[int] = []
    unset_positions: list[str] = []
    for offset, channel in enumerate(channels):
        if not isinstance(channel, dict):
            raise ConfigError(f"Channel entry {offset + 1} must be an object in {source}")
        name = channel.get("name")
        index = channel.get("gui_index")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"Channel entry {offset + 1} has no valid name in {source}")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ConfigError(f"Channel '{name}' has no valid gui_index in {source}")
        names.append(name)
        indexes.append(index)
        if not channel.get("electrode_position"):
            unset_positions.append(name)

    if len(set(names)) != len(names) or len(set(indexes)) != len(indexes):
        raise ConfigError(f"Channel names and gui_index values must be unique in {source}")
    if order != names:
        raise ConfigError(f"'channel_order' must match the order of 'channels' in {source}")
    if sorted(indexes) != list(range(len(channels))):
        raise ConfigError(f"Channel gui_index values must be contiguous from 0 in {source}")
    if unset_positions:
        warnings.append(
            "Electrode positions are not filled in for: " + ", ".join(unset_positions)
        )
    if "DRAFT" in str(config.get("config_version", "")).upper():
        warnings.append("The channel configuration is marked as a draft.")
    return warnings


def load_and_validate_configs(
    stimulus_path: Path, channel_path: Path
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    stimulus_config = _read_json_object(stimulus_path)
    channel_config = _read_json_object(channel_path)
    warnings = validate_stimulus_config(stimulus_config, stimulus_path)
    warnings.extend(validate_channel_config(channel_config, channel_path))
    stimulus_hardware = stimulus_config["hardware"]
    for stimulus_key, channel_key in (
        ("board", "board"),
        ("channels", None),
        ("sampling_rate_hz", "sampling_rate_hz"),
    ):
        stimulus_value = stimulus_hardware[stimulus_key]
        channel_value = (
            len(channel_config["channels"])
            if channel_key is None
            else channel_config[channel_key]
        )
        if stimulus_value != channel_value:
            raise ConfigError(
                f"Hardware mismatch: stimulus {stimulus_key}={stimulus_value!r}, "
                f"channel configuration value={channel_value!r}"
            )
    return stimulus_config, channel_config, warnings
