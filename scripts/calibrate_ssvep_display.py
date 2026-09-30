"""Run a short GPU/VSync display calibration without opening an EEG session."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_tools.workstation.acquisition_worker import (  # noqa: E402
    _create_stimulus_window,
    _frame_timing_summary,
    _wait_for_frame_swap,
)
from eeg_tools.workstation.ssvep import SSVEPProtocol  # noqa: E402
from neurostation_display import (  # noqa: E402
    center_divider_geometry,
    eye_half_geometry,
    ssvep_frame_is_lit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eye-side", choices=("left", "right"), required=True)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    return parser


def _verify_rendered_frame(window, eye_side: str, lit: bool) -> dict[str, object]:
    width = max(1, round(window.width() * window.devicePixelRatio()))
    height = max(1, round(window.height() * window.devicePixelRatio()))
    active = eye_half_geometry(0, 0, width, height, eye_side)
    inactive_side = "right" if eye_side == "left" else "left"
    inactive = eye_half_geometry(0, 0, width, height, inactive_side)
    divider = center_divider_geometry(0, 0, width, height)
    expected_active = (255, 255, 255) if lit else (0, 0, 0)
    window.makeCurrent()
    gl = window.context().extraFunctions()
    gl.glReadBuffer(0x0404)  # GL_FRONT: inspect the just-swapped frame.
    try:
        frame = bytearray(width * height * 4)
        gl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, frame)
        pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 4)[::-1, :, :3]
        for region, expected, label in (
            (active, expected_active, "active half"),
            (inactive, (0, 0, 0), "inactive half"),
            (divider, (255, 255, 255), "center divider"),
        ):
            rendered = pixels[
                region["y"]:region["y"] + region["height"],
                region["x"]:region["x"] + region["width"],
            ]
            mismatch = np.argwhere(np.any(rendered != expected, axis=2))
            if mismatch.size:
                y, x = mismatch[0]
                actual = tuple(int(value) for value in rendered[y, x])
                raise RuntimeError(
                    f"OpenGL framebuffer {label} pixel "
                    f"({region['x'] + int(x)},{region['y'] + int(y)}) "
                    f"was {actual}, expected {expected}"
                )
    finally:
        gl.glReadBuffer(0x0405)  # GL_BACK: restore QOpenGLWindow's default.
        window.doneCurrent()
    return {
        "status": "pass",
        "width": width,
        "height": height,
        "pixel_checks": width * height,
    }


def _render_checks(application, window, eye_side: str) -> dict[str, object]:
    checks: dict[str, object] = {}
    for lit, name in ((True, "lit"), (False, "dark")):
        swap_count = window.frame_swapped_count
        generation = window.show_target(0, lit)
        _wait_for_frame_swap(
            application,
            window,
            swap_count,
            timeout_s=2.0,
            minimum_generation=generation,
        )
        checks[name] = _verify_rendered_frame(window, eye_side, lit)
    return checks


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.duration_seconds < 1.0:
        raise SystemExit("duration must be at least 1 second per frequency")

    protocol = SSVEPProtocol.load(ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json")
    application, window, mapping = _create_stimulus_window(arguments.eye_side)
    screen = mapping[arguments.eye_side]["screen"]
    screen_refresh_hz = float(screen.refreshRate())
    if screen_refresh_hz <= 0 or abs(screen_refresh_hz - protocol.refresh_rate_hz) > 1.0:
        window.close()
        raise SystemExit(
            f"protocol requires about {protocol.refresh_rate_hz} Hz; "
            f"selected display reports {screen_refresh_hz:g} Hz"
        )

    frame_rows: list[dict[str, object]] = []
    frequency_results: list[dict[str, object]] = []
    try:
        warmup_frames = protocol.refresh_rate_hz
        for warmup_index in range(warmup_frames):
            swap_count = window.frame_swapped_count
            generation = window.show_target(
                0,
                ssvep_frame_is_lit(warmup_index, 10, protocol.refresh_rate_hz),
            )
            _wait_for_frame_swap(
                application,
                window,
                swap_count,
                timeout_s=2.0,
                minimum_generation=generation,
            )
        render_check = _render_checks(application, window, arguments.eye_side)

        for target_index, (target_id, frequency_hz) in enumerate(protocol.targets):
            frame_total = max(1, round(arguments.duration_seconds * protocol.refresh_rate_hz))
            frame_index = 0
            previous_swap: float | None = None
            first_swap: float | None = None
            start_row = len(frame_rows)
            while frame_index < frame_total:
                swap_count = window.frame_swapped_count
                lit = ssvep_frame_is_lit(frame_index, frequency_hz, protocol.refresh_rate_hz)
                generation = window.show_target(target_index, lit)
                actual = _wait_for_frame_swap(
                    application,
                    window,
                    swap_count,
                    timeout_s=2.0,
                    minimum_generation=generation,
                )
                if first_swap is None:
                    first_swap = actual
                interval = None if previous_swap is None else actual - previous_swap
                frame_step = 1 if interval is None else max(1, round(interval * protocol.refresh_rate_hz))
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "frame_interval_ms": "" if interval is None else interval * 1000,
                        "dropped_since_previous": max(0, frame_step - 1),
                    }
                )
                previous_swap = actual
                frame_index += frame_step

            rows = frame_rows[start_row:]
            active_duration_deadline = (first_swap or time.perf_counter()) + arguments.duration_seconds
            while time.perf_counter() < active_duration_deadline:
                application.processEvents()
                time.sleep(min(0.002, active_duration_deadline - time.perf_counter()))
            swap_count = window.frame_swapped_count
            generation = window.show_target(target_index, False)
            off_swap = _wait_for_frame_swap(
                application,
                window,
                swap_count,
                timeout_s=2.0,
                minimum_generation=generation,
            )
            summary = _frame_timing_summary(
                rows, expected_refresh_rate_hz=screen_refresh_hz
            )
            observed_duration_s = 0.0 if first_swap is None else off_swap - first_swap
            duration_error_ms = (
                observed_duration_s - arguments.duration_seconds
            ) * 1000
            duration_ok = abs(duration_error_ms) <= 2000.0 / screen_refresh_hz
            summary.update(
                {
                    "target_id": target_id,
                    "frequency_hz": frequency_hz,
                    "requested_duration_s": arguments.duration_seconds,
                    "observed_duration_s": observed_duration_s,
                    "duration_error_ms": duration_error_ms,
                    "status": (
                        "pass"
                        if summary["cadence_status"] == "pass" and duration_ok
                        else "review"
                    ),
                }
            )
            frequency_results.append(summary)

        output_path = arguments.output or (
            Path(tempfile.gettempdir())
            / f"ssvep_display_calibration_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        result = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "eye_side": arguments.eye_side,
            "screen": {
                "name": str(screen.name()),
                "refresh_rate_hz": screen_refresh_hz,
                "geometry": {
                    "x": int(screen.geometry().x()),
                    "y": int(screen.geometry().y()),
                    "width": int(screen.geometry().width()),
                    "height": int(screen.geometry().height()),
                },
            },
            "protocol_refresh_rate_hz": protocol.refresh_rate_hz,
            "graphics": window.rendering_metadata,
            "render_check_source": "full OpenGL front-buffer pixel readback",
            "timing_source": "Qt frameSwapped callback; not a hardware presentation timestamp",
            "render_check": render_check,
            "frequencies": frequency_results,
            "limitations": (
                "Qt frameSwapped timestamps may include event-loop jitter and cannot verify "
                "panel light output or optical latency; use a photodiode or high-speed camera "
                "for optical validation."
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(output_path), **result}, ensure_ascii=False, indent=2))
        return 0 if all(item["status"] == "pass" for item in frequency_results) else 2
    finally:
        window.close()
        application.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
