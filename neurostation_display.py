"""Display geometry and frame-accurate SSVEP helpers."""

from __future__ import annotations

from typing import Any


CENTER_DIVIDER_WIDTH = 6


def ssvep_frame_is_lit(frame_index: int, frequency_hz: int, refresh_rate_hz: int) -> bool:
    """Balance half-cycle edges across frames for an exact average duty cycle."""
    if frame_index < 0 or frequency_hz <= 0 or refresh_rate_hz <= 0:
        raise ValueError("frame index, frequency and refresh rate must be positive")
    half_cycles, remainder = divmod((2 * frame_index + 1) * frequency_hz, refresh_rate_hz)
    if remainder == 0 and frame_index % 2:
        half_cycles -= 1
    return half_cycles % 2 == 0


def eye_half_geometry(x: int, y: int, width: int, height: int, eye_side: str) -> dict[str, int]:
    if width < 2 or height < 1:
        raise RuntimeError("validation.screens: no usable stimulus display")
    divider = center_divider_geometry(x, y, width, height)
    left_width = divider["x"] - x
    right_x = divider["x"] + divider["width"]
    if eye_side == "left":
        return {"x": x, "y": y, "width": left_width, "height": height}
    if eye_side == "right":
        return {"x": right_x, "y": y, "width": x + width - right_x, "height": height}
    raise ValueError("validation.eye_side")


def center_divider_geometry(x: int, y: int, width: int, height: int) -> dict[str, int]:
    if width < 2 or height < 1:
        raise RuntimeError("validation.screens: no usable stimulus display")
    divider_width = min(CENTER_DIVIDER_WIDTH, max(0, width - 2))
    return {
        "x": x + (width - divider_width) // 2,
        "y": y,
        "width": divider_width,
        "height": height,
    }


def resolve_eye_regions(application: Any) -> dict[str, dict[str, Any]]:
    screens = list(application.screens())
    if not screens:
        raise RuntimeError("validation.screens: no usable stimulus display")
    primary = application.primaryScreen() if hasattr(application, "primaryScreen") else None
    screen_index = next((index for index, screen in enumerate(screens) if screen == primary), 0)
    screen = screens[screen_index]
    geometry = screen.geometry()
    screen_x = int(geometry.x())
    screen_y = int(geometry.y())
    screen_width = int(geometry.width())
    screen_height = int(geometry.height())
    divider_geometry = center_divider_geometry(
        screen_x, screen_y, screen_width, screen_height
    )
    return {
        side: {
            "screen_index": screen_index,
            "screen": screen,
            "region_geometry": eye_half_geometry(
                screen_x, screen_y, screen_width, screen_height, side,
            ),
            "divider_geometry": divider_geometry,
        }
        for side in ("left", "right")
    }


def resolve_monitor_screen(application: Any, stimulus_screen_index: int | None = None):
    """Pick an extended screen that cannot cover the stimulus display.

    Qt's screen order is not necessarily primary-first. Mirrored/overlapping
    displays are deliberately not treated as usable operator monitors.
    """
    screens = list(application.screens())
    if len(screens) < 2:
        return None
    primary = application.primaryScreen()
    if stimulus_screen_index is not None and 0 <= stimulus_screen_index < len(screens):
        stimulus = screens[stimulus_screen_index]
    else:
        stimulus = next((screen for screen in screens if screen == primary), screens[0])
    # The worker's recorded index can become stale after a hotplug/reorder.
    # Never put an operator window on the current primary display either.
    protected = [stimulus]
    if primary is not None and primary != stimulus:
        protected.append(primary)
    for screen in screens:
        if screen in protected:
            continue
        candidate = screen.geometry()
        if candidate.width() <= 0 or candidate.height() <= 0:
            continue
        overlaps = any(
            candidate.x() < rect.x() + rect.width()
            and rect.x() < candidate.x() + candidate.width()
            and candidate.y() < rect.y() + rect.height()
            and rect.y() < candidate.y() + candidate.height()
            for rect in (item.geometry() for item in protected)
        )
        if not overlaps:
            return screen
    return None
