"""Display geometry and frame-accurate SSVEP helpers."""

from __future__ import annotations

from typing import Any
import math

from neurostation_contract import StimulusCondition


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


def allow_stimulus_process_foreground(process_id: int | None) -> bool | None:
    """Let the user-started acquisition child request focus on Windows.

    This is an OS permission request, not evidence that its window is visible.
    Do not simulate keyboard input or attach the operator's input thread.
    """
    import sys
    if sys.platform != "win32":
        return None
    if type(process_id) is not int or not 0 < process_id < 0xFFFFFFFF:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        allow = user32.AllowSetForegroundWindow
        allow.argtypes = [wintypes.DWORD]
        allow.restype = wintypes.BOOL
        return bool(allow(process_id))
    except (AttributeError, OSError):
        return False


def show_stimulus_fullscreen(window: Any, screen: Any) -> None:
    """Keep the acquisition-owned stimulus above ordinary workstation windows.

    Bind the screen before showing; geometry remains in Qt logical coordinates.
    This function does not render or start a stimulus, open hardware, or change
    which screen was selected by the validated display mapping.
    """
    from PySide6.QtCore import Qt
    window.setFlags(window.flags() | Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint)
    window.setScreen(screen)
    window.setGeometry(screen.geometry())
    window.showFullScreen()
    window.raise_()
    window.requestActivate()


def stimulus_geometry(x: int, y: int, width: int, height: int,
                      condition: str, progress: float) -> dict[str, int]:
    """One horizontal traversal, clamped at the endpoint; no wrap or bounce.

    A moving rectangle retains the smaller half's width on odd-size displays.
    Coordinates are in the caller's units (physical pixels in the GL renderer).
    """
    mode = StimulusCondition(condition)
    if not math.isfinite(progress):
        raise ValueError("motion progress must be finite")
    left = eye_half_geometry(x, y, width, height, "left")
    right = eye_half_geometry(x, y, width, height, "right")
    if not mode.moving:
        return left if mode.start_side == "left" else right
    target_width = min(left["width"], right["width"])
    start_x, end_x = left["x"], right["x"]
    if mode is StimulusCondition.R_TO_L:
        start_x, end_x = end_x, start_x
    position = min(1.0, max(0.0, progress))
    return {"x": round(start_x + (end_x - start_x) * position), "y": y,
            "width": target_width, "height": height}
