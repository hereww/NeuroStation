"""Map each eye to a half of the primary display for UI and acquisition."""

from __future__ import annotations

from typing import Any


CENTER_DIVIDER_WIDTH = 6


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
