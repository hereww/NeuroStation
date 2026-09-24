"""Map each eye to a half of the primary display for UI and acquisition."""

from __future__ import annotations

from typing import Any


def eye_half_geometry(x: int, y: int, width: int, height: int, eye_side: str) -> dict[str, int]:
    if width < 2 or height < 1:
        raise RuntimeError("validation.screens: no usable stimulus display")
    left_width = width // 2
    if eye_side == "left":
        return {"x": x, "y": y, "width": left_width, "height": height}
    if eye_side == "right":
        return {"x": x + left_width, "y": y, "width": width - left_width, "height": height}
    raise ValueError("validation.eye_side")


def resolve_eye_regions(application: Any) -> dict[str, dict[str, Any]]:
    screens = list(application.screens())
    if not screens:
        raise RuntimeError("validation.screens: no usable stimulus display")
    primary = application.primaryScreen() if hasattr(application, "primaryScreen") else None
    screen_index = next((index for index, screen in enumerate(screens) if screen == primary), 0)
    screen = screens[screen_index]
    geometry = screen.geometry()
    return {
        side: {
            "screen_index": screen_index,
            "screen": screen,
            "region_geometry": eye_half_geometry(
                int(geometry.x()), int(geometry.y()),
                int(geometry.width()), int(geometry.height()), side,
            ),
        }
        for side in ("left", "right")
    }
