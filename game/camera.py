"""World (metres, y-up) <-> screen (pixels, y-down)."""

from __future__ import annotations

import pyray as rl


class Camera:
    def __init__(self, width: int, height: int,
                 origin_x: float = 70.0, origin_y: float = 0.0,
                 scale: float = 420.0, ground_pad: float = 80.0):
        self.width = width
        self.height = height
        self.ox = origin_x
        self.oy = height - ground_pad
        self.scale = scale

    def to_screen(self, wx: float, wy: float) -> rl.Vector2:
        return rl.Vector2(self.ox + wx * self.scale,
                          self.oy - wy * self.scale)

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.ox) / self.scale,
                (self.oy - sy) / self.scale)

    def px(self, metres: float) -> float:
        return metres * self.scale

    def visible_x(self) -> tuple[float, float]:
        """World-x at the left and right edges of the window."""
        return ((0.0 - self.ox) / self.scale,
                (self.width - self.ox) / self.scale)
