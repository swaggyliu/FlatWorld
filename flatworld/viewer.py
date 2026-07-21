"""2D viewer backed by warp.render.OpenGLRenderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

from .wp_init import ensure_warp

GALLERY_BG = 0x112F41
_BG_RGB = (
    ((GALLERY_BG >> 16) & 0xFF) / 255.0,
    ((GALLERY_BG >> 8) & 0xFF) / 255.0,
    (GALLERY_BG & 0xFF) / 255.0,
)

_GL_INTEROP_PATCHED = False
_PYGLET_PATCHED = False


def _patch_registered_gl_buffer() -> None:
    """Avoid Warp CUDA–OpenGL interop spam (error 304) on Windows.

    ``RegisteredGLBuffer`` always calls ``wp_cuda_graphics_register_gl_buffer``,
    which logs a native CUDA error on every new VBO when interop is unavailable.
    Skip registration on CPU devices, and after the first CUDA failure, reuse the
    copy fallback without re-calling the CUDA entry point.
    """
    global _GL_INTEROP_PATCHED
    if _GL_INTEROP_PATCHED:
        return

    from warp._src.context import RegisteredGLBuffer, get_device, log_warning, runtime

    if getattr(RegisteredGLBuffer, "_flatworld_patched", False):
        _GL_INTEROP_PATCHED = True
        return

    def __init__(
        self,
        gl_buffer_id: int,
        device=None,
        flags: int = RegisteredGLBuffer.NONE,
        fallback_to_copy: bool = True,
    ):
        self.gl_buffer_id = gl_buffer_id
        self.device = get_device(device)
        self.context = self.device.context
        self.flags = flags
        self.fallback_to_copy = fallback_to_copy
        self.resource = None
        self.warp_buffer = None
        self.warp_buffer_cpu = None

        try_cuda = bool(self.device.is_cuda) and getattr(RegisteredGLBuffer, "_flatworld_interop_ok", True)
        attempted_cuda = False
        if try_cuda:
            attempted_cuda = True
            self.resource = runtime.core.wp_cuda_graphics_register_gl_buffer(self.context, gl_buffer_id, flags)
            if self.resource is None:
                RegisteredGLBuffer._flatworld_interop_ok = False

        if self.resource is None:
            if not fallback_to_copy:
                raise RuntimeError(f"Failed to register OpenGL buffer {gl_buffer_id} with CUDA")
            # Name-mangled class attr from Warp's RegisteredGLBuffer
            warned_attr = "_RegisteredGLBuffer__fallback_warning_shown"
            if attempted_cuda and not getattr(RegisteredGLBuffer, warned_attr, False):
                log_warning(
                    "CUDA/OpenGL interop unavailable; using copy fallback for Warp OpenGLRenderer "
                    "(physics can still run on CUDA)."
                )
                setattr(RegisteredGLBuffer, warned_attr, True)

    RegisteredGLBuffer.__init__ = __init__
    RegisteredGLBuffer._flatworld_patched = True
    RegisteredGLBuffer._flatworld_interop_ok = True
    _GL_INTEROP_PATCHED = True


def _patch_pyglet_dead_weakmethod() -> None:
    """Silence pyglet AssertionError when focus/close runs after handlers were GC'd.

    On Windows, ``WM_KILLFOCUS`` / window teardown can dispatch ``on_deactivate``
    while a ``WeakMethod`` event handler is already gone, which triggers
    ``assert handler is not None`` inside pyglet.event.EventDispatcher.

    Keep the original dispatcher; only swallow that specific teardown assert.
    """
    global _PYGLET_PATCHED
    if _PYGLET_PATCHED:
        return
    try:
        from pyglet.event import EventDispatcher
    except ImportError:
        return

    if getattr(EventDispatcher, "_flatworld_patched", False):
        _PYGLET_PATCHED = True
        return

    _orig = EventDispatcher.dispatch_event

    def dispatch_event(self, event_type, *args):
        try:
            return _orig(self, event_type, *args)
        except AssertionError:
            # Dead WeakMethod during window teardown / focus loss.
            return None

    EventDispatcher.dispatch_event = dispatch_event
    EventDispatcher._flatworld_patched = True
    _PYGLET_PATCHED = True


def _hex_to_rgb(color: int) -> tuple[float, float, float]:
    return (
        ((color >> 16) & 0xFF) / 255.0,
        ((color >> 8) & 0xFF) / 255.0,
        (color & 0xFF) / 255.0,
    )


def _as_xy(pos) -> tuple[float, float]:
    if hasattr(pos, "__len__"):
        return float(pos[0]), float(pos[1])
    return float(pos), 0.0


class Viewer:
    """Minimal GUI-compatible surface for FlatWorld 2D demos.

    Simulation coordinates are treated as XY in a top-down view (camera along +Z).
    Circles become spheres; lines become thin capsules / line lists.

    Radius arguments follow the old GUI convention: **pixels**. Call sites draw a
    body of world-radius ``R`` as ``gui.circle(..., radius=R * resolution)`` with
    ``resolution ≈ window width``, so world radius is ``radius_px / width``.

    Each ``show()`` presents only the shapes submitted since the previous show
    (or since ``clear()``), matching continuous animation rather than trails.
    """

    def __init__(
        self,
        title: str = "FlatWorld",
        res=(720, 720),
        background_color: int = GALLERY_BG,
        headless: bool = False,
        world_size: float = 1.0,
        view_margin: float = 0.05,
    ):
        ensure_warp()
        _patch_registered_gl_buffer()
        _patch_pyglet_dead_weakmethod()
        w, h = int(res[0]), int(res[1])
        bg = _hex_to_rgb(background_color) if isinstance(background_color, int) else background_color
        self._bg = bg
        # Simulated XY domain is typically the unit square [0, 1]^2.
        self._world_size = float(world_size)
        self._view_margin = float(view_margin)
        self.running = True
        self._headless = headless

        import math
        import warp.render

        # Frame [0, world_size]^2 with a small margin under perspective projection.
        extent = self._world_size * (1.0 + 2.0 * self._view_margin)
        half = self._world_size * 0.5
        fov = 40.0
        cam_z = (0.5 * extent) / math.tan(math.radians(fov * 0.5))
        # Renderer buffers stay on CPU so we never need CUDA–GL interop.
        # Physics kernels can still run on CUDA via wp.set_device.
        self._renderer = warp.render.OpenGLRenderer(
            title=title,
            screen_width=w,
            screen_height=h,
            background_color=bg,
            headless=headless,
            draw_grid=False,
            draw_sky=False,
            draw_axis=False,
            show_info=False,
            camera_pos=(half, half, cam_z),
            camera_front=(0.0, 0.0, -1.0),
            camera_up=(0.0, 1.0, 0.0),
            camera_fov=fov,
            near_plane=0.01,
            far_plane=max(cam_z * 4.0, 100.0),
            vsync=False,
            device="cpu",
            # BaseInstance path crashes on some Windows GL drivers (access violation).
            use_legacy_opengl=True,
        )
        self._width = w
        self._height = h
        self._shapes: list[tuple] = []
        self._instance_names: set[str] = set()
        self._time = 0.0
        self._dt = 1.0 / 60.0
        self._flushed = False
        self._closed = False

        import atexit

        atexit.register(self.close)

    def _px_to_world(self, radius_px: float) -> float:
        """Convert GUI pixel radius to world units (unit square fills the window width)."""
        return max(float(radius_px) / float(self._width), 1e-5)

    def circle_world(self, pos, radius: float, color=0xFFFFFF):
        """Draw a circle using a world-space radius (preferred for rigid bodies)."""
        self._begin_draw()
        x, y = _as_xy(pos)
        self._shapes.append(("sphere", (x, y, 0.0), max(float(radius), 1e-5), _hex_to_rgb(color)))

    def line_world(self, a, b, radius: float = 0.002, color=0xFFFFFF):
        """Draw a line segment with world-space thickness radius."""
        self._begin_draw()
        ax, ay = _as_xy(a)
        bx, by = _as_xy(b)
        self._shapes.append(
            ("line", (ax, ay, 0.0), (bx, by, 0.0), max(float(radius), 1e-5), _hex_to_rgb(color))
        )

    # --- drawing API -----------------------------------------------------------

    def clear(self, color=None):
        if color is not None:
            if isinstance(color, int):
                self._bg = _hex_to_rgb(color)
            else:
                self._bg = color
        self._shapes.clear()
        self._flushed = False

    def circle(self, pos, color=0xFFFFFF, radius=5):
        self._begin_draw()
        x, y = _as_xy(pos)
        self._shapes.append(("sphere", (x, y, 0.0), self._px_to_world(radius), _hex_to_rgb(color)))

    def circles(self, pos, radius=2, color=0xFFFFFF):
        self._begin_draw()
        arr = np.asarray(pos, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        r = self._px_to_world(radius)
        rgb = _hex_to_rgb(color)
        for p in arr:
            self._shapes.append(("sphere", (float(p[0]), float(p[1]), 0.0), r, rgb))

    def line(self, a, b, radius=1, color=0xFFFFFF):
        self._begin_draw()
        ax, ay = _as_xy(a)
        bx, by = _as_xy(b)
        # Same pixel→world mapping as circles (capsule radius / box stroke).
        self._shapes.append(
            ("line", (ax, ay, 0.0), (bx, by, 0.0), self._px_to_world(radius), _hex_to_rgb(color))
        )

    def lines(self, a, b, radius=1, color=0xFFFFFF):
        aa = np.asarray(a, dtype=np.float32)
        bb = np.asarray(b, dtype=np.float32)
        if aa.ndim == 1:
            self.line(aa, bb, radius=radius, color=color)
            return
        for i in range(len(aa)):
            self.line(aa[i], bb[i], radius=radius, color=color)

    def text(self, content, pos=(0.0, 0.0), font_size=20, color=0xFFFFFF):
        return

    def get_event(self):
        return None

    def show(self):
        if self._closed:
            self.running = False
            return
        self._flush_frame()
        self._shapes.clear()
        self._flushed = True
        self._time += self._dt
        if getattr(self._renderer, "has_exit", False):
            self.close()

    def close(self):
        """Release the OpenGL window cleanly (avoids pyglet teardown asserts)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.running = False
        r = getattr(self, "_renderer", None)
        if r is None:
            return
        try:
            win = getattr(r, "window", None)
            if win is not None:
                # Detach handlers before destroy so WM_KILLFOCUS cannot hit dead WeakMethods.
                try:
                    win.remove_handlers()
                except Exception:
                    pass
                try:
                    win.set_visible(False)
                except Exception:
                    pass
            r.close()
        except Exception:
            pass
        self._renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_image(self) -> np.ndarray:
        """Return HxWx3 float RGB in [0, 1] (row 0 = top of image)."""
        if self._closed or self._renderer is None:
            return np.zeros((self._height, self._width, 3), dtype=np.float32)
        self._flush_frame(capture=True)
        pixels = wp.zeros((self._height, self._width, 3), dtype=float, device="cpu")
        self._renderer.get_pixels(pixels, split_up_tiles=False, mode="rgb")
        # Warp's get_pixels already flips OpenGL's bottom-left origin to image top-left.
        return np.clip(pixels.numpy(), 0.0, 1.0).astype(np.float32)

    def save_png(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img = self.get_image()
        u8 = (img * 255.0).astype(np.uint8)
        try:
            from PIL import Image

            Image.fromarray(u8, mode="RGB").save(str(path))
        except ImportError:
            import matplotlib.pyplot as plt

            plt.imsave(str(path), u8)
        return path

    def _begin_draw(self):
        if self._flushed:
            self._shapes.clear()
            self._flushed = False

    def _flush_frame(self, capture: bool = False):
        r = self._renderer
        if r is None:
            return
        r.begin_frame(self._time)

        _OFF = (-1.0e6, -1.0e6, -1.0e6)
        used: set[str] = set()

        for i, shape in enumerate(self._shapes):
            name = f"fw_{i}"
            used.add(name)
            kind = shape[0]
            if kind == "sphere":
                _, center, radius, color = shape
                r.render_sphere(
                    name=name,
                    pos=center,
                    rot=(0.0, 0.0, 0.0, 1.0),
                    radius=radius,
                    color=color,
                )
            elif kind == "line":
                _, p0, p1, radius, color = shape
                dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                length = float(np.hypot(dx, dy))
                if length < 1e-9:
                    r.render_sphere(
                        name=name,
                        pos=p0,
                        rot=(0.0, 0.0, 0.0, 1.0),
                        radius=radius,
                        color=color,
                    )
                else:
                    try:
                        r.render_line_list(
                            name=name,
                            vertices=np.array([p0, p1], dtype=np.float32),
                            indices=np.array([0, 1], dtype=np.int32),
                            color=color,
                            radius=radius,
                        )
                    except TypeError:
                        mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, 0.0)
                        r.render_capsule(
                            name=name,
                            pos=mid,
                            radius=radius,
                            half_height=max(length * 0.5, radius),
                            color=color,
                        )

        for name in self._instance_names - used:
            r.update_shape_instance(name, pos=_OFF)
        self._instance_names |= used

        r.end_frame()


def create_viewer(
    title: str,
    res=(720, 720),
    background_color: int = GALLERY_BG,
    headless: bool | None = None,
) -> Viewer | None:
    """Factory used by tests; returns None only if OpenGL init fails."""
    if headless is None:
        headless = os_environ_headless()
    try:
        return Viewer(title=title, res=res, background_color=background_color, headless=headless)
    except Exception as e:
        print(f"Warning: Failed to create Warp Viewer: {e}")
        return None


def os_environ_headless() -> bool:
    import os

    if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True
    return False
