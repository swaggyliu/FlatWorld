"""Off-screen Warp Viewer capture helpers for the docs gallery."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from flatworld.viewer import GALLERY_BG


class GalleryGUI:
    """Wrap Viewer: capture one frame to disk, then stop interactive loops."""

    def __init__(self, gui, output_path: str | Path, capture_frame: int = 1, max_frames: int = 120):
        self._gui = gui
        self._output_path = Path(output_path)
        self._capture_frame = max(1, capture_frame)
        self._max_frames = max(self._capture_frame, max_frames)
        self._shown = 0
        self.running = True
        self.captured = False

    def __getattr__(self, name):
        return getattr(self._gui, name)

    def clear(self, color=None):
        self._gui.clear(GALLERY_BG if color is None else color)

    def show(self):
        self._shown += 1
        if self._shown == self._capture_frame and not self.captured:
            save_gui_image(self._gui, self._output_path)
            self.captured = True
        self._gui.show()
        if self._shown >= self._max_frames:
            self.running = False

    def get_event(self):
        return None


def save_gui_image(gui, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(gui, "save_png"):
        return gui.save_png(output_path)
    img = gui.get_image()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.shape[-1] == 4:
        img = img[..., :3]
    try:
        from PIL import Image

        Image.fromarray(img, mode="RGB").save(str(output_path))
    except ImportError:
        import matplotlib.pyplot as plt

        plt.imsave(str(output_path), img)
    return output_path


def install_gallery_capture(output_path: str | Path, capture_frame: int = 1, max_frames: int = 120):
    """Patch test_utils so demos render off-screen and save one screenshot."""
    import test_utils
    from flatworld.viewer import create_viewer
    from flatworld.wp_init import init_warp

    init_warp(prefer_cuda=True)
    output_path = Path(output_path)

    def create_gui_for_capture(title, res=(720, 720), background_color=GALLERY_BG):
        inner = create_viewer(title, res=res, background_color=GALLERY_BG, headless=True)
        if inner is None:
            raise RuntimeError("Failed to create headless Warp Viewer for gallery capture")
        return GalleryGUI(inner, output_path, capture_frame, max_frames)

    test_utils.is_display_available = lambda: True
    test_utils.should_use_gui = lambda: True
    test_utils.create_gui_if_available = create_gui_for_capture
    test_utils.create_window_if_available = lambda title, size=(720, 720): None
    test_utils.init_sim = lambda prefer_cuda=True, device=None: init_warp(device=device, prefer_cuda=prefer_cuda)

    os.environ["FLATWORLD_GALLERY_CAPTURE"] = "1"
    os.environ["HEADLESS"] = "1"


def gallery_active() -> bool:
    return os.environ.get("FLATWORLD_GALLERY_CAPTURE", "").lower() in ("1", "true", "yes")
