"""
Utility functions for test suite
Provides headless mode detection and GUI / Warp init helpers
"""

import os
import sys


def is_display_available():
    """Check if a display/GUI environment is available."""
    if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
        return False

    ci_vars = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI"]
    if any(os.environ.get(var) for var in ci_vars):
        return False

    # Pytest + live OpenGL in one process is unstable on Windows (access
    # violations inside Warp/pyglet). Opt in with FLATWORLD_GUI=1.
    gui_opt_in = os.environ.get("FLATWORLD_GUI", "").lower() in ("1", "true", "yes")
    if not gui_opt_in and ("pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST")):
        return False

    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY"):
            return False

    return True


def should_use_gui():
    """Determine if GUI should be used based on environment."""
    return is_display_available()


def create_gui_if_available(title, res=(720, 720), background_color=0x112F41):
    """Create a Warp-backed Viewer if a display is available.

    Under HEADLESS/CI, returns None so physics tests skip OpenGL entirely
    (avoids flaky/hanging headless GL). Gallery capture should construct
    ``Viewer(..., headless=True)`` directly when off-screen pixels are needed.
    """
    from flatworld.viewer import GALLERY_BG, create_viewer

    if background_color is None:
        background_color = GALLERY_BG

    if not is_display_available():
        return None
    return create_viewer(title, res=res, background_color=background_color, headless=False)


def create_window_if_available(title, size=(720, 720)):
    """3D window helper — maps to Viewer for Warp."""
    return create_gui_if_available(title, res=size)


def parse_test_args():
    """Parse common test arguments including --headless flag."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    return parser.parse_args()


def init_sim(prefer_cuda: bool = True, device: str | None = None) -> str:
    """Initialize Warp for a test module."""
    from flatworld.wp_init import init_warp

    # CI / explicit env prefers CPU unless overridden
    if device is None and os.environ.get("FLATWORLD_DEVICE"):
        device = os.environ["FLATWORLD_DEVICE"]
    if device is None and os.environ.get("CI"):
        prefer_cuda = False
    return init_warp(device=device, prefer_cuda=prefer_cuda)
