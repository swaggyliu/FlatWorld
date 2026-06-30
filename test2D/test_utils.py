"""
Utility functions for test suite
Provides headless mode detection and GUI initialization helpers
"""

import os
import sys


def is_display_available():
    """
    Check if a display/GUI environment is available.

    Returns:
        bool: True if display is available, False otherwise
    """
    # Check for explicit headless flag in environment
    if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
        return False

    # Check for CI environment variables
    ci_vars = ["CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI"]
    if any(os.environ.get(var) for var in ci_vars):
        return False

    # Platform-specific checks
    if sys.platform.startswith("linux"):
        # Check for DISPLAY environment variable on Linux
        if not os.environ.get("DISPLAY"):
            return False

    # If all checks pass, assume display is available
    return True


def should_use_gui():
    """
    Determine if GUI should be used based on environment.
    Can be overridden with --headless command line argument.

    Returns:
        bool: True if GUI should be used, False for headless mode
    """
    return is_display_available()


def create_gui_if_available(title, res=(720, 720), background_color=0x000000):
    """
    Create a Taichi GUI if display is available, otherwise return None.

    Args:
        title (str): Window title
        res (tuple): Resolution (width, height)
        background_color (int): Background color in hex format

    Returns:
        ti.GUI or None: GUI object if display available, None otherwise
    """
    import taichi as ti

    if is_display_available():
        try:
            return ti.GUI(title, res=res, background_color=background_color, show_gui=True)
        except Exception as e:
            print(f"Warning: Failed to create GUI: {e}")
            print("Falling back to headless mode...")
            return None
    else:
        print(f"Running in headless mode (no display available)")
        return None


def create_window_if_available(title, size=(720, 720)):
    """
    Create a Taichi 3D `Window` if display is available, otherwise return None.
    """
    import taichi as ti

    if is_display_available():
        try:
            return ti.ui.Window(title, size)
        except Exception as e:
            print(f"Warning: Failed to create ti.ui.Window: {e}")
            print("Falling back to headless mode...")
            return None
    else:
        print(f"Running in headless mode (no display available)")
        return None


def parse_test_args():
    """
    Parse common test arguments including --headless flag.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
    )
    return parser.parse_args()
