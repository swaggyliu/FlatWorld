"""Regression tests for device-resident BVH pair counts in contact activation."""

import numpy as np
import warp as wp

import flatworld  # noqa: F401  (adds the package directory to sys.path)
from mixedcontact import _activate_pairs_kernel
from wp_init import ensure_warp


def test_pair_activation_reads_count_from_device_array():
    ensure_warp()
    device = "cpu"
    pairs = wp.array(np.array([[0, 1], [2, 3]], dtype=np.int32), dtype=wp.vec2i, device=device)
    pair_a = wp.array(np.array([1, 3], dtype=np.int32), dtype=int, device=device)
    pair_b = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)

    zero_count = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    active = wp.zeros(2, dtype=int, device=device)
    wp.launch(_activate_pairs_kernel, dim=2, inputs=[pairs, zero_count, pair_a, pair_b, active, 2], device=device)
    np.testing.assert_array_equal(active.numpy(), np.array([0, 0], dtype=np.int32))

    one_count = wp.array(np.array([1], dtype=np.int32), dtype=int, device=device)
    wp.launch(_activate_pairs_kernel, dim=2, inputs=[pairs, one_count, pair_a, pair_b, active, 2], device=device)
    np.testing.assert_array_equal(active.numpy(), np.array([1, 0], dtype=np.int32))

    two_count = wp.array(np.array([2], dtype=np.int32), dtype=int, device=device)
    wp.launch(_activate_pairs_kernel, dim=2, inputs=[pairs, two_count, pair_a, pair_b, active, 2], device=device)
    np.testing.assert_array_equal(active.numpy(), np.array([1, 1], dtype=np.int32))
