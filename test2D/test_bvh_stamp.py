"""Regression tests for BVH domain stamp lifecycle."""

from flatworld import ExplicitLoop


class _StampArray:
    def __init__(self):
        self.zero_calls = 0

    def zero_(self):
        self.zero_calls += 1


def _bare_loop(stamp_counter=1):
    loop = ExplicitLoop.__new__(ExplicitLoop)
    loop._bvh_stamp_counter_py = stamp_counter
    loop._bvh_domain_stamp = _StampArray()
    return loop


def test_bvh_stamp_counter_does_not_rewind_with_simulation_reset():
    loop = _bare_loop()

    first = loop._next_bvh_domain_stamp()
    # ExplicitLoop.reset() rewinds this simulation counter; the BVH stamp
    # counter must remain independent because the device stamp array is stale.
    loop._counter_py = 0
    second = loop._next_bvh_domain_stamp()

    assert first == 2
    assert second == 3
    assert second != first
    assert loop._bvh_domain_stamp.zero_calls == 0


def test_bvh_stamp_wrap_clears_old_domain_stamps_before_reuse():
    loop = _bare_loop(stamp_counter=2_000_000_000)

    stamp = loop._next_bvh_domain_stamp()

    assert stamp == 1
    assert loop._bvh_stamp_counter_py == 1
    assert loop._bvh_domain_stamp.zero_calls == 1
