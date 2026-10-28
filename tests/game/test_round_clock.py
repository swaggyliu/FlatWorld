"""Round timing uses the production session with lightweight physics/AI doubles."""

import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def game(monkeypatch):
    # These tests exercise timing, reset and finish logic, not model inference.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=lambda: lambda fn: fn))
    monkeypatch.setitem(sys.modules, "tensordict", types.SimpleNamespace(TensorDict=object))
    config = types.SimpleNamespace(
        scene=types.SimpleNamespace(ee_radius=0.05),
        collect=types.SimpleNamespace(force_max=6))

    class Env:
        def reset(self, rng): pass
        def set_ee_radius(self, radius): pass
        def set_force(self, force): self.force = force
        def step(self, action): pass

    class Task:
        def __init__(self, cfg, **kwargs):
            self.cfg, self.env, self.ee_scale = cfg, Env(), 1
            self.budget = 1100
            self.stem_goal, self.bar_goal = np.zeros(2), np.ones(2)
            self.goal_theta, self.goal, self.success = 0, None, False
        def observe(self):
            return {"obj_states": np.zeros((3, 6)), "obj_geom": np.ones((3, 2)) * 0.05}
        def sample_goal(self, rng, obs): return np.zeros(2)
        def succeeded(self, obs, goal): return self.success

    monkeypatch.setitem(sys.modules, "learning.tasks.push_t", types.SimpleNamespace(
        PushTTask=Task, push_t_config=lambda: config))
    source = Path(__file__).resolve().parents[2] / "game" / "pusht_session.py"
    spec = importlib.util.spec_from_file_location("_pusht_clock_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    clock = types.SimpleNamespace(now=0)
    module.time = types.SimpleNamespace(perf_counter=lambda: clock.now)
    assets = types.SimpleNamespace(device="cpu", wm=None, policy=None, hold=5,
                                  task_features=None, inference_lock=threading.Lock())
    session = module.PushTVersusSession(shared=assets)
    session._ai_action = lambda obs: np.zeros(2, dtype=np.float32)
    return session, clock


def test_round_adds_ten_seconds_and_countdown_does_not_spend_it(game):
    session, clock = game
    assert session.round_duration == pytest.approx(1100 / (30 * 3) + 10)
    initial = session.snapshot()["time_remaining"]
    clock.now = 1
    session.tick()
    assert session.phase == "countdown"
    assert session.snapshot()["time_remaining"] == initial
    clock.now = 3
    session.tick()
    assert session.phase == "race"
    clock.now = 8
    session.tick()
    assert session.snapshot()["time_remaining"] == pytest.approx(initial - 5)


def test_pause_freezes_time_and_the_old_step_budget_cannot_end_early(game):
    session, clock = game
    clock.now = 3
    session.tick()
    clock.now = 5
    session.tick()
    remaining = session.snapshot()["time_remaining"]
    session.paused = True
    clock.now = 15
    session.tick()
    assert session.snapshot()["time_remaining"] == remaining
    session.paused = False
    session.left.t = session.right.t = 1100
    clock.now = 16
    session.tick()
    assert session.phase == "race"
    assert session.left.t > 1100
    assert session.snapshot()["time_remaining"] == pytest.approx(remaining - 1)


def test_timeout_finishes_once_and_rematch_resets_the_clock(game):
    session, clock = game
    clock.now = 3
    session.tick()
    session.set_stick(0.5, 0)
    clock.now += session.round_duration + 0.1
    session.tick()
    assert session.phase == "over"
    assert session.draws == 1
    assert session.snapshot()["time_remaining"] == 0
    assert not session.player_stick.any()
    session.tick()
    assert session.draws == 1
    session.paused = True
    session.next_episode()
    assert session.phase == "countdown"
    assert not session.paused
    assert session.snapshot()["time_remaining"] == session.round_duration
    assert session.draws == 1


def test_early_win_stops_the_clock_and_increments_score_once(game):
    session, clock = game
    clock.now = 3
    session.tick()
    session.left.task.success = True
    clock.now = 4
    session.tick()
    assert session.phase == "over"
    assert session.winner == "player"
    assert session.player_wins == 1
    remaining = session.snapshot()["time_remaining"]
    clock.now = 100
    session.tick()
    assert session.snapshot()["time_remaining"] == remaining
    assert session.player_wins == 1
