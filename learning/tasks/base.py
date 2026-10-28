"""Shared FlatWorld task protocol for CEM / BPTT / PPO consumers.

One scene task exposes a small surface:

  observe · sample_goal · potential (Φ) · metric · succeeded · step

- **WM+CEM** scores ``metric`` / geometric costs; may ignore Φ.
- **WM+PPO** uses ``r = Φ_prev − Φ`` (+ sparse success).
- **WM+BPTT** (SuperMimic) uses a parallel mimic interface
  (:class:`MimicRefTask`) with dense tracking potentials.

Concrete CEM tasks subclass :class:`CEMWorldTask` (or PushToGoalTask) and
inherit the defaults below. Push-T overrides ``potential`` to match the
RL progress reward in ``learning.rl.imagine``.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FlatWorldTask(Protocol):
    """Minimal task API shared across controllers."""

    tol: float
    budget: int

    def observe(self) -> dict:
        """Current structured observation (obj_states, geom, tactile, …)."""
        ...

    def sample_goal(self, rng: np.random.Generator, obs: dict) -> Any:
        """Sample a goal for this episode (array or dict)."""
        ...

    def potential(self, obs: dict, goal: Any) -> float:
        """Scalar potential Φ(s, g). Lower is better. PPO uses ΔΦ."""
        ...

    def metric(self, obs: dict, goal: Any) -> float:
        """Scalar distance / error reported in evals (often equals Φ)."""
        ...

    def succeeded(self, obs: dict, goal: Any) -> bool:
        """Episode success predicate."""
        ...


@runtime_checkable
class SteppableTask(FlatWorldTask, Protocol):
    """Task that can apply an action in the real simulator."""

    def reset(self, rng: np.random.Generator) -> dict:
        ...

    def step(self, action: np.ndarray) -> dict:
        ...


@runtime_checkable
class MimicRefTask(Protocol):
    """BPTT / SuperMimic-style tracking interface (parallel to FlatWorldTask)."""

    def reference_pose(self, t: int) -> Any:
        ...

    def tracking_potential(self, state: Any, ref: Any) -> float:
        """Dense mimic cost (pose / vel / root); BPTT differentiates through WM."""
        ...


class TaskMixin:
    """Default implementations mixed into CEM / Push tasks.

    Subclasses usually already have ``_observe``, ``metric``, ``succeeded``,
    ``sample_goal``. This mixin adds the unified names and a default Φ.
    """

    tol: float = 0.08
    budget: int = 400
    goal: Any = None

    def observe(self) -> dict:
        if hasattr(self, "_observe"):
            return self._observe()
        return self.env._observe()  # type: ignore[attr-defined]

    def potential(self, obs: dict, goal: Any = None) -> float:
        """Default Φ = metric (distance-to-goal style tasks)."""
        g = self.goal if goal is None else goal
        return float(self.metric(obs, g))  # type: ignore[attr-defined]

    def reset(self, rng: np.random.Generator) -> dict:
        obs = self.env.reset(rng)  # type: ignore[attr-defined]
        if hasattr(self, "_observe"):
            return self._observe()
        return obs

    def step(self, action: np.ndarray) -> dict:
        self.env.step(np.asarray(action, dtype=np.float32))  # type: ignore[attr-defined]
        return self.observe()


def progress_reward(
    task: FlatWorldTask,
    obs_prev: dict,
    obs: dict,
    goal: Any = None,
    *,
    success_bonus: float = 1.0,
) -> tuple[float, bool, float]:
    """Shared PPO-style progress: ``r = Φ_prev − Φ + bonus·success``.

    Returns ``(reward, success, phi_now)``.
    """
    g = goal if goal is not None else getattr(task, "goal", None)
    phi_prev = float(task.potential(obs_prev, g))
    phi = float(task.potential(obs, g))
    ok = bool(task.succeeded(obs, g))
    r = (phi_prev - phi) + (success_bonus if ok else 0.0)
    return r, ok, phi


def as_flatworld_task(obj: Any) -> Optional[FlatWorldTask]:
    """Return ``obj`` if it satisfies :class:`FlatWorldTask`, else ``None``."""
    if isinstance(obj, FlatWorldTask):
        return obj
    needed = ("observe", "sample_goal", "potential", "metric", "succeeded")
    if all(callable(getattr(obj, n, None)) for n in needed):
        return obj  # duck-typed
    return None
