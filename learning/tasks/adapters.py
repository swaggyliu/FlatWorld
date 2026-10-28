"""Controller adapters over the unified FlatWorldTask API.

CEM tasks already plan via :class:`~learning.tasks.cem_loop.CEMWorldTask`.
PPO / BPTT entrypoints should prefer these thin helpers so reward and
success definitions stay in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from learning.tasks.base import (
    FlatWorldTask,
    MimicRefTask,
    progress_reward,
    as_flatworld_task,
)


def cem_act(task, obs: dict, goal: Any) -> np.ndarray:
    """One CEM (or servo) action from a task that implements ``plan_action``."""
    if hasattr(task, "plan_action"):
        return np.asarray(task.plan_action(obs, goal), dtype=np.float32)
    raise TypeError(f"{type(task).__name__} has no plan_action")


def ppo_step_reward(
    task: FlatWorldTask,
    obs_prev: dict,
    obs: dict,
    goal: Any = None,
    *,
    success_bonus: float = 1.0,
) -> tuple[float, bool, dict]:
    """Progress reward for physics-loop PPO using ``task.potential``."""
    r, ok, phi = progress_reward(
        task, obs_prev, obs, goal, success_bonus=success_bonus)
    return r, ok, {"phi": phi, "success": ok}


class MimicAdapter:
    """Wrap a SuperMimic-style ref provider as :class:`MimicRefTask`."""

    def __init__(
        self,
        reference_fn: Callable[[int], Any],
        potential_fn: Callable[[Any, Any], float],
    ):
        self._ref = reference_fn
        self._pot = potential_fn

    def reference_pose(self, t: int) -> Any:
        return self._ref(t)

    def tracking_potential(self, state: Any, ref: Any) -> float:
        return float(self._pot(state, ref))


def ensure_task(obj: Any) -> FlatWorldTask:
    t = as_flatworld_task(obj)
    if t is None:
        raise TypeError(
            f"{type(obj).__name__} does not satisfy FlatWorldTask "
            f"(need observe/sample_goal/potential/metric/succeeded)")
    return t


__all__ = [
    "cem_act",
    "ppo_step_reward",
    "MimicAdapter",
    "ensure_task",
    "progress_reward",
    "FlatWorldTask",
    "MimicRefTask",
]
