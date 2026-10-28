"""Task modules. Import submodules directly (``learning.tasks.two_room``)
so solver-CEM eval does not pull torch via the world-model planner.

Unified controller surface: :class:`learning.tasks.base.FlatWorldTask`
(``observe`` / ``sample_goal`` / ``potential`` / ``metric`` / ``succeeded``).
See :mod:`learning.tasks.adapters` for CEM / PPO / BPTT helpers.
"""

__all__ = [
    "FlatWorldTask", "TaskMixin", "progress_reward",
    "PushToGoalTask", "PushToGoalCost", "CEMPlanner",
    "load_model", "load_ensemble",
    "TwoRoomTask", "TwoRoomCost", "two_room_config",
    "PushTTask", "PushTCost", "push_t_config",
    "ReacherTask", "ReacherCost", "reacher_config",
    "CEMWorldTask",
]


def __getattr__(name: str):
    if name in ("FlatWorldTask", "TaskMixin", "progress_reward",
                "MimicRefTask", "as_flatworld_task"):
        from learning.tasks import base as _base
        return getattr(_base, name)
    if name == "CEMWorldTask":
        from learning.tasks.cem_loop import CEMWorldTask
        return CEMWorldTask
    if name in ("PushToGoalTask", "PushToGoalCost", "CEMPlanner",
                "load_model", "load_ensemble"):
        from learning.tasks import push_to_goal as _m
        return getattr(_m, name)
    if name in ("TwoRoomTask", "TwoRoomCost", "two_room_config"):
        from learning.tasks import two_room as _m
        return getattr(_m, name)
    if name in ("PushTTask", "PushTCost", "push_t_config"):
        from learning.tasks import push_t as _m
        return getattr(_m, name)
    if name in ("ReacherTask", "ReacherCost", "reacher_config"):
        from learning.tasks import reacher as _m
        return getattr(_m, name)
    raise AttributeError(name)
