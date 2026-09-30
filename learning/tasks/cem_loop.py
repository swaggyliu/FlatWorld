"""Shared CEM episode loop for StateLeWM tasks."""

import numpy as np


class CEMWorldTask:
    """Bind a cost, plan with CEM, execute in the simulator."""

    def __init__(self, cfg, env, model, norm, device="cpu",
                 tol: float = None, vel_tol: float = None, budget: int = None,
                 planner_kwargs: dict = None, stride: int = 1,
                 ee_scale: float = 1.0, cost=None, **_ignore):
        self.cfg = cfg
        self.env = env
        self.device = device
        tc = getattr(cfg, "task", None)
        self.tol = float(tc.tol if tol is None and tc is not None else (tol if tol is not None else 0.08))
        self.vel_tol = float(
            tc.vel_tol if vel_tol is None and tc is not None else (vel_tol if vel_tol is not None else 0.08))
        self.budget = int(
            tc.budget if budget is None and tc is not None else (budget if budget is not None else 400))
        self.stride = int(stride)
        self.ee_scale = float(ee_scale)
        pk = dict(planner_kwargs or {})
        self.use_solver_cem = bool(pk.pop("use_solver_cem", False))
        self.use_servo = bool(pk.pop("use_servo", False))
        self.exec_horizon = int(pk.pop("exec_horizon", 1))
        cost_overrides = {}
        for k in ("action_cost", "reach_cost", "settle_w"):
            if k in pk:
                cost_overrides[k] = pk.pop(k)
        for k in ("tactile_mode", "ensemble_reduce", "uncert_cost"):
            pk.pop(k, None)
        if self.use_servo:
            self.models = []
            self.model = None
            self.planner = None
        elif self.use_solver_cem:
            from learning.solver_planner import SolverCEMPlanner
            self.models = []
            self.model = None
            self.planner = SolverCEMPlanner(
                self.env, stride=self.stride,
                force_max=cfg.collect.force_max, **pk)
        elif model is None:
            self.models = []
            self.model = None
            self.planner = None
        else:
            from learning.planner import CEMPlanner
            self.models = list(model) if isinstance(model, (list, tuple)) else [model]
            self.model = self.models[0] if self.models else None
            if "tactile_residual" not in pk and self.models:
                pk["tactile_residual"] = bool(getattr(self.models[0], "tactile_residual", False))
            self.planner = CEMPlanner(
                self.models, norm, device=device,
                force_max=cfg.collect.force_max, **pk)
        self.cost = cost
        for k, v in cost_overrides.items():
            if self.cost is not None and hasattr(self.cost, k):
                setattr(self.cost, k, v)
        self.ee_idx = 0
        self.goal = None

    def _geom(self) -> np.ndarray:
        return np.asarray(self.env.obj_geom, dtype=np.float32).copy()

    def _observe(self):
        cur = self.env._observe()
        if "obj_geom" not in cur:
            cur = dict(cur)
            cur["obj_geom"] = self._geom()
        return cur

    def _hold(self, action, t, frames):
        obs = None
        for _ in range(self.stride):
            obs = self.env.step(action)
            t += 1
            if frames is not None:
                frames["obj_states"].append(obs["obj_states"])
                frames["contact_mask"].append(obs["contact_mask"])
                frames["actions"].append(np.asarray(action, dtype=np.float32))
            if t >= self.budget:
                break
        return obs, t

    def bind_cost(self, obs, goal):
        raise NotImplementedError

    def sample_goal(self, rng, obs):
        raise NotImplementedError

    def succeeded(self, obs, goal) -> bool:
        raise NotImplementedError

    def metric(self, obs, goal) -> float:
        raise NotImplementedError

    def _idle_action(self) -> np.ndarray:
        if hasattr(self.env, "hold_action"):
            return np.asarray(self.env.hold_action(), dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _plan(self, cur, goal):
        if self.succeeded(cur, goal):
            return self._idle_action()
        self.bind_cost(cur, goal)
        if getattr(self, "use_servo", False):
            return np.asarray(self.cost.hint_action, dtype=np.float32)
        if self.use_solver_cem:
            return self.planner.plan_sequence(self.cost)
        z0 = self.planner.encode_obs(cur)
        action = self.planner.plan(
            z0, geom=self._geom(), states_now=cur["obj_states"], cost=self.cost,
            idle_action=self._idle_action())
        self.planner.remember_first_step(z0, action, self._geom())
        return action

    def plan_action(self, cur, goal) -> np.ndarray:
        out = self._plan(cur, goal)
        if isinstance(out, np.ndarray) and out.ndim == 2:
            return out[0]
        return out

    def extra_result(self, obs, goal) -> dict:
        return {}

    def run_episode(self, rng: np.random.Generator, record: bool = False):
        obs = self.env.reset(rng)
        if not getattr(self.env, "torque_control", False):
            self.env.set_ee_radius(float(self.cfg.scene.ee_radius) * self.ee_scale)
            obs = self._observe()
        goal = self.sample_goal(rng, obs)
        self.goal = goal
        if self.planner is not None:
            if self.use_solver_cem:
                self.planner.reset(self._idle_action())
            else:
                self.planner.reset()
        init_dist = self.metric(obs, goal)
        frames = {"obj_states": [obs["obj_states"]],
                  "contact_mask": [obs["contact_mask"]],
                  "actions": []} if record else None
        final_dist = init_dist
        success = False
        settle_frame = self.budget
        t = 0
        while t < self.budget:
            cur = self._observe()
            final_dist = self.metric(cur, goal)
            if self.succeeded(cur, goal):
                success = True
                settle_frame = t
                self.env.set_force(self._idle_action())
                break
            planned = self._plan(cur, goal)
            if (self.use_solver_cem and self.exec_horizon > 1
                    and isinstance(planned, np.ndarray) and planned.ndim == 2):
                n_exec = min(self.exec_horizon, len(planned))
                for j in range(n_exec):
                    if t >= self.budget:
                        break
                    obs, t = self._hold(planned[j], t, frames)
                    if obs is not None:
                        final_dist = self.metric(obs, goal)
                        if self.succeeded(obs, goal):
                            success = True
                            settle_frame = t
                            self.env.set_force(self._idle_action())
                            break
                if success:
                    break
                continue
            action = planned[0] if (isinstance(planned, np.ndarray)
                                    and planned.ndim == 2) else planned
            obs, t = self._hold(action, t, frames)
            if obs is not None:
                final_dist = self.metric(obs, goal)
                if self.succeeded(obs, goal):
                    success = True
                    settle_frame = t
                    break
        result = {
            "success": bool(success),
            "target_idx": int(getattr(self, "target_idx", 1)),
            "init_dist": float(init_dist),
            "final_dist": float(final_dist),
            "settle_frame": settle_frame,
            "goal": np.asarray(goal["xy"] if isinstance(goal, dict) else goal,
                               dtype=np.float32).reshape(-1)[:2].tolist(),
            "frames": (np.stack(frames["obj_states"]),
                       np.stack(frames["contact_mask"]),
                       (np.stack(frames["actions"]) if frames["actions"]
                        else np.zeros((0, 2), dtype=np.float32))) if record else None,
        }
        result.update(self.extra_result(self._observe(), goal))
        return result
