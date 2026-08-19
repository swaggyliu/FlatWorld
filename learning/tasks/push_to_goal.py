"""PushToGoal task + CEM (cross-entropy method) latent-space planner.

Task: using ONLY the learned world model for planning, apply forces to
the end-effector so that a designated target object ends up within a
tolerance of a goal position. Execution and success measurement happen
in the FlatWorld simulator (the world model never sees future states).

Planner: receding-horizon CEM.
- encode the TRUE current observation -> z0
- sample P action sequences of horizon H, roll the latent dynamics,
  decode predicted states and score a cost
- keep elites, refit Gaussian mean/std, iterate; execute the first action
  of the best sequence, then replan (warm-start by shifting the mean)
"""

import numpy as np
import torch

from learning.data.normalizer import Normalizer
from learning.env.flatworld_wrapper import PushSceneEnv, OBJ_TYPE_EE
from learning.models.lewm import StateLeWM


def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StateLeWM(n_obj=ckpt["n_obj"], latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norm = Normalizer.load(
        ckpt_path.replace("best.pt", "").replace("last.pt", "") + ckpt["normalizer"]
    )
    stride = int(ckpt.get("stride", 1))  # action repeat (frames per model step)
    return model, norm, stride


class CEMPlanner:
    def __init__(self, model: StateLeWM, norm: Normalizer, device: str = "cpu",
                 horizon: int = 8, population: int = 96, elites: int = 16,
                 iterations: int = 4, force_max: float = 6.0,
                 action_cost: float = 0.002, approach_cost: float = 0.5,
                 align_cost: float = 0.5, seed: int = 0):
        self.model = model
        self.norm = norm
        self.device = device
        self.H = horizon
        self.P = population
        self.E = elites
        self.iters = iterations
        self.fmax = force_max
        self.action_cost = action_cost
        self.approach_cost = approach_cost
        self.align_cost = align_cost
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((horizon, 2), dtype=np.float32)
        # per-dimension bounds: full horizontal force, moderate vertical
        # authority (enough to press the EE below a box's centre of mass
        # before contact so the push does not tip the box over)
        self.low = np.array([-force_max, -2.5], dtype=np.float32)
        self.high = np.array([force_max, 2.5], dtype=np.float32)
        self.std = (self.high - self.low) * 0.25
        # Precomputed normalization tensors
        st = norm.to_tensors(device)
        self.st_mean = {k: v[0] for k, v in st.items()}
        self.st_std = {k: v[1] for k, v in st.items()}

    def reset(self):
        self.mean[:] = 0.0
        self.std[:] = (self.high - self.low) * 0.25

    def _normalize(self, key, x):
        return (x - self.st_mean[key]) / self.st_std[key]

    def _denormalize(self, key, x):
        return x * self.st_std[key] + self.st_mean[key]

    def encode_obs(self, obs: dict) -> torch.Tensor:
        """Encode a raw simulator observation -> normalized latent z0 (1, L)."""
        with torch.no_grad():
            states = self._normalize(
                "obj_states", torch.as_tensor(obs["obj_states"], device=self.device))
            cfeat = self._normalize(
                "contact_feat", torch.as_tensor(obs["contact_feat"], device=self.device))
            cmask = torch.as_tensor(obs["contact_mask"], device=self.device)
            tsum = self._normalize(
                "tactile_summary", torch.as_tensor(obs["tactile_summary"], device=self.device))
            types = torch.as_tensor(obs["obj_types"], device=self.device)
            z = self.model.encode(types.unsqueeze(0), states.unsqueeze(0),
                                  cfeat.unsqueeze(0), cmask.unsqueeze(0),
                                  tsum.unsqueeze(0))
        return z  # (1, L)

    def plan(self, z0: torch.Tensor, ee_idx: int, target_idx: int,
             goal: np.ndarray, standoff: float = 0.2, contact_y: float = 0.105,
             slip: float = 0.0, coast_frac: float = 0.0,
             force_cap: float = None, states_now: np.ndarray = None):
        """Optimize an action sequence. Returns best first action (2,) in N.

        The learned latent dynamics are used to roll out the EE trajectory
        under candidate force sequences. Target object motion is NOT taken
        from the decoder (the world model "telekinesis" hallucination:
        it moves far-away objects as a function of the action alone);
        instead a monotone kinematic push prior is applied on top of the
        imagined EE path: whenever the (baseline-corrected) imagined EE
        reaches a contact configuration, the target is assumed to be
        pushed to ee_x + standoff, and never behind its previous position.

        standoff: center distance between EE and target at contact
        (ee_radius + target half width).
        contact_y: desired EE centre height while pushing. Below a box's
        centre of mass the push slides it instead of tipping it over.
        slip: contact slip margin. The prior assumes the target trails the
        EE contact front by this amount (friction lag for sliding boxes;
        0 for rolling balls). Without it the prior is over-optimistic and
        the CEM stops pushing too early.
        coast_frac: fraction of the push distance that the target keeps
        ROLLING after contact ends (balls coast, boxes stop almost
        immediately). The projected final position is
        pushed + coast_frac * (pushed - start), so the CEM stops pushing
        early enough that the coast lands the target on the goal.
        force_cap: optional per-call horizontal force bound (N). Boxes are
        capped below their tipping threshold; balls may use the full range.

        states_now: true current object states (N, 6), raw units. A
        zero-action baseline rollout is subtracted from the imagined
        states and replaced by the true current states, which cancels
        action-independent hallucinated drift of the learned model
        (resting objects must not appear to slide on their own).
        """
        z0 = z0.expand(self.P, -1)
        goal_x = float(goal[0])
        tgt_y = contact_y

        # ----- zero-action baseline rollout (drift reference) -----
        with torch.no_grad():
            a0 = self._normalize("actions", torch.zeros(1, 2, device=self.device))
            zb = z0[:1]
            hb = self.model.predictor.init_hidden(zb)
            base = []
            for _ in range(self.H):
                zb, hb = self.model.predictor.step(zb, a0, hb)
                base.append(self._denormalize(
                    "obj_states", self.model.decoder(zb)[0])[0])   # (N, 6)
            base = torch.stack(base)                               # (H, N, 6)
        if states_now is not None:
            now_t = torch.as_tensor(states_now, dtype=torch.float32,
                                    device=self.device)            # (N, 6)
        else:
            now_t = base[0]
        tgt_x0 = float(now_t[target_idx, 0])
        contact_x = tgt_x0 - standoff   # EE center x at first touch
        best_seq = None

        low = self.low if force_cap is None else \
            np.array([-force_cap, self.low[1]], dtype=np.float32)
        high = self.high if force_cap is None else \
            np.array([force_cap, self.high[1]], dtype=np.float32)

        for _ in range(self.iters):
            seqs = self.rng.normal(size=(self.P, self.H, 2)).astype(np.float32)
            seqs = self.mean[None] + self.std[None] * seqs
            seqs = np.clip(seqs, low, high)
            a = torch.as_tensor(seqs, device=self.device)

            with torch.no_grad():
                h = self.model.predictor.init_hidden(z0)
                z = z0
                cost = torch.zeros(self.P, device=self.device)
                pushed = torch.full((self.P,), float(tgt_x0), device=self.device)
                for t in range(self.H):
                    a_n = self._normalize("actions", a[:, t])
                    z, h = self.model.predictor.step(z, a_n, h)
                    states = self.model.decoder(z)[0]              # (P, N, 6) normalized
                    states = self._denormalize("obj_states", states)
                    # baseline-corrected EE pose: only the action-dependent
                    # displacement on top of the true current state
                    eff = states - base[t].unsqueeze(0) + now_t.unsqueeze(0)
                    ee_x, ee_y = eff[:, ee_idx, 0], eff[:, ee_idx, 1]
                    # approach: remaining distance to the first-touch point
                    # (the learned model underestimates EE displacement, so
                    # the push prior alone gives no gradient until contact
                    # already happens within the imagination horizon)
                    d_app = torch.relu(contact_x - ee_x)
                    # monotone push prior: the target can only be pushed
                    # right, and only as far as the EE contact front
                    # (minus the slip margin)
                    pushed = torch.maximum(pushed, ee_x + standoff - slip)
                    # projected resting position: rolling targets keep
                    # coasting after release (measured ~40% of push dist)
                    proj = pushed + torch.clamp(
                        coast_frac * (pushed - tgt_x0), max=0.15)
                    d_goal = (proj - goal_x).abs()
                    # flat push: keep the EE level with the target center
                    d_align = (ee_y - tgt_y).abs()
                    cost = cost + d_goal + self.approach_cost * d_app \
                        + self.align_cost * d_align \
                        + self.action_cost * (a[:, t] ** 2).mean(dim=-1)
                # terminal costs: goal distance + approach carry-over
                proj = pushed + torch.clamp(
                    coast_frac * (pushed - tgt_x0), max=0.15)
                cost = cost + 2.0 * (proj - goal_x).abs() \
                    + self.approach_cost * torch.relu(contact_x - ee_x)

            idx = torch.argsort(cost)[: self.E]
            elite = seqs[idx.cpu().numpy()]
            self.mean = elite.mean(axis=0)
            self.std = elite.std(axis=0) + 1e-3
            best_seq = elite[0]

        # Warm start for the next replan: shift by one step
        self.mean[:-1] = self.mean[1:]
        self.mean[-1] = 0.0
        self.std = np.maximum(self.std * 0.9, (self.high - self.low) * 0.05)
        return best_seq[0].astype(np.float32)


class PushToGoalTask:
    """Push a target object to a goal position with the learned world model."""

    def __init__(self, cfg, model, norm, device="cpu",
                 tol: float = 0.08, budget: int = 150, planner_kwargs: dict = None,
                 stride: int = 1):
        self.cfg = cfg
        self.env = PushSceneEnv(cfg)
        self.model = model
        self.device = device
        self.tol = tol
        self.budget = budget              # in simulation frames
        self.stride = stride              # frames per planned action (action repeat)
        self.planner = CEMPlanner(model, norm, device=device,
                                  force_max=cfg.collect.force_max,
                                  **(planner_kwargs or {}))
        # Target object picked per episode: the object nearest the EE spawn
        # (the pile's edge facing the EE), so the approach is a clean direct
        # push into free space. EE is index 0.
        self.target_idx = None
        self.ee_idx = 0

    def _half_width(self, idx: int) -> float:
        sc = self.cfg.scene
        t = int(self.env.obj_types[idx])
        if t == 1:      # box
            return sc.box_ext[0]
        if t == 2:      # ball
            return sc.ball_radius
        return sc.ee_radius

    def _free_run(self, obs, target_idx: int) -> float:
        """How far the target center can travel right before its surface
        touches the next object ahead (center gap minus both half widths)."""
        states = obs["obj_states"]
        tx = float(states[target_idx, 0])
        hw = self._half_width(target_idx)
        ahead = [(float(states[j, 0]), self._half_width(j))
                 for j in range(1, len(states))
                 if j != target_idx and states[j, 0] > tx + 0.05]
        if ahead:
            nx, nhw = min(ahead)
            return max(nx - tx - hw - nhw, 0.0)
        return 0.6

    def _pick_target(self, obs) -> int:
        """The LEFTMOST object. The EE spawns left of the whole pile, so
        the leftmost object has a clear approach corridor by construction
        and can be pushed directly without traversing the pile."""
        xs = obs["obj_states"][1:, 0]
        return 1 + int(np.argmin(xs))

    def _sample_goal(self, rng, obs, target_idx):
        """Goal to the RIGHT of the target (the EE spawn side is behind the
        object, so a direct push works and no detour is needed). The push
        distance is limited by the free run to the next object ahead so the
        goal stays physically reachable."""
        states = obs["obj_states"]
        target_x = float(states[target_idx, 0])
        # floor 0.10 > tol 0.08 so the episode can never start "already
        # successful" -- a real push of at least 0.02 m is always required
        dist = float(np.clip(self._free_run(obs, target_idx) - 0.02, 0.10, 0.30))
        goal_x = target_x + dist
        goal_y = float(states[target_idx, 1])  # resting height stays the same
        return np.array([goal_x, goal_y], dtype=np.float32)

    def run_episode(self, rng: np.random.Generator, record: bool = False):
        obs = self.env.reset(rng)
        self.target_idx = self._pick_target(obs)
        goal = self._sample_goal(rng, obs, self.target_idx)
        target_pos = obs["obj_states"][self.target_idx, :2]
        self.planner.reset()

        frames = {"obj_states": [obs["obj_states"]],
                  "contact_mask": [obs["contact_mask"]],
                  "actions": []} if record else None
        init_dist = float(np.linalg.norm(target_pos - goal))
        final_dist = init_dist
        success = False
        settle_frame = self.budget

        t = 0

        def hold_zero():
            """Zero force for one planning stride (momentum/settle pause)."""
            nonlocal obs, t
            self.env.set_force((0.0, 0.0))
            for _ in range(self.stride):
                obs = self.env.step(np.zeros(2, dtype=np.float32))
                t += 1
                if t >= self.budget:
                    break
            if record:
                frames["obj_states"].append(obs["obj_states"])
                frames["contact_mask"].append(obs["contact_mask"])
                frames["actions"].append(np.zeros(2, dtype=np.float32))

        while t < self.budget:
            cur = self.env._observe()  # fresh observation (same as last step's obs)
            target_pos = cur["obj_states"][self.target_idx, :2]
            target_vel = np.linalg.norm(cur["obj_states"][self.target_idx, 3:5])
            d = float(np.linalg.norm(target_pos - goal))
            final_dist = d
            if d < self.tol and target_vel < 0.08:
                success = True
                settle_frame = t
                self.env.set_force((0.0, 0.0))
                break
            is_box = int(self.env.obj_types[self.target_idx]) == 1
            if is_box:
                # Boxes are pushed with a force just above their max slide
                # force, which on a light / low-friction box means up to
                # ~6 m/s^2 of acceleration. The kinematic push prior is
                # velocity-blind, so built-up momentum can carry the box
                # far past the goal. Two velocity-gated pauses fix that:
                # - momentum gate: while the box moves faster than 0.25 m/s,
                #   hold zero force and let friction bleed the kinetic energy
                #   (bounded overshoot v^2 / (2 mu g) <= ~0.01 m)
                # - settle window: inside the tolerance, stop pushing at
                #   any speed below the gate -- friction stops a 0.25 m/s
                #   box within ~1 cm, while "topping off" a nearly-arrived
                #   box only adds kick + coast momentum (and pressing it
                #   keeps PGS contact jitter above the success speed check)
                if target_vel > 0.25 or d < self.tol:
                    hold_zero()
                    continue
            elif d < 0.75 * self.tol:
                # Balls roll smoothly and the release-coast prior already
                # stops the push early; inside 0.75*tol a zero-force glide
                # lets rolling resistance finish the job
                hold_zero()
                continue

            z0 = self.planner.encode_obs(cur)
            # center distance between EE and target at contact
            standoff = self._half_width(self.target_idx) \
                + self.cfg.scene.ee_radius
            # push parameters by target type. Boxes (0.24 x 0.12, CoM at
            # 0.06) are pushed at the EE floor height with a force cap
            # between their max slide force (mu*m*g <= 2.5 N) and min tip
            # force (m*g*0.12/0.102 >= 3.54 N), so they always slide and
            # never tip. Both types get an intentionally overestimated
            # release coast (see below).
            ee_floor = self.cfg.scene.ee_radius + 0.005
            if int(self.env.obj_types[self.target_idx]) == 1:      # box
                contact_y = ee_floor
                slip = 0.02        # near-zero lag: pushed boxes track the
                                  # contact front (force is above the slide
                                  # threshold, so they do not trail it)
                coast = 0.1        # small residual release coast: the
                                  # momentum gate caps the box speed at
                                  # 0.25 m/s, so the true coast after
                                  # release is <= ~1 cm (v^2 / 2 mu g).
                                  # A larger prior coast stops the push
                                  # short of the goal and the box stalls
                                  # just outside the tolerance
                fcap = 2.7         # slide-without-tip window, close to the
                                  # max slide force: slow push, little
                                  # release momentum
            else:                                                  # ball
                contact_y = max(float(goal[1]), ee_floor)
                slip = 0.0         # rolling: stays at the contact front
                coast = 0.9        # strongly overestimate release rolling
                                  # (clamped to 0.15 m): a pushed ball
                                  # accelerates hard (low mass, Coulomb
                                  # rolling resistance), so stopping the
                                  # push early and letting it roll in is
                                  # the only reliable way to land on the
                                  # goal; undershoot is recoverable, over-
                                  # shoot is not (push-right-only prior)
                fcap = 3.0         # gentle push: the approach kick plus
                                  # release momentum must stay below the
                                  # modeled coast or the ball overshoots
                                  # unrecoverably
            action = self.planner.plan(z0, self.ee_idx, self.target_idx, goal,
                                       standoff, contact_y, slip, coast, fcap,
                                       states_now=cur["obj_states"])
            # Hold the planned action for `stride` frames (matches the
            # frame-skipped dynamics the world model was trained on)
            for _ in range(self.stride):
                obs = self.env.step(action)
                t += 1
                if t >= self.budget:
                    break
            if record:
                frames["obj_states"].append(obs["obj_states"])
                frames["contact_mask"].append(obs["contact_mask"])
                frames["actions"].append(action)

        result = {
            "success": bool(success),
            "target_idx": int(self.target_idx),
            "init_dist": init_dist,
            "final_dist": final_dist,
            "settle_frame": settle_frame,
            "goal": goal.tolist(),
            "frames": (np.stack(frames["obj_states"]),
                       np.stack(frames["contact_mask"]),
                       np.stack(frames["actions"])) if record else None,
        }
        return result
