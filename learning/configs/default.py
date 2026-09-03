"""Phase 1 default configuration.

Uses Python dataclasses instead of YAML: zero extra dependencies for the
collection stage (no pyyaml needed). Can be migrated to YAML later once
torch is integrated for training.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class SceneConfig:
    """Push scene: circular end-effector + several boxes/balls + ground."""
    gravity: float = 10.0            # m/s^2, applied to objects only (EE floats, no gravity)
    friction: float = 0.5            # friction coefficient for all rigid bodies
    frame_dt: float = 1.0 / 60.0     # one action per visual frame

    # End-effector (circular "gripper")
    ee_radius: float = 0.1
    ee_mass: float = 1.0
    ee_height: float = 0.15          # EE hover height (roughly at object mid-height)
    # Admittance control: the commanded force acts against a viscous term
    # (-c * v) and the speed is hard-clamped. Without damping the EE is an
    # undamped double integrator -- contact solver impulses launch it
    # meters across the scene and the force -> motion statistics (and any
    # model trained on them) are garbage.
    ee_damping: float = 4.0          # N / (m/s)
    ee_vel_max: float = 2.0          # m/s hard cap on the EE speed

    # Objects
    num_boxes: int = 3
    # FLAT boxes: wide stance + low CoM so they slide instead of tipping
    # when pushed (a square box tips on spawn jitter and on any push).
    box_ext: tuple = (0.12, 0.06)    # box half-extents (half_w, half_h)
    box_mass: float = 0.5
    num_balls: int = 2
    ball_radius: float = 0.08
    ball_mass: float = 0.3

    # Object placement region (along x); y determined by object half-height (resting on ground)
    # Wide enough to hold all objects at min_gap PLUS a reserved free run
    # ahead of the leftmost object (task reachability), see the layout code.
    # Shared lineup for EE + objects (EE is inserted at a random rank).
    area_x: tuple = (0.16, 2.10)
    min_gap: float = 0.06            # minimum gap between bodies
    first_gap: float = 0.24          # unused leftover (kept for dump compat)
    spawn_drop: float = 0.005        # small drop margin to avoid initial penetration
    ee_spawn_x: tuple = (0.10, 0.26)  # fallback if rank placement is disabled
    # 1-indexed rank of the EE among (n_obj + 1) slots, inclusive.
    # With 5 objects this is "1st through 5th from the left".
    ee_rank_range: tuple = (1, 5)

    # Domain randomization (Phase 2): multiplicative mass factor and friction
    # are re-sampled per episode reset so the world model must learn the
    # general dynamics instead of memorizing one parameter setting.
    dr_enabled: bool = True
    dr_mass_range: tuple = (0.5, 1.4)
    dr_ee_mass_range: tuple = (0.8, 1.2)
    dr_friction_range: tuple = (0.2, 0.7)
    # Per-object size ranges (half-extents). Boxes stay wider than tall
    # so they slide rather than tip. Sampled in env.reset.
    dr_size_enabled: bool = True
    # Game scale slider is 0.5–2.0 on the base EE radius.
    dr_ee_r_scale: tuple = (0.5, 2.0)
    dr_box_hw: tuple = (0.08, 0.18)
    dr_box_hh: tuple = (0.04, 0.12)
    dr_ball_r: tuple = (0.05, 0.12)
    restitution: float = 0.05              # near-elastic contact makes light
                                           # boxes bounce forever on the ground
                                           # (PGS corner jitter); keep it low

    # Sensor noise (Phase 2): Gaussian noise added to observations
    noise_enabled: bool = True
    noise_pos: float = 0.002            # m, on positions
    noise_vel: float = 0.01             # m/s, on velocities
    noise_force: float = 0.05           # N, on contact forces / summary


@dataclass
class TactileConfig:
    """Variable-length contact storage parameters (Deep Sets scheme)."""
    max_contacts: int = 16           # per-frame padding limit K
    feat_dim: int = 7                # rel_pos(2) + normal(2) + force(2) + penetration(1)
    summary_dim: int = 4             # sum_Fx, sum_Fy, sum_tau, num_contacts


@dataclass
class CollectConfig:
    """Random collection parameters (OU noise + attraction toward objects)."""
    num_rollouts: int = 500
    episode_len: int = 200           # number of action frames (states stored: T+1)
    seed: int = 0
    force_max: float = 6.0           # force magnitude clamp (N)
    ou_theta: float = 0.85           # OU inertia (larger = smoother trajectory)
    ou_sigma: float = 1.5            # OU noise intensity
    attract_gain: float = 6.0        # N/m, attraction toward the nearest object
    attract_cap: float = 1.5         # N, attraction magnitude cap (an uncapped
                                     # attraction yanks the EE across the scene
                                     # and buries the force -> motion signal)
    # free-mode force sweep magnitude, fraction of force_max
    sweep_min: float = 0.15
    sweep_max: float = 1.0
    # soft workspace barriers shared by all collection modes (N/m gain and
    # bounds): keep the EE inside the working volume so the contact solver
    # cannot eject it to y > 1 m (out-of-distribution for the planner)
    barrier_k: float = 12.0
    x_lo: float = 0.08
    x_hi: float = 2.12
    y_lo: float = 0.105              # just above the ground (ee_radius margin)
    y_hi: float = 0.55
    # Hop may fly higher than the default band, then dive into contact.
    hop_y: float = 0.38
    hop_y_hi: float = 0.80
    out_dir: str = "learning/data/rollouts_500"
    # Collection-policy mix (must sum to 1). Demonstration only; the
    # planner never uses these ratios.
    mode_free: float = 0.15
    mode_attract: float = 0.15
    mode_push: float = 0.15
    mode_release: float = 0.10
    mode_hop: float = 0.10
    mode_pile: float = 0.35


@dataclass
class TaskConfig:
    """PushToGoal evaluation protocol (not CEM hyperparameters)."""
    tol: float = 0.08                # success: |target - goal| < tol
    vel_tol: float = 0.08            # and |v_xy| < vel_tol (m/s)
    budget: int = 400                # sim frames
    goal_min: float = 0.10           # min |goal_x - target_x|; > tol so not already done
    goal_max: float = 0.30           # max requested push distance


@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def dump(self) -> str:
        import json
        return json.dumps(asdict(self), indent=2)
