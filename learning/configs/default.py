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

    # Objects
    num_boxes: int = 3
    box_ext: tuple = (0.12, 0.12)    # box half-extents (half_w, half_h)
    box_mass: float = 0.5
    num_balls: int = 2
    ball_radius: float = 0.08
    ball_mass: float = 0.3

    # Object placement region (along x); y determined by object half-height (resting on ground)
    # Wide enough to hold all objects at min_gap PLUS a reserved free run
    # ahead of the leftmost object (task reachability), see the layout code.
    area_x: tuple = (0.40, 1.95)
    min_gap: float = 0.06            # minimum gap between objects
    first_gap: float = 0.24          # reserved free run ahead of the leftmost object
    spawn_drop: float = 0.005        # small drop margin to avoid initial penetration
    ee_spawn_x: tuple = (0.10, 0.26)  # EE spawn range (left of all objects)

    # Domain randomization (Phase 2): multiplicative mass factor and friction
    # are re-sampled per episode reset so the world model must learn the
    # general dynamics instead of memorizing one parameter setting.
    # Ranges are sized so the 2.9 N box force cap always slides any box
    # BELOW its tipping threshold (boxes are 0.24 x 0.12, contact at EE
    # height 0.102): max slide force mu*m*g = 0.5*0.5*10 = 2.5 N < 2.9 N
    # cap < min tip force mg*0.12/0.102 = 3.54 N.
    dr_enabled: bool = True
    dr_mass_range: tuple = (0.6, 1.0)      # factor applied to base masses (objects)
    dr_ee_mass_range: tuple = (0.8, 1.2)   # milder range for the end-effector
    dr_friction_range: tuple = (0.3, 0.5)
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
    num_rollouts: int = 100
    episode_len: int = 200           # number of action frames (states stored: T+1)
    seed: int = 0
    force_max: float = 6.0           # force magnitude clamp (N)
    ou_theta: float = 0.85           # OU inertia (larger = smoother trajectory)
    ou_sigma: float = 1.5            # OU noise intensity
    attract_gain: float = 6.0        # N/m, attraction toward the nearest object
    out_dir: str = "learning/data/rollouts"


@dataclass
class Config:
    scene: SceneConfig = field(default_factory=SceneConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)

    def dump(self) -> str:
        import json
        return json.dumps(asdict(self), indent=2)
