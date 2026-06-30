from bcs import EnforceAcc, EnforceRotAcc, EnforceRotVel, EnforceVel, Force, Torque
from definitions import JointType
import math
import numpy as np
import taichi as ti


@ti.data_oriented
class JointBase:
    """Base class for joints (lightweight placeholder)."""

    def __init__(self, id_a: int, id_b: int, anchor, bcs, name=""):
        self.id_a = id_a
        self.id_b = id_b
        self.anchor = anchor
        self.bcs = bcs
        self.has_motor = 0
        self.name = name
        self.jointType = None
        self.axis = None
        self.limits = [-1e10, 1e10]
        self.velocity_limit = 1e10
        self.effort_limit = 1e10
        self.stiff = 0.0
        self.damping = 0.0
        self.position_bias = 0.0
        self.angular_bias = 0.0
        self.motor_target_position = None

    def attach(self, manager):
        # Register or lookup a dynamic anchor in the manager
        self.manager = manager

        # compute body-local rest offsets from the initial anchor position
        self.id_a = manager.domainToRigid[self.id_a]
        self.id_b = manager.domainToRigid[self.id_b]

        # Compute world-space offsets
        l1_world = manager.rigidParams[self.id_a, 0] - self.anchor
        l2_world = manager.rigidParams[self.id_b, 0] - self.anchor

        # Store local offsets for subclass registration
        self.l1 = l1_world
        self.l2 = l2_world
        # Safety check for anchor/joint capacity
        if manager.numAnchors >= manager.MAX_JOINTS:
            print(
                f"\033[91mError: Exceeded MAX_JOINTS ({manager.MAX_JOINTS})! Current count: {manager.numAnchors}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum anchors/joints ({manager.MAX_JOINTS})")

        # Allocate anchor index in the tail region
        joint_idx = manager.numAnchors

        # Register joint using the anchor
        q0a = manager.quat[self.id_a].to_numpy()
        q0b = manager.quat[self.id_b].to_numpy()
        manager.joint_anchor[joint_idx] = self.anchor
        # Store θ_a0 - θ_b0 so angle_error = (θ_a - θ_a0) - (θ_b - θ_b0) = 0 at rest.
        manager.joint_q0_rel_inv[joint_idx] = q0a - q0b
        manager.joint_type[joint_idx] = self.jointType
        manager.joint_id_a[joint_idx] = self.id_a
        manager.joint_id_b[joint_idx] = self.id_b
        manager.joint_l1[joint_idx] = self.l1
        manager.joint_l2[joint_idx] = self.l2
        manager.joint_axis[joint_idx] = self.axis if self.axis is not None else ti.Vector([0.0] * manager.d)
        manager.kpd_field[joint_idx] = ti.Vector([self.stiff, self.damping])
        manager.joint_params[joint_idx] = ti.Vector(
            [
                self.position_bias,
                self.angular_bias,
                self.limits[0],
                self.limits[1],
                self.velocity_limit,
                self.effort_limit,
            ]
        )
        manager.kpd_field[joint_idx] = ti.Vector([self.stiff, self.damping])
        manager.joint_has_motor[joint_idx] = 1 if self.has_motor else 0
        manager.joint_control_target[joint_idx] = (
            self.motor_target_position if self.motor_target_position is not None else 0.0
        )
        manager.joint_motor_target_mode[joint_idx] = 0
        manager.joint_motor_target_vel[joint_idx] = 0.0

        self.anchor_id = joint_idx

    def draw(self, gui, color=0xFF0000, resolution=50):
        """Return simple debug geometry for visualization.

        Returns a dict with:
          - anchor: world-space anchor position (ti.Vector)
        """
        aid = self.anchor_id if hasattr(self, "anchor_id") else -1
        gui.circle(self.manager.rigidParams[aid, 0].to_numpy()[:2], radius=resolution * 0.01, color=color)
        anchor = self.getCurrentAnchorPoint()
        gui.line(self.manager.rigidParams[self.id_a, 0].to_numpy()[:2], anchor[:2], radius=5, color=0x0000AA)
        gui.line(self.manager.rigidParams[self.id_b, 0].to_numpy()[:2], anchor[:2], radius=5, color=0x0000AA)

    def getCurrentAnchorPoint(self):
        posa = self.manager.rigidParams[self.id_a, 0].to_numpy()
        rota = self.manager.quat[self.id_a].to_numpy()
        initial_rota = self.manager.quat_initial[self.id_a].to_numpy()
        theta = rota[0] - initial_rota[0]
        r_rel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        anchor = posa - r_rel @ self.l1.to_numpy()

        return anchor


@ti.data_oriented
class RevoluteJoint(JointBase):
    """A simple revolute joint that constrains two rigid reference points to a
    common anchor while allowing relative rotation about a given axis.

    This implementation is Taichi-friendly: use `solve(manager, ...)` to apply
    positional and angular constraint forces/torques to the provided
    `RigidManager` instance. The routine handles 2D and 3D manager layouts.
    """

    def __init__(
        self,
        id_a: int,
        id_b: int,
        anchor,
        axis,
        bcs=[],
        limits=None,
        velocity_limit=1e10,
        effort_limit=1e10,
        position_bias=0.2,
        pgs_iters=5,
        stiff=100.0,
        damping=1.0,
        name="",
        rpy=None,
        motor_target_position=None,
    ):
        """Create a revolute joint between two rigid indices.

        - id_a, id_b: integer rigid indices in the `RigidManager`.
        - anchor: world-space anchor point (sequence of length d).
        - axis: rotation axis (sequence of length d). For 2D use [0,0].
        - stiff: angular stiffness (default 100.0).
        - damping: angular damping (default 1.0).
        - limits: optional angular limits [lower, upper] (radians).
        - velocity_limit: maximum angular velocity (rad/s).
        - effort_limit: maximum torque (Nm).
        - rpy: joint origin roll-pitch-yaw [r, p, y] (radians), default [0,0,0].
        - motor_target/motor_torque: optional motor parameters.
        """
        super().__init__(id_a, id_b, anchor, bcs, name)
        self.id_a = int(id_a)
        self.id_b = int(id_b)
        self.jointType = JointType.Revolute
        # store as python lists; kernels will construct ti.Vector at call-time
        self.axis = axis
        self.rpy = list(rpy) if rpy is not None else [0.0, 0.0, 0.0]
        if limits is not None:
            self.limits = limits
        else:
            self.limits = [-1e10, 1e10]

        self.velocity_limit = float(velocity_limit)
        self.effort_limit = float(effort_limit)

        self.position_bias = position_bias
        self.pgs_iters = pgs_iters
        self.stiff = stiff
        self.damping = damping
        self.motor_target_position = motor_target_position
        self.has_motor = 0
        if len(bcs) > 0 or self.motor_target_position is not None:
            self.has_motor = 1
        # print(f"Creating RevoluteJoint {name} between {self.id_a} and {self.id_b} with anchor {anchor} and axis {axis}")
        # print("RevoluteJoint with limits:", self.limits)
        # print("RevoluteJoint velocity limit:", self.velocity_limit)
        # print("RevoluteJoint effort limit:", self.effort_limit)
        # print("RevoluteJoint has motor:", self.has_motor)

    def attach(self, manager):
        """Attach joint and register it in manager's unified storage."""
        super().attach(manager)


@ti.data_oriented
class WeldJoint(JointBase):
    """Weld joint: fully constrain translation and rotation between two rigids.

    The joint computes body-local anchors via `attach(manager)` and applies
    positional and angular penalty forces/torques in `_apply` to drive the
    two attachment frames to coincide (both position and orientation).
    """

    def __init__(
        self, id_a: int, id_b: int, anchor, bcs=[], position_bias: float = 0.2, angular_bias: float = 0.1, name=""
    ):
        """Create a weld joint (fully constrained).

        - angular_bias: Baumgarte stabilization for orientation (0.05-0.2 recommended)
        """
        super().__init__(id_a, id_b, anchor, bcs, name)
        self.position_bias = float(position_bias)
        self.angular_bias = float(angular_bias)
        self.jointType = JointType.Weld

    def attach(self, manager):
        """Attach joint and register it in manager's unified storage."""
        super().attach(manager)


@ti.data_oriented
class PrismaticJoint(JointBase):
    """Prismatic joint: allows translation only along a given axis between two rigids.

    Construct with body-local offsets `l1`, `l2` and an `axis` (length d). The
    kernel `_apply(manager)` enforces coincidence of the two attachment points
    in directions perpendicular to `axis` while allowing sliding along it. All
    rotations are locked (penalty angular locking).
    """

    def __init__(
        self,
        id_a: int,
        id_b: int,
        anchor,
        axis,
        bcs=[],
        limits=None,
        velocity_limit=1e10,
        effort_limit=1e10,
        position_bias: float = 0.2,
        angular_bias: float = 0.2,
        stiff: float = 100.0,
        damping: float = 1.0,
        name="",
        motor_target_position=None,
    ):
        super().__init__(id_a, id_b, anchor, bcs, name)
        self.axis = axis
        self.position_bias = float(position_bias)
        self.angular_bias = float(angular_bias)
        self.stiff = float(stiff)
        self.damping = float(damping)
        self.motor_target_position = motor_target_position
        self.jointType = JointType.Prismatic
        if limits is not None:
            self.limits = limits
        else:
            self.limits = [-1e10, 1e10]

        self.velocity_limit = float(velocity_limit)
        self.effort_limit = float(effort_limit)

        self.has_motor = 0
        if len(bcs) > 0 or self.motor_target_position is not None:
            self.has_motor = 1

    def attach(self, manager):
        """Attach joint and register it in manager's unified storage."""
        super().attach(manager)


@ti.data_oriented
class SphericalJoint(JointBase):
    """Spherical joint: constrains only translation DOFs (attachment points coincide).

    The joint computes body-local anchor offsets with `attach(manager)` and
    applies a positional penalty force in `_apply`. Rotation between bodies
    is left free (no angular penalty), but torques resulting from the
    positional force (lever arm) are applied to the bodies for correctness.
    """

    def __init__(
        self,
        id_a: int,
        id_b: int,
        anchor,
        bcs=[],
        limits=None,
        position_bias: float = 0.2,
        stiff: float = 100.0,
        damping: float = 1.0,
        name="",
    ):
        super().__init__(id_a, id_b, anchor, bcs, name)
        self.position_bias = float(position_bias)
        self.stiff = float(stiff)
        self.damping = float(damping)
        self.has_motor = 0
        self.jointType = JointType.Spherical
        if len(bcs) > 0:
            self.has_motor = 1

        if limits is not None:
            self.limits = limits
        else:
            self.limits = [-1e10, 1e10]

    def attach(self, manager):
        """Attach joint and register it in manager's unified storage."""
        super().attach(manager)
