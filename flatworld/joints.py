from definitions import JointType
import numpy as np


def _to_np(x):
    """Host copy of a Warp (.numpy) or Taichi (.to_numpy) array element/buffer."""
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        return x.numpy()
    if hasattr(x, "to_numpy") and callable(getattr(x, "to_numpy")):
        return x.to_numpy()
    return np.asarray(x)


def _vec_np(v):
    return np.asarray(_to_np(v), dtype=np.float32).reshape(-1)


def _patch_array(arr, index, value):
    """Host write for Warp arrays (no ``arr[i] =`` from Python on Warp 1.14+)."""
    if hasattr(arr, "numpy") and hasattr(arr, "assign"):
        np_arr = arr.numpy()
        np_arr[index] = value
        arr.assign(np_arr)
    else:
        arr[index] = value


class JointBase:
    """Base class for joints (lightweight placeholder)."""

    def __init__(self, id_a: int, id_b: int, anchor, bcs, name=""):
        self.id_a = id_a
        self.id_b = id_b
        self.anchor = np.asarray(anchor, dtype=np.float32)
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
        d2r = _to_np(manager.domainToRigid)
        self.id_a = int(d2r[self.id_a])
        self.id_b = int(d2r[self.id_b])

        # Compute world-space offsets (host reads via .numpy() for Warp arrays)
        params = _to_np(manager.rigidParams)
        l1_world = _vec_np(params[self.id_a, 0]) - self.anchor
        l2_world = _vec_np(params[self.id_b, 0]) - self.anchor

        # Store local offsets for subclass registration
        self.l1 = np.asarray(l1_world, dtype=np.float32)
        self.l2 = np.asarray(l2_world, dtype=np.float32)
        # Safety check for anchor/joint capacity
        if manager.numAnchors >= manager.MAX_JOINTS:
            print(
                f"\033[91mError: Exceeded MAX_JOINTS ({manager.MAX_JOINTS})! Current count: {manager.numAnchors}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum anchors/joints ({manager.MAX_JOINTS})")

        # Allocate anchor index in the tail region
        joint_idx = manager.numAnchors

        # Register joint using the anchor
        quat_np = _to_np(manager.quat)
        q0a = float(np.asarray(quat_np[self.id_a]).reshape(-1)[0])
        q0b = float(np.asarray(quat_np[self.id_b]).reshape(-1)[0])
        _patch_array(manager.joint_anchor, joint_idx, self.anchor)
        # Store θ_a0 - θ_b0 so angle_error = (θ_a - θ_a0) - (θ_b - θ_b0) = 0 at rest.
        _patch_array(manager.joint_q0_rel_inv, joint_idx, q0a - q0b)
        _patch_array(manager.joint_type, joint_idx, self.jointType)
        _patch_array(manager.joint_id_a, joint_idx, self.id_a)
        _patch_array(manager.joint_id_b, joint_idx, self.id_b)
        _patch_array(manager.joint_l1, joint_idx, self.l1)
        _patch_array(manager.joint_l2, joint_idx, self.l2)
        axis = self.axis if self.axis is not None else np.zeros((manager.d,), dtype=np.float32)
        _patch_array(manager.joint_axis, joint_idx, np.asarray(axis, dtype=np.float32))
        _patch_array(manager.kpd_field, joint_idx, [self.stiff, self.damping])
        _patch_array(
            manager.joint_params,
            joint_idx,
            [
                self.position_bias,
                self.angular_bias,
                self.limits[0],
                self.limits[1],
                self.velocity_limit,
                self.effort_limit,
            ],
        )
        _patch_array(manager.joint_has_motor, joint_idx, 1 if self.has_motor else 0)
        _patch_array(
            manager.joint_control_target,
            joint_idx,
            self.motor_target_position if self.motor_target_position is not None else 0.0,
        )
        _patch_array(manager.joint_motor_target_mode, joint_idx, 0)
        _patch_array(manager.joint_motor_target_vel, joint_idx, 0.0)

        self.anchor_id = joint_idx

    def draw(self, gui, color=0xFF0000, resolution=50):
        """Debug draw for joints.

        When the manager uses Warp arrays, rigidParams/quat must be read via
        ``.numpy()`` (handled by ``_vec_np`` / ``_to_np``).
        """
        anchor = self.getCurrentAnchorPoint()
        gui.circle(anchor[:2], radius=resolution * 0.01, color=color)
        params = _to_np(self.manager.rigidParams)
        ref_a = _vec_np(params[self.id_a, 0])
        ref_b = _vec_np(params[self.id_b, 0])
        gui.line(ref_a[:2], anchor[:2], radius=5, color=0x0000AA)
        gui.line(ref_b[:2], anchor[:2], radius=5, color=0x0000AA)

    def getCurrentAnchorPoint(self):
        # Host numpy reads — call .numpy() on Warp buffers when manager is migrated
        params = _to_np(self.manager.rigidParams)
        quat_np = _to_np(self.manager.quat)
        quat_init_np = _to_np(self.manager.quat_initial)
        posa = _vec_np(params[self.id_a, 0])
        rota = float(np.asarray(quat_np[self.id_a]).reshape(-1)[0])
        initial_rota = float(np.asarray(quat_init_np[self.id_a]).reshape(-1)[0])
        theta = float(rota - initial_rota)
        r_rel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        l1 = np.asarray(self.l1, dtype=np.float32)
        anchor = posa - r_rel @ l1

        return anchor


class RevoluteJoint(JointBase):
    """A simple revolute joint that constrains two rigid reference points to a
    common anchor while allowing relative rotation about a given axis.

    Use `attach(manager)` to register the joint on a `RigidManager`.
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

    def attach(self, manager):
        """Attach joint and register it in manager's unified storage."""
        super().attach(manager)


class WeldJoint(JointBase):
    """Weld joint: fully constrain translation and rotation between two rigids.

    The joint computes body-local anchors via `attach(manager)`.
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


class PrismaticJoint(JointBase):
    """Prismatic joint: allows translation only along a given axis between two rigids."""

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


class SphericalJoint(JointBase):
    """Spherical joint: constrains only translation DOFs (attachment points coincide)."""

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
