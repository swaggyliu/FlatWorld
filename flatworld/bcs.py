import abc
from definitions import *


class BcBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass


class Gravity(BcBase):
    def __init__(self, g):
        self.g = g
        self.type = BoundaryConditionType.GRAVITY

    def processData(self):
        return GRAVITY, None, self.g


class Force(BcBase):
    def __init__(self, nds, force):
        self.nds = nds
        self.force = force
        self.type = BoundaryConditionType.FORCE

    def processData(self):
        return FORCETYPE, self.nds, self.force


class EnforceAcc(BcBase):
    def __init__(self, nds, acc):
        self.nds = nds
        self.acc = acc
        self.type = BoundaryConditionType.ENFORCE_ACCELERATION

    def processData(self):
        return ATYPE, self.nds, self.acc


class EnforceVel(BcBase):
    def __init__(self, nds, vel):
        self.nds = nds
        self.vel = vel
        self.type = BoundaryConditionType.ENFORCE_VELOCITY

    def processData(self):
        return VTYPE, self.nds, self.vel


class Fixed(BcBase):
    def __init__(self, nds):
        self.nds = nds
        self.type = BoundaryConditionType.FIXED

    def processData(self):
        return UTYPE, self.nds, None


class FixedAll(BcBase):
    """A BC that fixes both translation and rotation"""

    def __init__(self, nds):
        self.nds = nds
        self.type = BoundaryConditionType.FIXED_ALL

    def processData(self):
        return RTYPE, self.nds, None


class EnforceRotVel(BcBase):
    def __init__(self, nds, rotVelocity, origin=[0.0, 0.0, 0.0]):
        self.nds = nds

        self.rotVel = rotVelocity
        self.origin = origin
        self.type = BoundaryConditionType.ENFORCE_ROTATION_VELOCITY

    def processData(self):
        return ROTVTYPE, self.nds, self.rotVel


class EnforceRotAcc(BcBase):
    def __init__(self, nds, rotAcc, origin=[0.0, 0.0, 0.0]):
        self.nds = nds
        self.origin = origin
        self.rotAcc = rotAcc
        self.type = BoundaryConditionType.ENFORCE_ROTATION_ACCELERATION

    def processData(self):
        return ROTATYPE, self.nds, self.rotAcc


class Torque(BcBase):
    def __init__(self, nds, torque):
        self.nds = nds
        self.torque = torque
        self.type = BoundaryConditionType.TORQUE

    def processData(self):
        return TORQUETYPE, self.nds, self.torque
