import abc
from definitions import *
import taichi as ti


class IntialConditionBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass


@ti.data_oriented
class InitialVel(IntialConditionBase):
    def __init__(self, nds, vel):
        # Support "ALL" keyword for applying to all nodes
        if nds == "ALL":
            self.is_all_nodes = True
            self.nds = None
        else:
            self.is_all_nodes = False
            numNds = len(nds)
            self.nds = ti.field(ti.i32, numNds)
            for i in range(numNds):
                self.nds[i] = nds[i]
        self.vel = ti.Vector(vel)
        self.type = VTYPE

    @ti.kernel
    def update(self, V: ti.template(), dim: ti.int32, offset: ti.int32, num_nodes: ti.int32):
        if ti.static(self.is_all_nodes):
            for i in range(num_nodes):
                V[i + offset] = self.vel
        else:
            for i in range(self.nds.shape[0]):
                nid = self.nds[i]
                V[nid + offset] = self.vel


@ti.data_oriented
class InitialAngVel(IntialConditionBase):
    def __init__(self, nds, ang_vel, ref_point=None):
        # Support "ALL" keyword for applying to all nodes
        if nds == "ALL":
            self.is_all_nodes = True
            self.nds = None
        else:
            self.is_all_nodes = False
            numNds = len(nds)
            self.nds = ti.field(ti.i32, numNds)
            for i in range(numNds):
                self.nds[i] = nds[i]
        self.ang_vel = ti.Vector(ang_vel)
        self.ref_point = ti.Vector(ref_point) if ref_point is not None else None
        self.type = ROTVTYPE

    @ti.kernel
    def update(self, RotV: ti.template(), dim: ti.int32, offset: ti.int32, num_nodes: ti.int32):
        # Set initial angular velocity for rigid bodies
        # This would typically be applied differently for rigid vs FEM
        if ti.static(self.is_all_nodes):
            for i in range(num_nodes):
                RotV[i + offset] = self.ang_vel
        else:
            for i in range(self.nds.shape[0]):
                nid = self.nds[i]
                RotV[nid + offset] = self.ang_vel
