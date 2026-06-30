from definitions import *
from numericaldomain import DomainBase
import numpy as np
import taichi as ti


@ti.data_oriented
class SpringMassDomain(DomainBase):
    def __init__(
        self,
        d,
        coords,
        conns,
        prop=[100.0, 1.0, 1.0],
        bcs=[],
        considerContact=True,
        considerGroundContact=True,
        initials=[],
        friction=0.0,
    ):
        self.d = d
        self.prop = prop
        self.bcs = bcs
        self.initials = initials
        self.friction = friction
        self.considerContact = considerContact
        self.considerGroundContact = considerGroundContact
        self.type = DomainType.SPRINGMASS
        self.spring, self.damping, self.mass = prop
        self.massInv = 1.0 / self.mass
        self.coords = np.array(coords, dtype=np.float32)
        self.connectivity = np.array(conns, dtype=np.int32)

        self.nnodes = coords.shape[0]
        self.nelements = conns.shape[0]
        self.restLength = np.zeros((self.nelements,), dtype=np.float32)

        self.caclulateRestLength_()
        self.stableTime = 0.5 * np.sqrt(self.mass / self.spring)  # 2.0 * 0.25, 0.25 is the safety scale factor

        self.category_bits = COLLISION_CATEGORY_FEM
        self.collide_bits = COLLISION_MASK_ALL

    def caclulateRestLength_(self):
        for i in range(self.nelements):
            conn = self.connectivity[i]
            ia, ib = conn
            xab = self.coords[ia] - self.coords[ib]
            lold = np.linalg.norm(xab)
            self.restLength[i] = lold

    def attach(self, femManager, domainIdx):
        self.femManager = femManager
        self.domainIdx = domainIdx

    def getCurrentCoords(self):
        ndStart = self.femManager.domainNodeOffset[self.domainIdx]
        ndEnd = ndStart + self.nnodes
        return self.femManager.coords.to_numpy()[ndStart:ndEnd, :]

    def getBBox(self):
        aabb = self.getBBoxKernel2D()

        lb = ti.Vector([aabb[0, k] for k in ti.static(range(self.d))])
        ub = ti.Vector([aabb[1, k] for k in ti.static(range(self.d))])
        return lb, ub

    def getBBoxKernel2D(self):
        aabb = np.zeros((2, 2), dtype=np.float32)
        aabb[0, :] = np.min(self.coords, axis=0)
        aabb[1, :] = np.max(self.coords, axis=0)

        return aabb

    def getBBoxKernel3D(self):
        aabb = np.zeros((2, 3), dtype=np.float32)
        aabb[0, :] = np.min(self.coords, axis=0)
        aabb[1, :] = np.max(self.coords, axis=0)

        return aabb
