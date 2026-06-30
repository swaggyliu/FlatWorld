from definitions import *
from numericaldomain import DomainBase
import numpy as np
import taichi as ti


@ti.data_oriented
class FemDomain(DomainBase):

    def __init__(self, mesh, prop, bcs=[], considerContact=True, considerGroundContact=True, initials=[], friction=0.0):
        self.mesh = mesh
        self.d = mesh.d
        self.prop = prop
        self.bcs = bcs
        self.friction = friction
        self.initials = initials
        self.nnodes = self.mesh.numNodes
        self.nelements = self.mesh.numElements
        self.considerContact = considerContact
        self.considerGroundContact = considerGroundContact
        self.type = DomainType.FEM
        self.category_bits = COLLISION_CATEGORY_FEM
        self.collide_bits = COLLISION_MASK_ALL

        mat = self.prop.mat
        self.stableTime = 0.4 * self.mesh.charLength / (mat.E / mat.rho) ** 0.5
        print("FEM Domain stable time step: ", self.stableTime)

    def getBBox(
        self,
    ):
        domain_idx = self.femManager.femDomainIds[self.domainIdx]
        return self.femManager.aabb[domain_idx, 0], self.femManager.aabb[domain_idx, 1]

    def attach(self, femManager, domainIdx):
        self.femManager = femManager
        self.domainIdx = domainIdx

    def getNumDofs(self):
        return self.d * self.mesh.numNodes

    def getNumNodes(self):
        return self.mesh.numNodes

    def getNumElements(self):
        return self.mesh.numElements

    def getCurrentCoords(self):
        ndStart = self.femManager.domainNodeOffset[self.domainIdx]
        ndEnd = ndStart + self.mesh.numNodes
        return self.femManager.coords.to_numpy()[ndStart:ndEnd, :]

    def draw(self, gui, color=0xFFFFFF):
        ndStart = self.femManager.domainNodeOffset[self.domainIdx]
        ndEnd = ndStart + self.mesh.numNodes
        pos = self.femManager.coords.to_numpy()[ndStart:ndEnd, :2]
        a, b, c = self.mesh.connectivity.T
        gui.triangles(pos[a], pos[b], pos[c], color=color)
