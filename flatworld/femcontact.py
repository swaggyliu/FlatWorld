from contact_detection import (
    detectPointToMeshBoundaries, pointToEdgeContact
)
from definitions import *
import taichi as ti
from utils import *


@ti.data_oriented
class ContactBase:
    def __init__(self, domain1, domain2, type, is_tied=False):
        """Create a Contact controller for two domains and a contact type."""
        self.domain1 = domain1
        self.domain2 = domain2
        self.penalty = 1.0
        self.type = type
        self.tied = is_tied

    @staticmethod
    def _bulk_modulus(mat):
        """Bulk modulus K = E / (3*(1 - 2*nu)), clamped for near-incompressible."""
        nu = min(mat.nu, 0.4999)
        return mat.E / (3.0 * (1.0 - 2.0 * nu))

    @staticmethod
    def _segment_penalty(dom, slsfac=1.0):
        """Compute per-segment penalty stiffness following LS-DYNA convention.

        k = slsfac * K * A_seg^2 / V_elem

        With A_tri = L^2/2, V_tet = L^3/6:
        - 2D:    A_edge=L, V_tri=L^2/2   =>  k = 2 * slsfac * K
        - Shell: A=L^2/2, V=A*t=L^2*t/2  =>  k = slsfac * K * L^2 / (2t)
        - Solid: A=L^2/2, V=L^3/6        =>  k = 3/2 * slsfac * K * L
        """
        mat = dom.prop.mat
        K = ContactBase._bulk_modulus(mat)
        L = dom.mesh.charAverageLength
        if dom.d == 2:
            # A_edge=L, V_tri=L^2/2 => A^2/V = 2
            return 2.0 * slsfac * K
        else:
            # A=L^2/2, V=L^3/6 => A^2/V = 3L/2
            return 1.5 * slsfac * K * L

    def calStableTime(self, penalty, femdomain):
        """Stable time step: dt = 0.9 * sqrt(m_min / k).

        m_min is approximated from lumped mass: rho * V_elem / n_nodes.
        """
        mat = femdomain.prop.mat
        L = femdomain.mesh.charLength
        if femdomain.d == 2:
            # 2D tri: area ~ L^2/2, 3 nodes
            m_node = mat.rho * (L * L * 0.5) / 3.0
        else:
            # 3D tet: volume ~ L^3/6, 4 nodes
            m_node = mat.rho * (L * L * L / 6.0) / 4.0

        stableTime = 0.9 * (m_node / penalty) ** 0.5
        print(
            f"[Contact] Contact stable time: {stableTime:.3e}, Penalty: {penalty:.3e}, "
            f"m_node: {m_node:.3e}, L: {L:.3e}, rho: {mat.rho:.3e}, "
            f"K: {self._bulk_modulus(mat):.3e}"
        )
        return stableTime

    @ti.kernel
    def update(self, dt: ti.f32):
        pass

    @ti.kernel
    def calculate(self, dt: ti.f32):
        pass


# ====================================================================
# Contact of different combinations!!!
# ====================================================================


@ti.data_oriented
class ContactFlexAnalytical(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        if self.domain1.type == DomainType.ANALYTICAL:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


@ti.data_oriented
class ContactFlexFlex(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        # Symmetric: take the more conservative (smaller) side penalty
        self.penalty = min(self._segment_penalty(self.domain1), self._segment_penalty(self.domain2)) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)
        self.stableTime = min(self.stableTime, self.calStableTime(self.penalty, self.domain2))


@ti.data_oriented
class ContactFlexRigid(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        if self.domain1.type == DomainType.RIGID:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)



@ti.data_oriented
class ContactFlexHeightField(ContactBase):
    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        if self.domain1.type == DomainType.HEIGHTFIELD:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


@ti.data_oriented
class ContactFlexVoxelMap(ContactBase):
    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        if self.domain1.type == DomainType.VOXELMAP:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


@ti.data_oriented
class ContactSpringFlex(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0  # fixed multiplier for spring contacts

    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        if self.domain1.type == DomainType.FEM:
            self.domain1, self.domain2 = self.domain2, self.domain1
        # Spring side: use spring constant directly
        # FEM side: use segment penalty
        penalty_fem = self._segment_penalty(self.domain2)
        penalty_spring = self.domain1.spring
        self.penalty = min(penalty_spring, penalty_fem) * self._SPRING_PENALTY_SCALE
        self.stableTime = self.calStableTime(self.penalty, self.domain2)

    @ti.func
    def _detect_node_mesh_contact(self, nodeCoord, domain):
        """Helper: detect contact between a point and mesh boundary elements by direct iteration.

        Returns: (found_contact, best_penetration, best_normal)
        """
        best_penetration = ti.f32(1e9)
        best_normal = ti.Vector.zero(ti.f32, ti.static(self.domain1.d))
        weights = ti.Vector.zero(ti.f32, ti.static(self.domain1.d))
        targetElconn = ti.Vector.zero(ti.i32, ti.static(self.domain1.d))

        found_contact = False
        target_mesh = domain.mesh
        elementOffset = domain.femManager.domainBoundaryElemOffset[domain.domainIdx]

        # Direct iteration over all boundary elements
        for j in range(target_mesh.numBoundElements):
            elem_id = j + elementOffset
            if elem_id < 0:
                break  # No more valid elements

            # Get element nodes
            elem_conn = domain.femManager.boundaryElements[elem_id]

            pen = ti.f32(1e9)
            normal = ti.Vector.zero(ti.f32, ti.static(self.domain1.d))
            cp = nodeCoord
            is_inside = False

            # 2D: edge contact
            n0 = domain.femManager.coords[elem_conn[0]]
            n1 = domain.femManager.coords[elem_conn[1]]
            pen, normal, cp, is_inside, weights = pointToEdgeContact(nodeCoord, n0, n1, self.domain1.d)

            if pen < best_penetration and ti.abs(pen) < 1.0 and is_inside:
                best_penetration = pen
                best_normal = normal
                targetElconn = elem_conn
                found_contact = True

        return found_contact, best_penetration, best_normal, weights, targetElconn

    @ti.kernel
    def calculate(self, dt: ti.f32):
        """Kernel: compute contact forces between SpringMass and FEM using direct iteration."""
        numLoops = self.domain1.nnodes
        if ti.abs(dt) < 1e-9:
            numLoops = 0

        # Get SpringMass node offset in femSpringManager
        node_offset = self.domain1.femManager.domainNodeOffset[self.domain1.domainIdx]
        friction_coeff = ti.max(self.domain1.friction, self.domain2.friction)

        # Domain1 (SpringMass) nodes against Domain2 (FEM) mesh
        for i in range(numLoops):
            global_node_idx = node_offset + i
            nodeCoord = self.domain1.femManager.coords[global_node_idx]
            node_vel = self.domain1.femManager.V[global_node_idx]

            found_contact, penetration, normal, weights, targetElconn = self._detect_node_mesh_contact(
                nodeCoord, self.domain2
            )

            if found_contact and penetration < 0.0:
                # Normal force
                normal_force = -normal * self.penalty * penetration
                total_force = normal_force

                # Friction force (simplified - relative to mesh surface)
                # Estimate contact surface velocity from element nodes
                if friction_coeff > 1e-9:
                    surf_vel = ti.Vector.zero(ti.f32, self.domain1.d)
                    surf_vel += self.domain2.femManager.V[targetElconn[0]] * weights[0]
                    surf_vel += self.domain2.femManager.V[targetElconn[1]] * weights[1]

                    relative_vel = node_vel - surf_vel
                    tangential_vel = relative_vel - relative_vel.dot(normal) * normal
                    friction_force = ti.Vector.zero(ti.f32, self.domain1.d)

                    if tangential_vel.norm() > 1e-9:
                        friction_dir = -tangential_vel.normalized()
                        friction_magnitude = friction_coeff * normal_force.norm()
                        friction_force = friction_dir * friction_magnitude
                    else:
                        friction_force = ti.Vector.zero(ti.f32, self.domain1.d)

                    # Apply total force
                    total_force = normal_force + friction_force

                # Apply force to domain1 (SpringMass)
                self.domain1.femManager.Fext[global_node_idx] += total_force

                # Apply equal and opposite force to domain2 (distributed to element nodes)
                self.domain2.femManager.Fext[targetElconn[0]] -= total_force * weights[0]
                self.domain2.femManager.Fext[targetElconn[1]] -= total_force * weights[1]


@ti.data_oriented
class ContactSpringRigid(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0

    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        # here we assume the second domain is the more rigid one
        if self.domain1.type == DomainType.RIGID:
            self.domain1, self.domain2 = self.domain2, self.domain1

        self.penalty = self.domain1.spring * self._SPRING_PENALTY_SCALE
        self.stableTime = 0.5 * (self.domain1.mass / self.penalty) ** 0.5


@ti.data_oriented
class ContactSpringAnalytical(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0

    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        # Ensure domain1 is Spring and domain2 is Analytical
        if self.domain1.type == DomainType.ANALYTICAL:
            self.domain1, self.domain2 = self.domain2, self.domain1

        self.penalty = self.domain1.spring * self._SPRING_PENALTY_SCALE
        self.stableTime = 0.5 * (self.domain1.mass / self.penalty) ** 0.5


@ti.data_oriented
class ContactSpringHeightField(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0

    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        # Ensure domain1 is Spring and domain2 is HeightField
        if self.domain1.type == DomainType.HEIGHTFIELD:
            self.domain1, self.domain2 = self.domain2, self.domain1

        self.penalty = self.domain1.spring * self._SPRING_PENALTY_SCALE
        self.stableTime = 0.5 * (self.domain1.mass / self.penalty) ** 0.5


# A helper function to create different contact types based on contact type
def contactHelper(domain1, domain2, type: int):
    if type == ContactType.FLEXANLAYTICAL:
        return ContactFlexAnalytical(domain1, domain2, type)
    elif type == ContactType.FLEXFLEX:
        return ContactFlexFlex(domain1, domain2, type)
    elif type == ContactType.FLEXRIGID:
        return ContactFlexRigid(domain1, domain2, type)
    elif type == ContactType.FLEXVOXELMAP:
        return ContactFlexVoxelMap(domain1, domain2, type)
    elif type == ContactType.FEXHEIGHTFIELD:
        return ContactFlexHeightField(domain1, domain2, type)
    elif type == ContactType.RIGIDSPRING:
        return ContactSpringRigid(domain1, domain2, type)
    elif type == ContactType.FLEXSPRING:
        return ContactSpringFlex(domain1, domain2, type)
    elif type == ContactType.ANALYTICALSPRING:
        return ContactSpringAnalytical(domain1, domain2, type)
    elif type == ContactType.SPRINGHEIGHTFIELD:
        return ContactSpringHeightField(domain1, domain2, type)
    else:
        return None
