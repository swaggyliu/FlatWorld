from contact_detection import pointToEdgeContact
from definitions import *
from wp_init import ensure_warp
import warp as wp


# ---------------------------------------------------------------------------
# Spring–FEM contact kernel (2D)
# ---------------------------------------------------------------------------


@wp.kernel
def _spring_flex_contact_kernel(
    num_nodes: int,
    node_offset: int,
    penalty: float,
    friction_coeff: float,
    fem_domain_idx: int,
    num_bound_elements: int,
    domain_boundary_elem_offset: wp.array(dtype=int),
    boundary_elements: wp.array(dtype=wp.vec3i),
    coords: wp.array(dtype=wp.vec2),
    V: wp.array(dtype=wp.vec2),
    Fext: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    if tid >= num_nodes:
        return

    global_node_idx = node_offset + tid
    node_coord = coords[global_node_idx]
    node_vel = V[global_node_idx]

    element_offset = domain_boundary_elem_offset[fem_domain_idx]

    best_penetration = float(1e9)
    best_normal = wp.vec2(0.0, 0.0)
    weights = wp.vec2(0.0, 0.0)
    target_n0 = 0
    target_n1 = 0
    found_contact = int(0)

    for j in range(num_bound_elements):
        elem_id = j + element_offset
        if elem_id < 0:
            break

        elem_conn = boundary_elements[elem_id]
        n0 = coords[elem_conn[0]]
        n1 = coords[elem_conn[1]]
        pen, normal, cp, is_inside, w = pointToEdgeContact(node_coord, n0, n1, 2)

        if pen < best_penetration and wp.abs(pen) < 1.0 and is_inside:
            best_penetration = pen
            best_normal = normal
            weights = w
            target_n0 = elem_conn[0]
            target_n1 = elem_conn[1]
            found_contact = 1

    if found_contact == 1 and best_penetration < 0.0:
        normal_force = -best_normal * penalty * best_penetration
        total_force = normal_force

        if friction_coeff > 1e-9:
            surf_vel = V[target_n0] * weights[0] + V[target_n1] * weights[1]
            relative_vel = node_vel - surf_vel
            tangential_vel = relative_vel - wp.dot(relative_vel, best_normal) * best_normal
            friction_force = wp.vec2(0.0, 0.0)

            tlen = wp.length(tangential_vel)
            if tlen > 1e-9:
                friction_dir = -tangential_vel / tlen
                friction_magnitude = friction_coeff * wp.length(normal_force)
                friction_force = friction_dir * friction_magnitude

            total_force = normal_force + friction_force

        wp.atomic_add(Fext, global_node_idx, total_force)
        wp.atomic_add(Fext, target_n0, -(total_force * weights[0]))
        wp.atomic_add(Fext, target_n1, -(total_force * weights[1]))


class ContactBase:
    def __init__(self, domain1, domain2, type, is_tied=False):
        """Create a Contact controller for two domains and a contact type."""
        ensure_warp()
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

    def update(self, dt: float):
        pass

    def calculate(self, dt: float):
        pass


# ====================================================================
# Contact of different combinations!!!
# ====================================================================


class ContactFlexAnalytical(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        if self.domain1.type == DomainType.ANALYTICAL:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


class ContactFlexFlex(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        # Symmetric: take the more conservative (smaller) side penalty
        self.penalty = min(self._segment_penalty(self.domain1), self._segment_penalty(self.domain2)) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)
        self.stableTime = min(self.stableTime, self.calStableTime(self.penalty, self.domain2))


class ContactFlexRigid(ContactBase):
    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        if self.domain1.type == DomainType.RIGID:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


class ContactFlexHeightField(ContactBase):
    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        if self.domain1.type == DomainType.HEIGHTFIELD:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


class ContactFlexVoxelMap(ContactBase):
    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        if self.domain1.type == DomainType.VOXELMAP:
            self.domain1, self.domain2 = self.domain2, self.domain1
        self.penalty = self._segment_penalty(self.domain1) * self.penalty
        self.stableTime = self.calStableTime(self.penalty, self.domain1)


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

    def calculate(self, dt: float):
        """Compute contact forces between SpringMass and FEM using direct iteration."""
        if abs(dt) < 1e-9:
            return

        mgr = self.domain1.femManager
        node_offset = int(mgr.domainNodeOffset.numpy()[self.domain1.domainIdx])
        friction_coeff = max(self.domain1.friction, self.domain2.friction)
        num_bound = int(self.domain2.mesh.numBoundElements)

        wp.launch(
            _spring_flex_contact_kernel,
            dim=int(self.domain1.nnodes),
            inputs=[
                int(self.domain1.nnodes),
                node_offset,
                float(self.penalty),
                float(friction_coeff),
                int(self.domain2.domainIdx),
                num_bound,
                mgr.domainBoundaryElemOffset,
                mgr.boundaryElements,
                mgr.coords,
                mgr.V,
                mgr.Fext,
            ],
        )


class ContactSpringRigid(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0

    def __init__(self, domain1, domain2, type, is_tied=False):
        super().__init__(domain1, domain2, type, is_tied)
        # here we assume the second domain is the more rigid one
        if self.domain1.type == DomainType.RIGID:
            self.domain1, self.domain2 = self.domain2, self.domain1

        self.penalty = self.domain1.spring * self._SPRING_PENALTY_SCALE
        self.stableTime = 0.5 * (self.domain1.mass / self.penalty) ** 0.5


class ContactSpringAnalytical(ContactBase):
    _SPRING_PENALTY_SCALE = 100.0

    def __init__(self, domain1, domain2, type):
        super().__init__(domain1, domain2, type)
        # Ensure domain1 is Spring and domain2 is Analytical
        if self.domain1.type == DomainType.ANALYTICAL:
            self.domain1, self.domain2 = self.domain2, self.domain1

        self.penalty = self.domain1.spring * self._SPRING_PENALTY_SCALE
        self.stableTime = 0.5 * (self.domain1.mass / self.penalty) ** 0.5


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
