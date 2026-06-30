"""
FemSpringManager - Unified manager for FEM and SpringMass domains
Similar to RigidManager, this batches all FEM/Spring computations into GPU kernels
for massive speedup (expected 5-10x improvement over per-domain loops).

Key optimizations:
1. Unified data layout: all nodes/elements from all domains in single Taichi fields
2. Batched force computation: single kernel processes all domains
3. Batched velocity/position update: single kernel for all nodes
4. Reduced CPU-GPU synchronization overhead

Author: Dongyu Liu
Date: 2025-12-19
"""

from definitions import *
from materials import getStress2D, getStressPlaneStress2D, misesReturnMap2D
from mesh import *
import numpy as np
from spatialmanager import SpatialHashManager
import taichi as ti


@ti.data_oriented
class FemSpringManager:
    """
    Manages all FEM and SpringMass domains in a unified data structure.

    Architecture:
    - Global node arrays: position, velocity, acceleration, force, mass
    - Global element arrays: connectivity, properties, volumes
    - Domain metadata: offsets and counts for indexing
    - Batched kernels: process all domains in parallel
    """

    def __init__(self, d, domains, spatial_hash=None):
        """
        Initialize FemSpringManager with pre-allocated buffers.

        Args:
            max_nodes: Maximum total nodes across all domains (ignored, calculated from domains)
            max_elements: Maximum total elements across all domains (ignored, calculated from domains)
            d: Spatial dimension (2 or 3)
            spatial_hash: SpatialHashManager for FEM boundary element acceleration
        """
        self.d = d
        self._sh_fem_elapsed = 0.0
        self._sh_rebuild_interval = 1.0 / 500.0  # Rebuild spatial hash every 2 ms

        # --- OPTIMIZATION: Pre-scan domains to calculate actual memory requirements ---
        count_nodes = 0
        count_elements = 0
        count_fem_domains = 0
        count_spring_domains = 0
        count_boundary_nodes = 0
        count_boundary_elements = 0

        for dom in domains:
            if dom.type == DomainType.FEM:
                count_nodes += dom.nnodes
                count_elements += dom.nelements
                count_fem_domains += 1
                count_boundary_nodes += dom.mesh.numBoundNodes
                count_boundary_elements += dom.mesh.numBoundElements
            elif dom.type == DomainType.SPRINGMASS:
                count_nodes += dom.nnodes
                count_elements += dom.nelements
                count_spring_domains += 1

        # Allocate memory with safety buffer (count + buffer)
        self.MAX_NODES = max(count_nodes + 1024, 2048)
        self.MAX_ELEMENTS = max(count_elements + 2048, 4096)
        self.MAX_BOUNDARY_NODES = max(count_boundary_nodes + 1024, self.MAX_NODES)
        self.MAX_BOUNDARY_ELEMENTS = max(count_boundary_elements + 1024, self.MAX_NODES)

        print(f"[FemSpringManager] Memory allocation: {count_nodes} nodes + buffer → {self.MAX_NODES} MAX_NODES")
        print(
            f"[FemSpringManager] Memory allocation: {count_elements} elements + buffer → {self.MAX_ELEMENTS} MAX_ELEMENTS"
        )

        self.spatialHash = SpatialHashManager(
            d, max_elements=self.MAX_BOUNDARY_ELEMENTS
        )  # Store reference for FEM-FEM contact optimization

        # Domain counters
        self.numFemDomains = ti.field(ti.i32, shape=())
        self.numSpringDomains = ti.field(ti.i32, shape=())
        self.totalNodes = ti.field(ti.i32, shape=())
        self.totalElements = ti.field(ti.i32, shape=())
        self.totalFEMElements = ti.field(ti.i32, shape=())
        self.totalFEMNodes = ti.field(ti.i32, shape=())

        # Node data (shared by all domains)
        self.coords = ti.Vector.field(d, ti.f32, self.MAX_NODES)  # Current positions
        self.U = ti.Vector.field(d, ti.f32, self.MAX_NODES)  # Displacement
        self.V = ti.Vector.field(d, ti.f32, self.MAX_NODES)  # Velocity
        self.A = ti.Vector.field(d, ti.f32, self.MAX_NODES)  # Acceleration
        self.Fext = ti.Vector.field(d, ti.f32, self.MAX_NODES)  # External force
        self.mass = ti.field(ti.f32, self.MAX_NODES)  # Nodal mass

        # Boundary conditions (per node)
        self.bcNodes = ti.field(ti.i32, self.MAX_NODES)
        self.bcValues = ti.Vector.field(d, ti.f32, self.MAX_NODES)

        # Element data
        self.connectivity = ti.Vector.field(4, ti.i32, self.MAX_ELEMENTS)  # Max 4 nodes per element (tet/quad)
        self.elemType = ti.field(ti.i32, self.MAX_ELEMENTS)  # 0=Spring, 1=Tri(2D), 2=Tet(3D)
        self.elemNodesCount = ti.field(ti.i32, self.MAX_ELEMENTS)  # Nodes per element

        # Element properties (union for different types)
        self.elemVolume = ti.field(ti.f32, self.MAX_ELEMENTS)  # FEM: element volume and Spring: rest length

        # FEM-specific: Jacobian inverse (for 2D tri and 3D tet)
        self.J_inv = ti.Matrix.field(d, d, ti.f32, self.MAX_ELEMENTS)

        # Material properties (per element, indexed)
        self.matE = ti.field(ti.f32, self.MAX_ELEMENTS)  # Young's modulus and stiffness for spring
        self.matNu = ti.field(ti.f32, self.MAX_ELEMENTS)  # Poisson's ratio
        self.matRho = ti.field(ti.f32, self.MAX_ELEMENTS)  # Density
        self.matDamping = ti.field(ti.f32, self.MAX_ELEMENTS)  # Damping coefficient for spring
        self.matType = ti.field(ti.i32, self.MAX_ELEMENTS)  # Material type
        self.matYieldStress = ti.field(ti.f32, self.MAX_ELEMENTS)  # Mises: initial yield stress
        self.matH = ti.field(ti.f32, self.MAX_ELEMENTS)  # Mises: linear hardening modulus

        # Plasticity history variables (per element, for Mises J2 plasticity)
        voigt_dim = 3 if d == 2 else 6
        self.plasticStrain = ti.Vector.field(voigt_dim, ti.f32, self.MAX_ELEMENTS)  # accumulated plastic strain (Voigt)
        self.eqPlasticStrain = ti.field(ti.f32, self.MAX_ELEMENTS)  # equivalent (accumulated) plastic strain scalar

        # Element stress (Cauchy, Voigt notation) — stored every substep for optional export
        self.elemStress = ti.Vector.field(voigt_dim, ti.f32, self.MAX_ELEMENTS)
        # Element strain (Green-Lagrange, Voigt notation) — stored every substep for optional export
        self.elemStrain = ti.Vector.field(voigt_dim, ti.f32, self.MAX_ELEMENTS)

        # Domain metadata (for indexing)
        self.MAX_DOMAINS = max(len(domains), 128)  # Safety buffer for domains
        self.domainNodeOffset = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainNodeCount = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainElemOffset = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainElemCount = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainType = ti.field(ti.i32, self.MAX_DOMAINS)  # 0=Spring, 1=FEM
        self.boundaryNodes = ti.field(ti.i32, self.MAX_BOUNDARY_NODES)
        self.boundaryElements = ti.Vector.field(3, ti.i32, self.MAX_BOUNDARY_ELEMENTS)
        self.domainBoundaryNodeOffset = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainBoundaryNodeCount = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainBoundaryElemOffset = ti.field(ti.i32, self.MAX_DOMAINS)
        self.domainBoundaryElementCount = ti.field(ti.i32, self.MAX_DOMAINS)
        self.boundaryNodeNormals = ti.Vector.field(
            d, ti.f32, self.MAX_BOUNDARY_NODES
        )  # For FEM-FEM contact, store boundary node normals
        self.boundaryNodeNormals.fill(0.0)
        self.boundaryNodeCounts = ti.field(ti.i32, shape=())
        self.boundaryNodeCounts[None] = count_boundary_nodes
        self.boundaryElementCounts = ti.field(ti.i32, shape=())
        self.boundaryElementCounts[None] = count_boundary_elements
        self.femDomainIds = ti.field(ti.i32, self.MAX_DOMAINS)

        # Python-side domain list (for reference)
        self.domains = []
        self.domainMeshes = []  # Keep reference to mesh objects

        # Initialize counters
        self.numFemDomains[None] = 0
        self.numSpringDomains[None] = 0
        self.totalNodes[None] = 0
        self.totalElements[None] = 0
        self.totalFEMElements[None] = 0
        self.totalFEMNodes[None] = 0

        self.stableTime = 1.0

        self.processDomains(domains)
        self.processConditions()

        if self.totalFEMElements[None] > 0:
            self.prepareFEMData_()

    def processDomains(self, domains):
        """
        Add FEM and SpringMass domains to this manager.
        """
        for domain_idx, dom in enumerate(domains):
            if dom.type == DomainType.FEM:
                fem_idx = self.addFemDomain(dom)
            elif dom.type == DomainType.SPRINGMASS:
                fem_idx = self.addSpringDomain(dom)
            else:
                continue

            self.femDomainIds[fem_idx] = domain_idx
            self.stableTime = min(dom.stableTime, self.stableTime)

    def processConditions(self):
        for domain_idx, domain in enumerate(self.domains):
            numBcs = len(domain.bcs)
            for i in range(numBcs):
                bc = domain.bcs[i]
                type, nodes, values = bc.processData()
                if values is None:
                    values = [0.0] * domain.d
                if nodes is None:
                    nodes = range(domain.nnodes)
                for idx in nodes:
                    nd = self.domainNodeOffset[domain_idx] + idx
                    self.bcNodes[nd] |= type
                    self.bcValues[nd] = ti.Vector(values)

            # Handle initial conditions
            for initialCondition in domain.initials:
                initialCondition.update(self.V, self.d, self.domainNodeOffset[domain_idx], domain.nnodes)

    def addFemDomain(self, domain):

        domain_idx = len(self.domains)
        if domain_idx >= self.MAX_DOMAINS:
            print(f"\033[91mError: Exceeded MAX_DOMAINS ({self.MAX_DOMAINS})! Cannot add more domains.\033[0m")
            raise RuntimeError(f"Exceeded maximum domains ({self.MAX_DOMAINS})")

        node_offset = self.totalNodes[None]
        elem_offset = self.totalElements[None]

        mesh = domain.mesh
        nnodes = mesh.numNodes
        nelems = mesh.numElements

        if node_offset + nnodes > self.MAX_NODES:
            print(
                f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES})! Trying to add {nnodes} nodes, current total: {node_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum nodes ({self.MAX_NODES})")
        if elem_offset + nelems > self.MAX_ELEMENTS:
            print(
                f"\033[91mError: Exceeded MAX_ELEMENTS ({self.MAX_ELEMENTS})! Trying to add {nelems} elements, current total: {elem_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum elements ({self.MAX_ELEMENTS})")

        # Copy mesh data to manager fields
        coords_np = mesh.coords
        conns_np = mesh.connectivity

        # Store domain metadata
        self.domainNodeOffset[domain_idx] = node_offset
        self.domainNodeCount[domain_idx] = nnodes
        self.domainElemOffset[domain_idx] = elem_offset
        self.domainElemCount[domain_idx] = nelems
        self.domainType[domain_idx] = 1  # FEM type

        # Copy node data
        for i in range(nnodes):
            global_idx = node_offset + i
            self.coords[global_idx] = ti.Vector(coords_np[i])

        # Copy element data
        mat = domain.prop.getMaterial()

        nodeCount = 3
        elemeType = 1  # 2D triangle
      
        for i in range(nelems):
            global_idx = elem_offset + i
            conn = conns_np[i]

            # Store connectivity (offset by node_offset)
            conn_global = [node_offset + conn[j] for j in range(len(conn))]
            # Pad to 4 elements
            while len(conn_global) < 4:
                conn_global.append(-1)
            self.connectivity[global_idx] = ti.Vector(conn_global)

            # Element type and node count
            self.elemType[global_idx] = elemeType
            self.elemNodesCount[global_idx] = nodeCount

            # Material properties
            self.matE[global_idx] = mat.E
            self.matNu[global_idx] = mat.nu
            self.matRho[global_idx] = mat.rho
            self.matType[global_idx] = mat.type
            # Plasticity parameters (default 0 for non-Mises materials)
            self.matYieldStress[global_idx] = getattr(mat, "sigma_y", 0.0)
            self.matH[global_idx] = getattr(mat, "H", 0.0)

        # copy boundary nodes and elements
        if domain_idx > 0:
            self.domainBoundaryNodeOffset[domain_idx] = (
                self.domainBoundaryNodeOffset[domain_idx - 1] + self.domainBoundaryNodeCount[domain_idx - 1]
            )
        for i in range(domain.mesh.numBoundNodes):
            global_idx = self.domainBoundaryNodeOffset[domain_idx] + i
            self.boundaryNodes[global_idx] = node_offset + domain.mesh.boundaryNodes[i]
        self.domainBoundaryNodeCount[domain_idx] = domain.mesh.numBoundNodes

        if domain_idx > 0:
            self.domainBoundaryElemOffset[domain_idx] = (
                self.domainBoundaryElemOffset[domain_idx - 1] + self.domainBoundaryElementCount[domain_idx - 1]
            )
        for i in range(domain.mesh.numBoundElements):
            global_idx = self.domainBoundaryElemOffset[domain_idx] + i
            conn = domain.mesh.boundaryElements[i]
            conn_global = [node_offset + conn[j] for j in range(len(conn))]
            while len(conn_global) < 3:
                conn_global.append(-1)
            self.boundaryElements[global_idx] = ti.Vector(conn_global)
        self.domainBoundaryElementCount[domain_idx] = domain.mesh.numBoundElements

        # Update counters
        self.totalNodes[None] += nnodes
        self.totalElements[None] += nelems
        self.totalFEMElements[None] += nelems
        self.totalFEMNodes[None] += nnodes
        self.numFemDomains[None] += 1

        # Compute and store boundary node normals from mesh
        boundary_normals = mesh.computeBoundaryNodeNormals()
        for i in range(nnodes):
            global_idx = node_offset + i
            self.boundaryNodeNormals[global_idx] = ti.Vector(boundary_normals[i])

        # Store domain reference
        self.domains.append(domain)
        self.domainMeshes.append(mesh)

        domain.attach(self, domain_idx)

        return domain_idx

    def addSpringDomain(self, domain):
        """
        Add a SpringMass domain to the manager.

        Args:
            domain: SpringMassDomain object

        Returns:
            domain_index: Index of this domain in the manager
        """
        domain_idx = len(self.domains)
        if domain_idx >= self.MAX_DOMAINS:
            print(f"\033[91mError: Exceeded MAX_DOMAINS ({self.MAX_DOMAINS})! Cannot add more domains.\033[0m")
            raise RuntimeError(f"Exceeded maximum domains ({self.MAX_DOMAINS})")

        node_offset = self.totalNodes[None]
        elem_offset = self.totalElements[None]

        nnodes = domain.nnodes
        nelems = domain.nelements

        if node_offset + nnodes > self.MAX_NODES:
            print(
                f"\033[91mError: Exceeded MAX_NODES ({self.MAX_NODES})! Trying to add {nnodes} nodes, current total: {node_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum nodes ({self.MAX_NODES})")
        if elem_offset + nelems > self.MAX_ELEMENTS:
            print(
                f"\033[91mError: Exceeded MAX_ELEMENTS ({self.MAX_ELEMENTS})! Trying to add {nelems} elements, current total: {elem_offset}\033[0m"
            )
            raise RuntimeError(f"Exceeded maximum elements ({self.MAX_ELEMENTS})")

        # Get data from domain
        coords_np = domain.coords
        conns_np = domain.connectivity
        rest_len_np = domain.restLength

        # Store domain metadata
        self.domainNodeOffset[domain_idx] = node_offset
        self.domainNodeCount[domain_idx] = nnodes
        self.domainElemOffset[domain_idx] = elem_offset
        self.domainElemCount[domain_idx] = nelems
        self.domainType[domain_idx] = 0  # Spring type

        # Copy node data
        for i in range(nnodes):
            global_idx = node_offset + i
            self.coords[global_idx] = ti.Vector(coords_np[i])

            # Spring mass is uniform
            self.mass[global_idx] = domain.mass

        # Copy element data (springs)
        for i in range(nelems):
            global_idx = elem_offset + i
            conn = conns_np[i]

            # Store connectivity (2 nodes for spring, pad to 4)
            conn_global = [node_offset + conn[0], node_offset + conn[1], -1, -1]
            self.connectivity[global_idx] = ti.Vector(conn_global)

            self.elemType[global_idx] = 0  # Spring type
            self.elemNodesCount[global_idx] = 2

            # Spring properties
            self.elemVolume[global_idx] = rest_len_np[i]
            self.matE[global_idx] = domain.spring
            self.matDamping[global_idx] = domain.damping

        domain.attach(self, domain_idx)
        # Update counters
        self.totalNodes[None] += nnodes
        self.totalElements[None] += nelems
        self.numSpringDomains[None] += 1

        # Store domain reference
        self.domains.append(domain)

        return domain_idx

    def setGlobalAABB(self, aabb):
        self.aabb = aabb

        # warmup proces
        self.substep(0.0, 0.0)

    @ti.kernel
    def prepareFEMData_(self):
        for i in range(self.totalFEMElements[None]):
            n1, n2, n3, n4 = self.connectivity[i]

            Jac = getJacobian2D(self.coords[n1], self.coords[n2], self.coords[n3], 0.0, 0.0, 0.0)
            self.elemVolume[i] = getWeights2D(Jac)
            self.J_inv[i] = Jac.inverse()

            ndMass = self.elemVolume[i] * self.matRho[i] / 3.0  # Lumped mass
            self.mass[n1] += ndMass
            self.mass[n2] += ndMass
            self.mass[n3] += ndMass
           

    # ── LS-DYNA-style selective mass scaling ─────────────────────────
    def applyMassScaling(self, dt_target: float):
        """Scale nodal masses so that every element's critical time step >= dt_target.

        Only elements whose natural dt is below dt_target receive extra mass.
        Springs are also handled: dt_spring = 0.5*sqrt(m/k).

        Following LS-DYNA:
            dt_elem = safety * L_e / c_e,   c_e = sqrt(E / rho)
            if dt_elem < dt_target:
                rho_new = E * (dt_target / (safety * L_e))^2
                added_mass_e = (rho_new - rho_old) * V_e, split to nodes

        Args:
            dt_target: Desired minimum time step (seconds).
        """
        added = self._mass_scaling_kernel(float(dt_target))
        total_mass = self._total_mass()
        pct = added / total_mass * 100.0 if total_mass > 0 else 0.0
        print(
            f"[MassScaling] dt_target={dt_target:.3e}  added_mass={added:.4e}  "
            f"total_mass={total_mass:.4e}  ({pct:.2f}%)"
        )

    @ti.kernel
    def _mass_scaling_kernel(self, dt_target: ti.f32) -> ti.f32:
        """Kernel: add mass to under-resolved FEM elements and springs."""
        added_mass = ti.f32(0.0)
        safety = ti.f32(0.4)

        # ── FEM elements (tri-2D) ──
        for i in range(self.totalFEMElements[None]):
            n1, n2, n3, n4 = self.connectivity[i]
            E_val = self.matE[i]
            rho_val = self.matRho[i]
            V_e = self.elemVolume[i]

            # Compute minimum edge length as characteristic length
            L_e = ti.f32(1e30)
            n_nodes = 1.0
            # 2D triangle: 3 edges
            e01 = (self.coords[n2] - self.coords[n1]).norm()
            e12 = (self.coords[n3] - self.coords[n2]).norm()
            e20 = (self.coords[n1] - self.coords[n3]).norm()
            L_e = ti.min(e01, ti.min(e12, e20))
            n_nodes = ti.f32(3.0)
         
            # Element critical dt = safety * L_e / sqrt(E / rho)
            c_e = ti.sqrt(E_val / rho_val)
            dt_elem = safety * L_e / c_e

            if dt_elem < dt_target and L_e > 1e-15:
                # Required density so that dt = dt_target
                rho_new = E_val * (dt_target / (safety * L_e)) ** 2
                dm = (rho_new - rho_val) * V_e / n_nodes
                self.mass[n1] += dm
                self.mass[n2] += dm
                self.mass[n3] += dm
              
                ti.atomic_add(added_mass, (rho_new - rho_val) * V_e)

        # ── Spring elements ──
        # dt_spring = 0.5 * sqrt(m / k)  →  m_min = 4 * k * dt_target^2
        for i in range(self.totalFEMElements[None], self.totalElements[None]):
            conn = self.connectivity[i]
            ia = conn[0]
            ib = conn[1]
            k_spring = self.matE[i]
            m_min = ti.f32(4.0) * k_spring * dt_target * dt_target
            # Each spring node needs at least m_min / 2 (shared by 2 nodes)
            # but mass is shared across springs, so ensure per-node minimum
            for nd in ti.static(range(2)):
                nid = conn[nd]
                if self.mass[nid] < m_min:
                    dm = m_min - self.mass[nid]
                    self.mass[nid] = m_min
                    ti.atomic_add(added_mass, dm)

        return added_mass

    @ti.kernel
    def _total_mass(self) -> ti.f32:
        total = ti.f32(0.0)
        for i in range(self.totalNodes[None]):
            ti.atomic_add(total, self.mass[i])
        return total

    def substep(self, dt: ti.f32, damping: ti.f32):
        self.femStep(dt, damping)
        # self.populate_fem_spatial_hash()   # later we should use this

    @ti.kernel
    def femStep(self, dt: ti.f32, damping: ti.f32):

        for i in range(self.totalFEMElements[None]):
            # FEM elements
            n1, n2, n3, n4 = self.connectivity[i]

            # calculate the current Jacobian
            Jac = getJacobian2D(self.coords[n1], self.coords[n2], self.coords[n3], 0.0, 0.0, 0.0)
            J_inv = self.J_inv[i]
            F = Jac @ J_inv
            BMat = getBMatrix2D(0.0, 0.0, 0.0, F, J_inv)

            sigma = ti.Vector.zero(ti.f32, 3)
            strain_voigt = ti.Vector.zero(ti.f32, 3)
            if self.matType[i] == MaterialType.MISES:
                sigma, eps_p_new, eqps_new, strain_voigt = misesReturnMap2D(
                    self.matE[i],
                    self.matNu[i],
                    self.matYieldStress[i],
                    self.matH[i],
                    F,
                    self.plasticStrain[i],
                    self.eqPlasticStrain[i],
                )
                self.plasticStrain[i] = eps_p_new
                self.eqPlasticStrain[i] = eqps_new
            else:
                sigma, strain_voigt = getStress2D(
                    self.matType[i], self.matE[i], self.matNu[i], ti.Vector([0.0, 0.0]), F
                )

            # Store element stress as Cauchy stress, while constitutive model returns PK2.
            S = ti.Matrix([[sigma[0], sigma[2]], [sigma[2], sigma[1]]])
            J = F.determinant()
            sigma_cauchy = (F @ S @ F.transpose()) / ti.max(J, 1e-12)
            self.elemStress[i] = ti.Vector([sigma_cauchy[0, 0], sigma_cauchy[1, 1], sigma_cauchy[0, 1]])
            self.elemStrain[i] = strain_voigt

            elFint = BMat.transpose() @ (sigma)
            elFint *= self.elemVolume[i]

            self.Fext[n1] -= ti.Vector([elFint[0], elFint[1]])
            self.Fext[n2] -= ti.Vector([elFint[2], elFint[3]])
            self.Fext[n3] -= ti.Vector([elFint[4], elFint[5]])

           

        for i in range(self.totalFEMElements[None], self.totalElements[None]):
            # These are spring elements
            conn = self.connectivity[i]
            # spring forces
            ia = conn[0]
            ib = conn[1]
            xab = self.coords[ia] - self.coords[ib]
            lnew = xab.norm()
            dir = xab / lnew

            # spring force of the spring
            ndForce = self.matE[i] * (lnew / self.elemVolume[i] - 1) * dir
            vab = self.V[ia] - self.V[ib]
            # damping force of the spring
            ndForce += self.matDamping[i] * vab.dot(dir) * dir
            self.Fext[ia] -= ndForce
            self.Fext[ib] += ndForce

        for i in range(self.totalNodes[None]):
            # Apply BCs
            if (self.bcNodes[i] & UTYPE) != 0:
                self.Fext[i].fill(0.0)
            elif (self.bcNodes[i] & ATYPE) != 0:
                self.Fext[i] = self.mass[i] * self.bcValues[i]
            elif (self.bcNodes[i] & GRAVITY) != 0:
                self.Fext[i] += self.mass[i] * self.bcValues[i]
            elif (self.bcNodes[i] & FORCETYPE) != 0:
                self.Fext[i] += self.bcValues[i]
            a = self.A[i] + (self.Fext[i] - damping * self.V[i]) * (1.0 / self.mass[i])
            self.V[i] += a * dt

            if (self.bcNodes[i] & VTYPE) != 0:
                self.V[i] = self.bcValues[i]
            elif (self.bcNodes[i] & UTYPE) != 0:
                self.V[i].fill(0.0)

            du = self.V[i] * dt
            self.U[i] += du
            self.coords[i] += du

        self.updateBBox()
        # sync mesh coords for fem
        # self.mesh.spatial_hash.reset()
        # self.mesh.spatial_hash.updateMesh(self.mesh, 0)
        # self.mesh.spatial_hash.splitAndAssignElement()
        self.Fext.fill(0.0)

    @ti.func
    def updateBBox(self):
        """Kernel to compute FEM domain bbox and write directly to aabb field.
         use boundary nodes for solid elements
          — critical for correct FEM-rigid AABB intersection."""

        for femidx in range(self.numFemDomains[None]):
            lb = ti.Vector([1e30 for k in range(self.d)])
            ub = ti.Vector([-1e30 for k in range(self.d)])
            nodeOffset = self.domainBoundaryNodeOffset[femidx]
            nodeCount = self.domainBoundaryNodeCount[femidx]

            for I in range(nodeOffset, nodeOffset + nodeCount):
                coord = self.coords[I]
                lb = ti.min(lb, coord)
                ub = ti.max(ub, coord)
            domain_idx = self.femDomainIds[femidx]
            self.aabb[domain_idx, 0] = lb
            self.aabb[domain_idx, 1] = ub

        offset = self.numFemDomains[None]
        for femidx in range(offset, offset + self.numSpringDomains[None]):
            lb = ti.Vector([1e30 for k in range(self.d)])
            ub = ti.Vector([-1e30 for k in range(self.d)])
            nodeOffset = self.domainNodeOffset[femidx]
            nodeCount = self.domainNodeCount[femidx]
            for I in range(nodeOffset, nodeOffset + nodeCount):
                coord = self.coords[I]
                lb = ti.min(lb, coord)
                ub = ti.max(ub, coord)
            domain_idx = self.femDomainIds[femidx]
            self.aabb[domain_idx, 0] = lb
            self.aabb[domain_idx, 1] = ub

    def maybe_rebuild_fem_spatial_hash(self, dt: float):
        self._sh_fem_elapsed += dt
        if self._sh_fem_elapsed >= self._sh_rebuild_interval:
            self.populate_fem_spatial_hash(velocity_buffer=0.0)
            self._sh_fem_elapsed = 0.0

    def populate_fem_spatial_hash(self, velocity_buffer: float = 0.0):
        """
        Populate spatial hash with FEM boundary elements for FEM-FEM contact optimization.

        Args:
            velocity_buffer: Extra AABB expansion per element (meters) to account for
                movement between spatial hash rebuilds. Typically v_max * rebuild_interval.
        """
        if self.spatialHash is None:
            return

        num_fem = self.numFemDomains[None]
        if num_fem == 0:
            return

        self.spatialHash.reset()
        self._populate_fem_spatial_hash_kernel(float(velocity_buffer))
        self.spatialHash.build()

    @ti.kernel
    def _populate_fem_spatial_hash_kernel(self, velocity_buffer: ti.f32):
        """Taichi kernel: insert all FEM boundary elements into spatial hash.

        Domain IDs stored in spatial hash are FEM-LOCAL indices (0, 1, 2, ...),
        matching boundaryElemOffset / boundaryElementCount indexing.
        """
        # First pass: compute global bounding box across all FEM domains
        global_lb = ti.Vector([1e9 for _ in range(self.d)])
        global_ub = ti.Vector([-1e9 for _ in range(self.d)])

        for fem_local_idx in range(self.numFemDomains[None]):
            global_did = self.femDomainIds[fem_local_idx]
            global_lb = ti.min(global_lb, self.aabb[global_did, 0])
            global_ub = ti.max(global_ub, self.aabb[global_did, 1])

        # Expand by velocity buffer + 1% safety margin
        expand = (global_ub - global_lb).max() * 0.01 + velocity_buffer
        global_lb -= expand
        global_ub += expand

        # Second pass: add all FEM boundary elements (using FEM-local index as domainID)
        for fem_local_idx in range(self.numFemDomains[None]):
            elem_offset = self.domainBoundaryElemOffset[fem_local_idx]
            elem_count = self.domainBoundaryElementCount[fem_local_idx]

            for local_elem_idx in range(elem_count):
                global_elem_idx = elem_offset + local_elem_idx
                elem_conn = self.boundaryElements[global_elem_idx]

                # Compute element bounding box
                lb = ti.Vector([1e30 for _ in range(self.d)])
                ub = ti.Vector([-1e30 for _ in range(self.d)])
                for k in ti.static(range(3)):
                    node_id = elem_conn[k]
                    if node_id >= 0:
                        coord = self.coords[node_id]
                        lb = ti.min(lb, coord)
                        ub = ti.max(ub, coord)

                # Buffer = velocity_buffer to cover movement between rebuilds
                self.spatialHash.addElement(lb, ub, fem_local_idx, global_elem_idx, velocity_buffer)

        # Store bounds for Python-level build
        self.spatialHash.setBounds(global_lb, global_ub)

    def drawMesh(self, gui, color, resoultion=800):
        totalEls = self.totalElements[None]
        totalNds = self.totalNodes[None]
        pos = self.coords.to_numpy()[:totalNds, :2]
        e2n = self.connectivity.to_numpy()[:totalEls]
        gui.circles(pos, radius=2, color=0xFFAA33)
        a, b, c = pos[e2n[:, 0]], pos[e2n[:, 1]], pos[e2n[:, 2]]
        gui.triangles(a, b, c, color=color)
