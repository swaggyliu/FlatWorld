from collections import defaultdict
from definitions import *
import numpy as np
import taichi as ti

# =============================================================================
# Mesh
# =============================================================================


@ti.data_oriented
class Mesh:
    def __init__(self, d, conns, coords, is_rigid=False):
        self.d = d
        self.connectivity = conns
        self.coords = coords
        self.numNodes = self.coords.shape[0]
        self.numElements = conns.shape[0]
        self.is_rigid = is_rigid
        if is_rigid:
            # For rigid bodies, conns already contains boundary triangular faces
            # Initialize with triangular connectivity
            self.numNodePerEl = 2
            self.massWeights = 1 / 2

            # For rigid bodies, all elements are boundary elements
            self.numBoundElements = self.numElements
            self.boundaryElements = self.connectivity

            # All nodes are boundary nodes
            self.numBoundNodes = self.numNodes
            self.boundaryNodes = np.array([i for i in range(self.numNodes)])
        else:
            if d == 2:
                self.numNodePerEl = 3
                self.massWeights = 1 / 3
            else:
                self.numNodePerEl = 4
                self.massWeights = 1 / 4

            self.boundaryElements = np.array(self.getBoundaryEdges_(conns))
            self.numBoundElements = self.boundaryElements.shape[0]
            boundaryNodes = set()
            for i in range(self.numBoundElements):
                boundaryNodes.update(self.boundaryElements[i])

            self.boundaryNodes = np.array(sorted(list(boundaryNodes)))
            self.numBoundNodes = self.boundaryNodes.shape[0]

        self.charLength = self.calCharacteristicLength()
        self.charAverageLength = self.calPenaltyLength()
        self.charLengthSquare = 0.5 * (self.charLength * self.charLength)

    def getBoundaryEdges_(self, conns):
        """Return a list of edges that appear only once in connectivity (boundary edges)."""
        edge_count = defaultdict(int)
        n_vertices = self.d + 1  # 2D: 3, 3D: 4

        for cell in conns:
            for i in range(n_vertices):
                # define an edge，Make sure the vertex indices are sorted from small to large to avoid double counting
                v1, v2 = cell[i], cell[(i + 1) % n_vertices]
                edge = (v1, v2)
                edge_reverse = (v2, v1)
                if edge_reverse in edge_count:
                    edge_count[edge_reverse] += 1
                else:
                    edge_count[edge] += 1


        # The number of occurrences is1The edge of is the boundary edge
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        return np.array(boundary_edges)

    def calCharacteristicLength(self):
        """Estimate a conservative characteristic length as global minimum edge length."""
        edge_lengths = self._collect_edge_lengths()
        if edge_lengths.size == 0:
            return 0.0
        return float(np.min(edge_lengths))

    def calPenaltyLength(self):
        """Estimate a representative characteristic length for penalty scaling."""
        edge_lengths = self._collect_edge_lengths()
        if edge_lengths.size == 0:
            return 0.0

        # TODO: Here we should use A_e / max(L_e) to calculate the characteristic length
        # The mass scaling should then be used to scale the mass of really small elements
        return float(np.mean(edge_lengths))

    def _collect_edge_lengths(self):
        """Collect all element-edge lengths for simplex meshes."""
        lmin = 1e30
        # Enumerate all unique local edges for supported simplex elements.
        if self.numNodePerEl == 3:
            edge_pairs = ((0, 1), (1, 2), (2, 0))
        elif self.numNodePerEl == 4:
            edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        else:
            edge_pairs = tuple((j, (j + 1) % self.numNodePerEl) for j in range(self.numNodePerEl))

        edge_lengths = []
        for i in range(self.numElements):
            conn = self.connectivity[i]
            for ia, ib in edge_pairs:
                a, b = self.coords[conn[ia]], self.coords[conn[ib]]
                l = np.linalg.norm(a - b)
                lmin = min(l, lmin)
                edge_lengths.append(l)

        if lmin >= 1e29:
            return np.array([], dtype=np.float32)
        return np.asarray(edge_lengths, dtype=np.float32)

    def getCenterPoint(self):
        """Return the arithmetic center of node coordinates (host function)."""
        coord = np.zeros((self.d,), dtype=np.float32)
        for I in range(self.numNodes):
            coord += self.coords[I]

        center = coord / self.numNodes
        return center

    def computeBoundaryNodeNormals(self):
        """Compute boundary node normals for FEM-FEM contact."""
        boundaryNodeNormals = np.zeros((self.numNodes, self.d), dtype=np.float32)

        for el in range(self.numBoundElements):
            conn = self.boundaryElements[el]
            n1, n2 = conn[0], conn[1]
            p1 = self.coords[n1]
            p2 = self.coords[n2]

            # 2D boundary edge normal (perpendicular to edge)
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]], dtype=np.float32)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm

            boundaryNodeNormals[n1] += normal
            boundaryNodeNormals[n2] += normal

        # Normalize each node's normal
        for nd in range(self.numNodes):
            norm = np.linalg.norm(boundaryNodeNormals[nd])
            if norm > 1e-10:
                boundaryNodeNormals[nd] = boundaryNodeNormals[nd] / norm

        return boundaryNodeNormals


@ti.func
def getShapeFnsTri(self, psi: ti.float32, eta: ti.float32):
    """Return triangular linear shape functions evaluated at (psi,eta)."""
    return ti.Vector([psi, eta, 1 - psi - eta], ti.float32)


@ti.func
def getShapeFnsTet(self, psi: ti.float32, eta: ti.float32, gamma: ti.float32) -> fVec4:
    """Return linear tetrahedral shape functions evaluated at (psi,eta,gamma)."""
    return ti.Vector([psi, eta, gamma, 1 - psi - eta - gamma], ti.float32)


@ti.func
def getBoundaryShapeFns(self, psi: ti.float32, eta: ti.float32):
    """Return 1D shape functions for a boundary edge parameter psi."""
    return ti.Vector([1 - psi, psi], ti.float32)


@ti.func
def getJacobian2D(c1, c2, c3, psi: ti.float32, eta: ti.float32, gamma: ti.f32):
    """Compute the 2x2 Jacobian for element i at param (psi,eta)."""
    jac = ti.Matrix([[c1[0] - c3[0], c2[0] - c3[0]], [c1[1] - c3[1], c2[1] - c3[1]]])

    return jac


@ti.func
def getWeights2D(jac: Mat2x2) -> ti.float32:
    """Return triangle area weight = 0.5 * |det(J)|."""
    return abs(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]) * 0.5


@ti.func
def getBMatrix2D(psi: ti.float32, eta: ti.float32, gamma: ti.f32, F, J_inv):
    """Assemble the 3x6 B-matrix for linear triangular elasticity at param coords."""

    shapeDparam = ti.Matrix([[1, 0, -1], [0, 1, -1]], ti.float32)
    dndx = shapeDparam.transpose() @ J_inv  # 3 x 2 matrix
    BMat = ti.Matrix.zero(ti.f32, 3, 6)
    for i in range(3):
        BMat[0, 2 * i] = dndx[i, 0] * F[0, 0]
        BMat[0, 2 * i + 1] = dndx[i, 0] * F[1, 0]
        BMat[1, 2 * i] = dndx[i, 1] * F[0, 1]
        BMat[1, 2 * i + 1] = dndx[i, 1] * F[1, 1]
        BMat[2, 2 * i] = dndx[i, 1] * F[0, 0] + dndx[i, 0] * F[0, 1]
        BMat[2, 2 * i + 1] = dndx[i, 0] * F[1, 1] + dndx[i, 1] * F[1, 0]

    return BMat


@ti.func
def getJacobian3D(c1, c2, c3, c4, psi: ti.float32, eta: ti.float32, gamma: ti.float32) -> Mat3x3:
    """Compute the 3x3 Jacobian for tetrahedron i at param coords (psi,eta,gamma)."""
    return ti.Matrix(
        [
            [c1[0] - c4[0], c2[0] - c4[0], c3[0] - c4[0]],
            [c1[1] - c4[1], c2[1] - c4[1], c3[1] - c4[1]],
            [c1[2] - c4[2], c2[2] - c4[2], c3[2] - c4[2]],
        ]
    )


@ti.func
def getWeights3D(jac: Mat3x3) -> ti.float32:
    """Return tetrahedron volume weight = |det(J)| / 6."""
    return (abs(jac.determinant())) * (1.0 / 6.0)


@ti.func
def getBMatrix3D(psi: ti.float32, eta: ti.float32, gamma: ti.float32, F, J_inv):
    """Assemble the 6x12 B-matrix for linear tetrahedral elasticity at param coords."""
    shapeDparam = ti.Matrix([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], ti.float32)
    dndx = shapeDparam.transpose() @ J_inv  # 4x3 matrix
    # 6 x 12 matrix
    BMat = ti.Matrix.zero(ti.f32, 6, 12)
    for i in range(4):
        BMat[0, 3 * i] = dndx[i, 0] * F[0, 0]
        BMat[0, 3 * i + 1] = dndx[i, 0] * F[1, 0]
        BMat[0, 3 * i + 2] = dndx[i, 0] * F[2, 0]
        BMat[1, 3 * i] = dndx[i, 1] * F[0, 1]
        BMat[1, 3 * i + 1] = dndx[i, 1] * F[1, 1]
        BMat[1, 3 * i + 2] = dndx[i, 1] * F[2, 1]
        BMat[2, 3 * i] = dndx[i, 2] * F[0, 2]
        BMat[2, 3 * i + 1] = dndx[i, 2] * F[1, 2]
        BMat[2, 3 * i + 2] = dndx[i, 2] * F[2, 2]

        BMat[3, 3 * i] = dndx[i, 1] * F[0, 0] + dndx[i, 0] * F[0, 1]
        BMat[3, 3 * i + 1] = dndx[i, 0] * F[1, 1] + dndx[i, 1] * F[1, 0]
        BMat[3, 3 * i + 2] = dndx[i, 1] * F[2, 0] + dndx[i, 0] * F[2, 1]

        BMat[4, 3 * i] = dndx[i, 1] * F[0, 2] + dndx[i, 2] * F[0, 1]
        BMat[4, 3 * i + 1] = dndx[i, 2] * F[1, 1] + dndx[i, 1] * F[1, 2]
        BMat[4, 3 * i + 2] = dndx[i, 1] * F[2, 2] + dndx[i, 2] * F[2, 1]

        BMat[5, 3 * i] = dndx[i, 2] * F[0, 0] + dndx[i, 0] * F[0, 2]
        BMat[5, 3 * i + 1] = dndx[i, 2] * F[1, 0] + dndx[i, 0] * F[1, 2]
        BMat[5, 3 * i + 2] = dndx[i, 0] * F[2, 2] + dndx[i, 2] * F[2, 0]

    return BMat

