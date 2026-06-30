from mesh import Mesh
import numpy as np

class FEMesher:
    def __init__(self, d):
        self.d = d

    def read(self, filename, readGroup=False):
        # Dispatch read based on file extension. Prefer meshio-based readers
        # for both .msh and .vtu files. If meshio isn't available, raise an
        # informative error.
        lower = filename.lower()
        if lower.endswith(".msh"):
            return self.read_msh(filename, readGroup)
        if lower.endswith(".vtu"):
            return self.read_vtu(filename, readGroup)
        raise RuntimeError(f"Unsupported mesh format for reading: {filename}")

    def read_msh(self, filename, readGroup=False):
        """Read a .msh (Gmsh) file using meshio and return Mesh or Mesh.

        This uses meshio so it does not require the gmsh Python API.
        """
        try:
            import meshio
        except Exception as e:
            raise RuntimeError(
                "meshio is required to read .msh files; install 'meshio' or use a different reader"
            ) from e

        mesh = meshio.read(filename)
        pts = np.asarray(mesh.points, dtype=np.float32)

        # meshio provides cells in a dict-like structure on recent versions
        cells_dict = getattr(mesh, "cells_dict", None)
        if cells_dict is None:
            # Older meshio versions expose mesh.cells as list of (type, data)
            cells_dict = {}
            for block in mesh.cells:
                ctype = block.type if hasattr(block, "type") else block[0]
                data = block.data if hasattr(block, "data") else block[1]
                cells_dict.setdefault(ctype, []).append(data)
            # Stack blocks if multiple
            for k, v in list(cells_dict.items()):
                cells_dict[k] = np.vstack(v)

        # Prefer tetra (3D) then triangle (2D)
        if "tetra" in cells_dict:
            conns = np.asarray(cells_dict["tetra"], dtype=np.int32)
            return Mesh(3, conns, pts[:, :3])
        if "triangle" in cells_dict:
            conns = np.asarray(cells_dict["triangle"], dtype=np.int32)
            return Mesh(2, conns, pts[:, :2])

        # If no triangle/tetra found, try to pick any cell block and infer dim
        if len(cells_dict) > 0:
            # pick first block
            first_type = next(iter(cells_dict))
            conns = np.asarray(cells_dict[first_type], dtype=np.int32)
            # infer element size to decide 2D/3D
            if conns.shape[1] == 3:
                return Mesh(2, conns, pts[:, :2])
            else:
                return Mesh(3, conns, pts[:, :3])

        raise RuntimeError(f"No supported cells (triangle/tetra) found in {filename}")

    def read_vtu(self, filename, readGroup=False):
        """Read a .vtu (VTK UnstructuredGrid) file using meshio and return Mesh or Mesh.

        VTU files are typically 3D (tetra/hex). We use meshio to parse them.
        """
        try:
            import meshio
        except Exception as e:
            raise RuntimeError(
                "meshio is required to read .vtu files; install 'meshio' or use a different reader"
            ) from e

        mesh = meshio.read(filename)
        pts = np.asarray(mesh.points, dtype=np.float32)
        cells_dict = getattr(mesh, "cells_dict", None)
        if cells_dict is None:
            cells_dict = {}
            for block in mesh.cells:
                ctype = block.type if hasattr(block, "type") else block[0]
                data = block.data if hasattr(block, "data") else block[1]
                cells_dict.setdefault(ctype, []).append(data)
            for k, v in list(cells_dict.items()):
                cells_dict[k] = np.vstack(v)

        if "tetra" in cells_dict:
            conns = np.asarray(cells_dict["tetra"], dtype=np.int32)
            return Mesh(3, conns, pts[:, :3])

        # vtu can contain triangles too
        if "triangle" in cells_dict:
            conns = np.asarray(cells_dict["triangle"], dtype=np.int32)
            return Mesh(2, conns, pts[:, :2])

        if len(cells_dict) > 0:
            first_type = next(iter(cells_dict))
            conns = np.asarray(cells_dict[first_type], dtype=np.int32)
            if conns.shape[1] == 3:
                return Mesh(2, conns, pts[:, :2])
            else:
                return Mesh(3, conns, pts[:, :3])

        raise RuntimeError(f"No supported cells found in {filename}")

    # =============================================================================
    # From here are 2D objects
    # =============================================================================

    def createTriangle(self, pt1, pt2, pt3):
        # Lightweight fallback: directly create a single-triangle Mesh
        verts = [pt1, pt2, pt3]
        conn = np.array([[0, 1, 2]], dtype=np.int32)
        coords = np.array(verts, dtype=np.float32)
        return Mesh(2, conn, coords)

    def createRectangle(self, lb, up):
        """Create rectangle and generate grid"""
        # Simple rectangle meshing without gmsh: two triangles
        lx, ly = lb
        ux, uy = up
        verts = [[lx, ly], [ux, ly], [ux, uy], [lx, uy]]
        conn = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        coords = np.array(verts, dtype=np.float32)
        return Mesh(2, conn, coords)

    def createCircle(self, origin, radius):
        """Create a circle and generate a grid"""
        # Approximate circle by triangle fan
        x, y = origin
        segments = 16
        verts = []
        for i in range(segments):
            theta = 2.0 * np.pi * i / segments
            verts.append([x + radius * np.cos(theta), y + radius * np.sin(theta)])
        center_idx = len(verts)
        verts.append([x, y])
        conn = []
        for i in range(segments):
            next_i = (i + 1) % segments
            conn.append([i, next_i, center_idx])
        conn = np.array(conn, dtype=np.int32)
        coords = np.array(verts, dtype=np.float32)
        return Mesh(2, conn, coords)
