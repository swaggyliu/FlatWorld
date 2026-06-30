from mesh import Mesh
import numpy as np
import os


def calculate_mesh_inertia(mesh, mass):
    """
    Computes the inertia tensor of a mesh, assuming mass is distributed
    at the vertices. The mesh is assumed to be centered at its origin.
    """
    # For 2D, calculate moment of inertia around the z-axis
    coords = mesh.coords
    center = mesh.getCenterPoint()

    rel_coords = coords - center

    # mass per vertex
    mass_per_vertex = mass / len(coords)

    # I_z = sum(m_i * (x_i^2 + y_i^2))
    inertia = np.sum(mass_per_vertex * (rel_coords[:, 0] ** 2 + rel_coords[:, 1] ** 2))
    return inertia
  