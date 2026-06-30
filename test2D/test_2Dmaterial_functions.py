"""2D material-function tests."""

import taichi as ti
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.definitions import MaterialType

if not ti.lang.impl.get_runtime().prog:
    ti.init(offline_cache=True, arch=ti.cpu)


def test_stress2d_elastic_identity():
    from flatworld.materials.materialfunctions import getStress2D

    @ti.kernel
    def compute() -> ti.types.vector(3, ti.f32):
        s, _ = getStress2D(MaterialType.ELASTIC, 1e6, 0.3, 0.0, ti.Matrix.identity(ti.f32, 2))
        return s

    for value in compute():
        assert abs(value) < 1.0


def test_stress2d_elastic_uniaxial():
    from flatworld.materials.materialfunctions import getStress2D

    @ti.kernel
    def compute() -> ti.types.vector(3, ti.f32):
        s, _ = getStress2D(MaterialType.ELASTIC, ti.cast(1e6, ti.f32), ti.cast(0.3, ti.f32), 0.0, ti.Matrix([[1.01, 0.0], [0.0, 1.0]]))
        return s

    assert compute()[0] > 0


def test_stress2d_elastic_shear():
    from flatworld.materials.materialfunctions import getStress2D

    @ti.kernel
    def compute() -> ti.types.vector(3, ti.f32):
        s, _ = getStress2D(MaterialType.ELASTIC, ti.cast(1e6, ti.f32), ti.cast(0.3, ti.f32), 0.0, ti.Matrix([[1.0, 0.01], [0.0, 1.0]]))
        return s

    assert abs(compute()[2]) > 0


def test_stress2d_neohookean_identity():
    from flatworld.materials.materialfunctions import getStress2D

    @ti.kernel
    def compute() -> ti.types.vector(3, ti.f32):
        s, _ = getStress2D(MaterialType.NEOHOOKEAN, ti.cast(1e6, ti.f32), ti.cast(0.3, ti.f32), 0.0, ti.Matrix.identity(ti.f32, 2))
        return s

    stress = compute()
    assert abs(stress[0]) < 10.0
    assert abs(stress[1]) < 10.0


def test_stress2d_neohookean_stretch():
    from flatworld.materials.materialfunctions import getStress2D

    @ti.kernel
    def compute() -> ti.types.vector(3, ti.f32):
        s, _ = getStress2D(MaterialType.NEOHOOKEAN, ti.cast(1e6, ti.f32), ti.cast(0.3, ti.f32), 0.0, ti.Matrix([[1.1, 0.0], [0.0, 1.0]]))
        return s

    assert compute()[0] != 0.0