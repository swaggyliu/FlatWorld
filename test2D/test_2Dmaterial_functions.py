"""2D material-function tests (Warp)."""

import os
import sys

import warp as wp
from test_utils import init_sim

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld.definitions import MaterialType
from flatworld.materials.materialfunctions import getStress2D

init_sim()


@wp.kernel
def _k_elastic_identity(out: wp.array(dtype=wp.vec3)):
    s, _ = getStress2D(
        MaterialType.ELASTIC,
        1.0e6,
        0.3,
        wp.vec2(0.0, 0.0),
        wp.identity(n=2, dtype=wp.float32),
    )
    out[0] = s


@wp.kernel
def _k_elastic_uniaxial(out: wp.array(dtype=wp.vec3)):
    s, _ = getStress2D(
        MaterialType.ELASTIC,
        1.0e6,
        0.3,
        wp.vec2(0.0, 0.0),
        wp.mat22(1.01, 0.0, 0.0, 1.0),
    )
    out[0] = s


@wp.kernel
def _k_elastic_shear(out: wp.array(dtype=wp.vec3)):
    s, _ = getStress2D(
        MaterialType.ELASTIC,
        1.0e6,
        0.3,
        wp.vec2(0.0, 0.0),
        wp.mat22(1.0, 0.01, 0.0, 1.0),
    )
    out[0] = s


@wp.kernel
def _k_neohookean_identity(out: wp.array(dtype=wp.vec3)):
    s, _ = getStress2D(
        MaterialType.NEOHOOKEAN,
        1.0e6,
        0.3,
        wp.vec2(0.0, 0.0),
        wp.identity(n=2, dtype=wp.float32),
    )
    out[0] = s


@wp.kernel
def _k_neohookean_stretch(out: wp.array(dtype=wp.vec3)):
    s, _ = getStress2D(
        MaterialType.NEOHOOKEAN,
        1.0e6,
        0.3,
        wp.vec2(0.0, 0.0),
        wp.mat22(1.1, 0.0, 0.0, 1.0),
    )
    out[0] = s


def _run_vec3(kernel):
    out = wp.zeros(1, dtype=wp.vec3)
    wp.launch(kernel, dim=1, inputs=[out])
    return out.numpy()[0]


def test_stress2d_elastic_identity():
    for value in _run_vec3(_k_elastic_identity):
        assert abs(value) < 1.0


def test_stress2d_elastic_uniaxial():
    assert _run_vec3(_k_elastic_uniaxial)[0] > 0


def test_stress2d_elastic_shear():
    assert abs(_run_vec3(_k_elastic_shear)[2]) > 0


def test_stress2d_neohookean_identity():
    stress = _run_vec3(_k_neohookean_identity)
    assert abs(stress[0]) < 10.0
    assert abs(stress[1]) < 10.0


def test_stress2d_neohookean_stretch():
    assert _run_vec3(_k_neohookean_stretch)[0] != 0.0
