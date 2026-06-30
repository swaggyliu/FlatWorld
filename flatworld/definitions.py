from enum import Enum
import taichi as ti


class DomainType:
    ANALYTICAL = 0b000001
    FEM = 0b000010
    RIGID = 0b000100
    SPRINGMASS = 0b001000
    HEIGHTFIELD = 0b100000
    VOXELMAP = 0b1000000


class ContactType:
    FLEXANLAYTICAL = 0b000011  # OK
    FLEXFLEX = 0b000010  # OK
    RIGIDANLAYTICAL = 0b000101  # OK
    FLEXRIGID = 0b000110  # OK
    RIGIDRIGID = 0b000100  # OK
    RIGIDSPRING = 0b001100  # OK
    FLEXSPRING = 0b001010  # OK
    ANALYTICALANALYTICAL = 0b000001  # Not going to implement
    ANALYTICALSPRING = 0b001001  # OK
    SPRINGSPRING = 0b001000  # TODO: NOT IMPLEMENTED
    FEXHEIGHTFIELD = 0b100010  # OK
    FLEXVOXELMAP = 0b1000010  # OK
    RIGIDHEIGHTFIELD = 0b100100  # OK
    RIGIDVOXELMAP = 0b1000100  # OK
    SPRINGHEIGHTFIELD = 0b101000  # OK
    SPRINGVOXELMAP = 0b1001000  # TODO: NOT IMPLEMENTED


class BoundaryConditionType:
    FIXED = 1
    PRESCRIBED_DISPLACEMENT = 2
    PRESCRIBED_VELOCITY = 3
    PRESCRIBED_ACCELERATION = 4
    FORCE = 5
    GRAVITY = 6
    ENFORCE_DISPLACEMENT = 7
    ENFORCE_ROTATION_DISPLACEMENT = 8
    ENFORCE_VELOCITY = 9
    ENFORCE_ROTATION_VELOCITY = 10
    ENFORCE_ACCELERATION = 11
    ENFORCE_ROTATION_ACCELERATION = 12
    TORQUE = 13
    FIXED_ALL = 14


class RigidType:
    BALL = 0b00001  # OK
    BOX = 0b00010  # OK
    CAPSULE = 0b01000  # OK
    MESH = 0b10000  # OK


class RigidContactType:
    BALLBALL = 0b00001  # OK
    BOXBALL = 0b00011  # OK
    CAPSULEBALL = 0b01001  # OK
    MESHBALL = 0b10001  # OK
    BOXBOX = 0b00010  # OK
    CAPSULEBOX = 0b01010  # OK
    MESHBOX = 0b10010  # OK
    CAPSULECAPSULE = 0b01000  # OK
    MESHCAPSULE = 0b11000  # OK
    MESHMESH = 0b10000  # OK


class JointType:
    Revolute = 1
    Spherical = 2
    Weld = 3  # Fixed
    Link = 4
    Beam = 5
    Universal = 6
    Prismatic = 7  # Axial/Translational in Abaqus


class MaterialType:
    ELASTIC = 1
    NEOHOOKEAN = 2
    MISES = 3


# ODE-style collision filtering bits.
# Pair (a, b) is valid only when both are true:
#   (collide_bits[a] & category_bits[b]) != 0
#   (collide_bits[b] & category_bits[a]) != 0
COLLISION_CATEGORY_ROBOT = 0b00000001
COLLISION_CATEGORY_GROUND = 0b00000010
COLLISION_CATEGORY_FOOT = 0b00000100
COLLISION_CATEGORY_FEM = 0b00001000
COLLISION_CATEGORY_ORDINARY_RIGID = 0b00100000
COLLISION_CATEGORY_VIRTUAL = 0b00000000
COLLISION_MASK_ALL = 0b11111111

# BC TYPES
UTYPE = 0b000000001
VTYPE = 0b000000010
ATYPE = 0b000000100
FORCETYPE = 0b000001000
GRAVITY = 0b000010000
RTYPE = 0b000100000  # Fixed rotation and translation
ROTVTYPE = 0b001000000  # Enforce rotation velocity
ROTATYPE = 0b010000000  # Enforce rotation acceleration
TORQUETYPE = 0b100000000  # Enforce torque


iVec3 = ti.types.vector(3, ti.i32)
iVec4 = ti.types.vector(4, ti.i32)
fVec2 = ti.types.vector(2, ti.f32)
fVec3 = ti.types.vector(3, ti.f32)
fVec4 = ti.types.vector(4, ti.f32)
Mat2x2 = ti.types.matrix(2, 2, ti.f32)
Mat3x3 = ti.types.matrix(3, 3, ti.f32)
Mat3x2 = ti.types.matrix(3, 2, ti.f32)
Mat2x3 = ti.types.matrix(2, 3, ti.f32)
Mat4x3 = ti.types.matrix(4, 3, ti.f32)
Mat3x4 = ti.types.matrix(3, 4, ti.f32)
Mat4x4 = ti.types.matrix(4, 4, ti.f32)
Mat3x6 = ti.types.matrix(3, 6, ti.f32)
Mat6x12 = ti.types.matrix(6, 12, ti.f32)
