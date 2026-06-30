import os
import sys

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir in sys.path:
    sys.path.remove(package_dir)
sys.path.insert(0, package_dir)

from grounddomain import GroundDomain
from bcs import EnforceAcc, EnforceRotAcc, EnforceRotVel, EnforceVel, Fixed, FixedAll, Force, Gravity, Torque
from definitions import *
from explicitloop import ExplicitLoop
from femdomain import FemDomain
from heightfielddomain import HeightFieldDomain
from initialconditions import InitialAngVel, InitialVel
from joints import PrismaticJoint, RevoluteJoint, SphericalJoint, WeldJoint
from materials import Elastic, HyperElastic, MisesPlastic
from mesh import Mesh
from rigid import BallRigid, BoxRigid, CapsuleRigid, MeshRigid, Transform
from rigiddomain import RigidBodyDomain
from solidprop import SolidProp
from springmass import SpringMassDomain
from voxeldomain import VoxelGridDomain

try:
    from femesher import FEMesher
except ImportError:
    FEMesher = None
