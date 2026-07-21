import abc
from definitions import *
import numpy as np


def _patch_array(arr, index, value):
    """Host write for Warp arrays (no ``arr[i] =`` from Python on Warp 1.14+)."""
    if hasattr(arr, "numpy") and hasattr(arr, "assign"):
        np_arr = arr.numpy()
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            value = float(np.asarray(value).reshape(-1)[0]) if not isinstance(value, (str, bytes)) else value
        elif hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            a = np.asarray(value, dtype=np.float32).reshape(-1)
            if a.size == 1:
                value = float(a[0])
        np_arr[index] = value
        arr.assign(np_arr)
    else:
        arr[index] = value


class IntialConditionBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass


class InitialVel(IntialConditionBase):
    def __init__(self, nds, vel):
        # Support "ALL" keyword for applying to all nodes
        if nds == "ALL":
            self.is_all_nodes = True
            self.nds = None
        else:
            self.is_all_nodes = False
            self.nds = np.asarray(nds, dtype=np.int32)
        self.vel = np.asarray(vel, dtype=np.float32)
        self.type = VTYPE

    def update(self, V, dim, offset, num_nodes):
        """Write initial velocity into manager array ``V`` (Warp or host)."""
        if self.is_all_nodes:
            for i in range(int(num_nodes)):
                _patch_array(V, i + int(offset), self.vel)
        else:
            for nid in self.nds:
                _patch_array(V, int(nid) + int(offset), self.vel)


class InitialAngVel(IntialConditionBase):
    def __init__(self, nds, ang_vel, ref_point=None):
        # Support "ALL" keyword for applying to all nodes
        if nds == "ALL":
            self.is_all_nodes = True
            self.nds = None
        else:
            self.is_all_nodes = False
            self.nds = np.asarray(nds, dtype=np.int32)
        self.ang_vel = np.asarray(ang_vel, dtype=np.float32)
        self.ref_point = np.asarray(ref_point, dtype=np.float32) if ref_point is not None else None
        self.type = ROTVTYPE

    def update(self, RotV, dim, offset, num_nodes):
        """Write initial angular velocity into manager array ``RotV``."""
        if self.is_all_nodes:
            for i in range(int(num_nodes)):
                _patch_array(RotV, i + int(offset), self.ang_vel)
        else:
            for nid in self.nds:
                _patch_array(RotV, int(nid) + int(offset), self.ang_vel)
