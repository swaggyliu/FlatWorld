import abc
import numpy as np
import taichi as ti


class DomainBase(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
    ):
        pass
