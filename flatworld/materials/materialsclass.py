from definitions import *
import taichi as ti


@ti.data_oriented
class Elastic:
    def __init__(self, E, nu, rho):
        self.type = MaterialType.ELASTIC
        self.E = E
        self.nu = nu
        self.rho = rho

    @ti.func
    def getRepresentativeModulus(self):
        return self.E


@ti.data_oriented
class HyperElastic:
    def __init__(self, E, nu, rho):
        self.type = MaterialType.NEOHOOKEAN
        self.E = E
        self.nu = nu
        self.rho = rho

    @ti.func
    def getRepresentativeModulus(self):
        return self.E


@ti.data_oriented
class MisesPlastic:
    """J2 von Mises plasticity with linear isotropic hardening.

    Parameters:
        E:  Young's modulus
        nu: Poisson's ratio
        rho: density
        sigma_y: initial yield stress
        H: linear hardening modulus (tangent modulus of the plastic regime)
    """

    def __init__(self, E, nu, rho, sigma_y, H=0.0):
        self.type = MaterialType.MISES
        self.E = E
        self.nu = nu
        self.rho = rho
        self.sigma_y = sigma_y
        self.H = H

    @ti.func
    def getRepresentativeModulus(self):
        return self.E
