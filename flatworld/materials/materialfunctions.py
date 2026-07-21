from definitions import *
import warp as wp


class MaterialBase:
    def __init__(self):
        pass


@wp.func
def getStress2D(type: int, E: float, nu: float, epsilon: wp.vec2, F: wp.mat22):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    stress_voigt = wp.vec3(0.0, 0.0, 0.0)
    I = wp.identity(n=2, dtype=wp.float32)
    strain = 0.5 * (wp.transpose(F) @ F - I)
    strain_voigt = wp.vec3(strain[0, 0], strain[1, 1], strain[0, 1])
    if type == MaterialType.ELASTIC:
        # Linear Elastic Material
        stress = (lam * wp.trace(strain)) * I + 2.0 * mu * strain
        stress_voigt = wp.vec3(stress[0, 0], stress[1, 1], stress[0, 1])

    elif type == MaterialType.NEOHOOKEAN:
        # Use Neo Hookean formulation
        Cinv = wp.inverse(wp.transpose(F) @ F)
        J = wp.determinant(F)
        stress = (lam * wp.log(J) - mu) * Cinv + mu * I
        stress_voigt = wp.vec3(stress[0, 0], stress[1, 1], stress[0, 1])
    return stress_voigt, strain_voigt


# ── Von Mises J2 plasticity – radial return mapping ──────────────────


@wp.func
def misesReturnMap2D(
    E: float, nu: float, sigma_y: float, H: float, F: wp.mat22, eps_p_old: wp.vec3, eqps_old: float
):
    """2D plane-strain von Mises return mapping (small-strain Green-Lagrange).

    Args:
        E, nu: elastic constants
        sigma_y: initial yield stress
        H: linear isotropic hardening modulus
        F: 2x2 deformation gradient
        eps_p_old: previous plastic strain (Voigt 3-vector: eps_xx, eps_yy, eps_xy)
        eqps_old: previous equivalent plastic strain scalar

    Returns:
        stress_voigt (3,): updated PK2 stress in Voigt form
        eps_p_new (3,):   updated plastic strain Voigt
        eqps_new:         updated equivalent plastic strain
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    # Total Green-Lagrange strain
    I = wp.identity(n=2, dtype=wp.float32)
    strain = 0.5 * (wp.transpose(F) @ F - I)
    eps_total = wp.vec3(strain[0, 0], strain[1, 1], strain[0, 1])

    # Elastic trial strain = total - plastic_old
    eps_e_trial = eps_total - eps_p_old

    # Trial stress (PK2, Voigt)
    tr_ee = eps_e_trial[0] + eps_e_trial[1]  # trace (plane-strain: eps_zz_e = 0)
    s_trial = wp.vec3(
        lam * tr_ee + 2.0 * mu * eps_e_trial[0],
        lam * tr_ee + 2.0 * mu * eps_e_trial[1],
        2.0 * mu * eps_e_trial[2],  # shear uses engineering strain
    )

    # Hydrostatic & deviatoric (2D plane-strain: σ_zz = ν(σ_xx+σ_yy) for elastic)
    p = (s_trial[0] + s_trial[1]) / 3.0  # approximate mean for plane stress
    dev = wp.vec3(s_trial[0] - p, s_trial[1] - p, s_trial[2])
    # von Mises equivalent stress  σ_vm = sqrt(dev_xx^2 + dev_yy^2 - dev_xx*dev_yy + 3*τ_xy^2)
    vm = wp.sqrt(dev[0] * dev[0] + dev[1] * dev[1] - dev[0] * dev[1] + 3.0 * dev[2] * dev[2])

    # Yield check
    f_trial = vm - (sigma_y + H * eqps_old)

    eps_p_new = eps_p_old
    eqps_new = eqps_old
    stress_voigt = s_trial

    if f_trial > 0.0:
        # Plastic correction – radial return
        denom = 3.0 * mu + H
        dgamma = f_trial / denom  # plastic multiplier
        eqps_new = eqps_old + dgamma

        # Scale deviatoric stress back to yield surface
        scale = 1.0 - 3.0 * mu * dgamma / vm
        dev_new = dev * scale
        stress_voigt = wp.vec3(dev_new[0] + p, dev_new[1] + p, dev_new[2])

        # Update plastic strain (Voigt, flow direction = dev/vm)
        n_hat = dev / vm  # unit normal
        eps_p_new = eps_p_old + dgamma * wp.vec3(n_hat[0], n_hat[1], n_hat[2])  # γ_xy component

    return stress_voigt, eps_p_new, eqps_new, eps_total


# ── Plane-stress constitutive law for membrane elements ──────────────


@wp.func
def getStressPlaneStress2D(type: int, E: float, nu: float, F: wp.mat22):
    """PK2 stress for a membrane element under plane-stress assumption.

    Uses the plane-stress Lamé parameter:
        λ_ps = E*ν / (1 - ν²)
    instead of the plane-strain λ = E*ν / ((1+ν)(1-2ν)).
    μ remains E / (2(1+ν)).

    Args:
        type: MaterialType enum (ELASTIC or NEOHOOKEAN)
        E, nu: Young's modulus and Poisson's ratio
        F: 2×2 in-plane deformation gradient

    Returns:
        stress_voigt (3,): PK2 stress in Voigt form [S11, S22, S12]
    """
    lam_ps = E * nu / (1.0 - nu * nu)
    mu = E / (2.0 * (1.0 + nu))
    stress_voigt = wp.vec3(0.0, 0.0, 0.0)
    I = wp.identity(n=2, dtype=wp.float32)

    if type == MaterialType.ELASTIC:
        strain = 0.5 * (wp.transpose(F) @ F - I)
        stress = lam_ps * wp.trace(strain) * I + 2.0 * mu * strain
        stress_voigt = wp.vec3(stress[0, 0], stress[1, 1], stress[0, 1])

    elif type == MaterialType.NEOHOOKEAN:
        C = wp.transpose(F) @ F
        Cinv = wp.inverse(C)
        J2D = wp.determinant(F)
        # Plane-stress Neo-Hookean: compressible approximation with plane-stress λ
        stress = (lam_ps * wp.log(J2D) - mu) * Cinv + mu * I
        stress_voigt = wp.vec3(stress[0, 0], stress[1, 1], stress[0, 1])

    return stress_voigt
