"""Versioned schema for solver-supervised contact transitions."""

SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "normal_relative_velocity",
    "tangent_relative_velocity",
    "penetration_depth",
    "friction_coefficient",
    "restitution_coefficient",
    "normal_inverse_effective_mass",
    "tangent_inverse_effective_mass",
    "restitution_target_velocity",
)
IMPULSE_NAMES = ("normal_impulse", "tangent_impulse")
MODE_NAMES = ("stick_or_unsaturated", "friction_saturated")


def validate_feature_array(features):
    """Validate the final feature dimension without importing torch."""
    shape = getattr(features, "shape", None)
    if shape is None or len(shape) < 1 or shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"contact features must end in dimension {len(FEATURE_NAMES)}")
    return features
