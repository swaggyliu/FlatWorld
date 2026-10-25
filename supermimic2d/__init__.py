"""SuperMimic-2D: FlatWorld recreation of SuperMimic for planar dance imitation.

Training loop mirrors SuperMimic (CoRL 2026):
  (1) collect simulator transitions with the current policy
  (2) fit a differentiable world model on replay windows
  (3) improve the policy by BPTT through short imagined rollouts
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
