# Contact Dynamics AI v1

This package adds an opt-in, solver-supervised local contact-response model to FlatWorld.

## Recorded labels

Use the collector option --record-contact-transitions. For the last PGS substep in each visual frame, the recorder stores:

- contact-point normal/tangent relative velocity before PGS;
- penetration depth, friction, restitution, inverse effective mass, and restitution target velocity;
- solved normal and tangential PGS impulses in N·s;
- a two-class friction-saturation label (unsaturated / saturated).

The existing simulator path is unchanged when recording is disabled. Recording intentionally synchronizes Warp arrays to the host and is intended for offline data generation.

## Collect and train

    python -m learning.data.collect --task push_t --contact-rich \
        --record-contact-transitions --num-rollouts 100 \
        --out learning/data/contact_dynamics

    python -m learning.contact_dynamics.train \
        --data learning/data/contact_dynamics \
        --out learning/checkpoints/contact_dynamics --epochs 40

Train/validation splitting is by rollout file, avoiding leakage between contact rows from the same trajectory.

## Model contract

ContactImpulseNet(features, friction) predicts (lambda_n, lambda_t) and a friction-saturation logit. The model enforces lambda_n >= 0 and |lambda_t| <= mu * lambda_n by parameterization. It assumes that a contact candidate has already been detected; it does not replace broadphase, collision detection, or PGS.

## Evaluation boundary

V1 labels come from FlatWorld's own PGS solver. Validation MAE therefore measures solver-response imitation, not superiority in physical accuracy. It establishes a learned contact-response baseline for later PGS warm starts or residual correction. Claims of higher physical accuracy require independent reference data (such as instrumented impacts or a separately validated high-precision solver) and equal end-to-end latency budgets.
