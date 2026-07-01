# Flat World Gallery

Screenshots captured from **live Taichi rendering** of `test2D` demos (off-screen `ti.GUI`, one frame per scene).

## Regenerate

```bash
python scripts/capture_gallery.py
# single demo:
python scripts/capture_gallery.py --only domino,fem-elastic
```

Requires a working Taichi backend (CPU/CUDA). Images are written to this folder and `manifest.json` is updated automatically.

All scenes use a unified background color: `0x112F41`.

## Scenes

| Preview | Title | Source test |
|---------|-------|-------------|
| ![FEM elastic](fem-elastic.png) | FEM elastic square | `test_2Dfem_elastic.py` |
| ![FEM contact](fem-contact.png) | FEM self-contact | `test_2Dfem_contact.py` |
| ![FEM BCs](fem-bcs.png) | FEM boundary conditions | `test_2Dfem_bcs.py` |
| ![Two FEM domains](fem-two-domains.png) | Two FEM domains | `test_2Dfem2domain_elastic.py` |
| ![FEM ICs](fem-ics.png) | FEM initial conditions | `test_2Dfem_ics.py` |
| ![FEM analytic](fem-analytic-contact.png) | FEM vs analytical ground | `test_2Dfemanalytic_contact.py` |
| ![FEM height field](fem-heightfield.png) | FEM on height field | `test_2Dfem_heightfield_contact.py` |
| ![FEM voxel](fem-voxel.png) | FEM in voxel pit | `test_2Dfem_voxel_contact.py` |
| ![FEM rigid](fem-rigid-mixed.png) | FEM + rigid bodies | `test_2dfemrigid_contact.py` |
| ![Spring mass](spring-mass.png) | Spring-mass lattice | `test_2Dspring_bcs.py` |
| ![Spring 10k](spring-mass-10k.png) | Large spring mesh | `test_2Dspring_10000.py` |
| ![Mesh rigid](mesh-rigid.png) | Mesh rigid bodies | `test_2Dmeshrigid.py` |
| ![Ball contact](rigid-balls.png) | Ball-ball contact | `test_2Drigid_contact.py` |
| ![100 balls](rigid-100-balls.png) | 100 balls | `test_2Drigid_contact100.py` |
| ![25 boxes](rigid-25-boxes.png) | 25 boxes | `test_2Drigid_contact100_boxes.py` |
| ![Capsules](rigid-capsules.png) | Capsule crowd | `test_2Drigid_contact100_capsules.py` |
| ![Rigid ground](rigid-ground.png) | Rigid on analytical planes | `test_2Drigidanalytic_contact.py` |
| ![Rigid BCs](rigid-bcs.png) | Rigid body BCs | `test_2Drigid_bcs.py` |
| ![Rigid rotation](rigid-rotation.png) | Rigid rotation | `test_2Drigid_rot.py` |
| ![Box friction](box-friction-dynamic.png) | Box friction (dynamic) | `test_2Dbox_friction.py` |
| ![Inclined friction](box-friction-inclined.png) | Box on inclined plane | `test_2Dbox_friction.py` |
| ![Domino](domino.png) | Domino chain | `test_2Ddomino.py` |
| ![Pendulum](pendulum.png) | Revolute pendulum | `test_2Dpendulum_simple.py` |
| ![Revolute joint](revolute-joint.png) | Revolute joint | `test_2Drevolute_joint.py` |
| ![Joint BCs](joint-bcs.png) | Joint rotation BCs | `test_2Djoint_bcs.py` |
| ![Joint PD velocity](joint-pd-velocity.png) | Joint PD velocity | `test_2Djoint_bcs_pdvel.py` |
| ![Joint PD force](joint-pd-force.png) | Joint PD force | `test_2Djoint_bcs_pdforce.py` |
| ![Linkage](linkage.png) | Four-bar linkage | `test_2Dlinkage.py` |
| ![Robot](robot.png) | 2D robot arm | `test_2Drobot.py` |
| ![Wheel](wheel-rolling.png) | Cylinder wheel rolling | `test_2D_wheel_rolling.py` |
| ![Wheel spin](wheel-rolling-initials.png) | Wheel with initial spin | `test_2Dwheel_rolling_initials.py` |
| ![Solar system](solar-system.png) | Solar-system orbits | `test_2Dsolar_system.py` |
| ![Analytical BCs](analytical-bcs.png) | Analytical domain BCs | `test_2Danalytical_bcs.py` |
| ![Rigid rotation bench](rigid-rotation-benchmark.png) | Rigid rotation benchmark | `test_2Drigid_rotation_benchmark.py` |
| ![Analytical rotation bench](analytical-rotation-benchmark.png) | Analytical rotation benchmark | `test_2Danalytical_rotation_benchmark.py` |

## Notes

- Demos with GUI code commented out (e.g. `test_2Dfem_elastic10000.py`, `test_2Drigid_contact100_mesh.py`) are not included.
- `test_2Drigid_bcs.py` sets `BENCHMARK_MODE = True` by default; the capture script forces visualization on for screenshots.
