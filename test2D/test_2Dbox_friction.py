import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from flatworld import GroundDomain, BoxRigid, ExplicitLoop, Force, Gravity, RigidBodyDomain
from test_utils import create_gui_if_available, init_sim


def _run_2d_box_friction(force_x, end_time=5.0, headless=True, write_output=False):
    """Run a 2D box-on-plane friction scene and return the final position."""
    init_sim()

    center = np.array([0.0, 0.05])
    mass = 1.0
    box = BoxRigid(2, center, [0.1, 0.1], [0, 0], mass)
    bcs = [Gravity([0.0, -10.0]), Force([0], [force_x, 0.0])]

    plane_point = [0.0, 0.0]
    plane_normal = [0.0, 1.0]
    analytical = GroundDomain(2, plane_point, plane_normal, [])

    domains = [analytical]
    domains += [RigidBodyDomain(box, bcs, friction=0.2)]

    frame_dt = 1.0 / 60.0

    gui = create_gui_if_available("2D Box Friction", res=(800, 800)) if not headless else None
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)
    colors = [0x888888, 0xFF3333]

    counter = 0
    sim_time = 0.0
    output = open("output.txt", "w") if write_output else None

    while sim_time < end_time:
        looper.advanceWithTime(frame_dt)
        sim_time += frame_dt
        counter += 1

        if gui is not None:
            gui.clear(0x112F41)

            # Draw all rigid bodies
            for i, domain in enumerate(domains):
                domain.draw(gui, color=colors[i], resolution=800)

            gui.show()

        pos = domains[1].getCurrentRefPoint()
        if output is not None:
            output.write(f"{sim_time:.3f} {pos[0]:.3f} {pos[1]:.3f}\n")

    if output is not None:
        output.close()

    return pos


def _run_2d_box_friction_inclined(angle=0.0, end_time=5.0, headless=True):
    """Run a 2D box-on-inclined-plane friction scene and return the final position."""
    init_sim()

    center = np.array([0.5, 0.5 * np.tan(np.radians(angle)) + 0.05 / np.cos(np.radians(angle))])
    mass = 1.0
    box = BoxRigid(2, center, [0.1, 0.1], [np.radians(angle)], mass)
    bcs = [Gravity([0.0, -10.0])]

    plane_point = [0.0, 0.0]
    plane_normal = [-np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    analytical = GroundDomain(2, plane_point, plane_normal, [])

    domains = [analytical]
    domains += [RigidBodyDomain(box, bcs, friction=0.6, restitution=0.5)]

    frame_dt = 1.0 / 60.0

    gui = create_gui_if_available("2D Box Friction Inclined", res=(800, 800)) if not headless else None
    looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)
    colors = [0x888888, 0xFF3333]

    sim_time = 0.0

    while sim_time < end_time:
        looper.advanceWithTime(frame_dt)
        sim_time += frame_dt

        if gui is not None:
            gui.clear(0x112F41)

            # Draw all rigid bodies
            for i, domain in enumerate(domains):
                domain.draw(gui, color=colors[i], resolution=800)

            gui.show()

    pos = domains[1].getCurrentRefPoint()
    return pos


def test_2Dbox_friction_static():
    pos = _run_2d_box_friction(force_x=1.0)
    assert pos[0] < 0.05, f"2D static-friction case drifted too much: x={pos[0]:.6f}"
    assert abs(pos[1] - 0.055) < 1e-2, f"2D box should stay near resting height: y={pos[1]:.6f}"


def test_2Dbox_friction_dynamic():
    pos = _run_2d_box_friction(force_x=4.0)
    assert pos[0] > 5.0, f"2D dynamic-friction case did not slide enough: x={pos[0]:.6f}"
    assert abs(pos[1] - 0.055) < 1e-2, f"2D box should stay near resting height: y={pos[1]:.6f}"


def test_2Dbox_friction_inclined_static():
    pos = _run_2d_box_friction_inclined(angle=30.0)
    assert abs(pos[0] - 0.5) < 1e-2, f"2D inclined-friction case drifted too much: x={pos[0]:.6f}"
    assert abs(pos[1] - 0.34641016) < 1e-2, f"2D inclined-friction case drifted too much: y={pos[1]:.6f}"


def test_2Dbox_friction_inclined_dynamic():
    pos = _run_2d_box_friction_inclined(angle=45.0)
    assert np.allclose(
        pos, [-24.5, -24.443], rtol=0.01
    ), f"2D inclined-friction case did not slide enough: x={pos[0]:.6f}, y={pos[1]:.6f}"


if __name__ == "__main__":
    static_pos = _run_2d_box_friction(force_x=1.0, headless=False, write_output=True)
    print(f"2D static-friction final position: {static_pos}")
    dynamic_pos = _run_2d_box_friction(force_x=4.0, headless=False)
    print(f"2D dynamic-friction final position: {dynamic_pos}")
    inclined_pos = _run_2d_box_friction_inclined(angle=30.0, headless=False)
    print(f"2D inclined-friction final position: {inclined_pos}")
    inclined_pos = _run_2d_box_friction_inclined(angle=45.0, headless=False)
    print(f"2D inclined-friction final position: {inclined_pos}")
