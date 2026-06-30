import numpy as np
import os
import sys
import taichi as ti

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from flatworld import BallRigid, Elastic, ExplicitLoop, FemDomain, FEMesher, Gravity, Mesh, RigidBodyDomain, SolidProp
# from learning import RigidExport
# from test_utils import create_gui_if_available


# def test_2Drigid_contact(headless=False):

#     ti.init(offline_cache=True, arch=ti.cpu)
#     export = RigidExport()

#     rigid1 = BallRigid(2, [0.5, 0.8], 0.1, 1.0)
#     bcs = [Gravity([0, -10.0])]
#     do1 = RigidBodyDomain(rigid1, bcs)

#     frame_dt = 1.0 / 60.0

#     rigid2 = BallRigid(2, [0.5, 0.3], 0.1, 1.0)
#     bcs = [Gravity([0, 10.0])]
#     do2 = RigidBodyDomain(rigid2, bcs)

#     domains = [do1, do2]
#     looper = ExplicitLoop(0.0, domains, useAdapativeDT=True)

#     export.exportRigidInput(do1, "2Drigid_in.csv")
#     export.exportRigidInput(do2, "2Drigid_in.csv")
#     gui = create_gui_if_available("FEM2D", res=(1080, 1080)) if not headless else None
#     t = 0.0
#     times = []
#     counter = 0
#     while counter < 100:
#         # advance exactly one visual frame using adaptive substeps
#         looper.advanceWithTime(frame_dt)
#         t += frame_dt
#         times.append(t)
#         counter += 1

#         nnd = 0
#         color = 0xAAFFFF
#         for i, rigiddomain in enumerate(looper.domains):
#             pos = rigiddomain.getCurrentRefPoint()
#             if gui is not None:
#                 gui.circle(pos, color, 108)
#             export.exportRigidOutput(rigiddomain, "2Drigid_out.csv")

#         export.outputcounter = 0
#         if gui is not None:
#             gui.show()
#     export.exportTimes(times, "2Drigid_time.csv")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="test_2Drigid_contact")
#     parser.add_argument(
#         "--headless", action="store_true", help="Run without GUI (auto-detected if no display available)"
#     )
#     args = parser.parse_args()
#     test_2Drigid_contact(headless=args.headless)
