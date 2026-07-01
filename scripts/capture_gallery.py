#!/usr/bin/env python3
"""Capture gallery screenshots from renderable test2D demos.

Usage:
    python scripts/capture_gallery.py
    python scripts/capture_gallery.py --only fem-elastic,domino
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GALLERY_DIR = ROOT / "docs" / "gallery"
MANIFEST_PATH = GALLERY_DIR / "manifest.json"

# Each entry: slug, import path (under test2D), callable, kwargs, capture frame, max show() calls
GALLERY_ENTRIES: list[dict] = [
    {
        "slug": "fem-elastic",
        "module": "test_2Dfem_elastic",
        "func": "test_2Dfem_elastic",
        "title_en": "FEM elastic square",
        "title_zh": "FEM 弹性方块",
        "capture": 28,
        "max_frames": 40,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-contact",
        "module": "test_2Dfem_contact",
        "func": "test_2Dfem_contact",
        "title_en": "FEM self-contact",
        "title_zh": "FEM 自接触",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-bcs",
        "module": "test_2Dfem_bcs",
        "func": "test_2Dfem_elastic",
        "title_en": "FEM boundary conditions",
        "title_zh": "FEM 边界条件",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-two-domains",
        "module": "test_2Dfem2domain_elastic",
        "func": "test_2Dfem2domain_elastic",
        "title_en": "Two FEM domains",
        "title_zh": "双 FEM 域",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-ics",
        "module": "test_2Dfem_ics",
        "func": "test_2Dfem_elastic",
        "title_en": "FEM initial conditions",
        "title_zh": "FEM 初始条件",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-analytic-contact",
        "module": "test_2Dfemanalytic_contact",
        "func": "test_2Dfem_contact",
        "title_en": "FEM vs analytical ground",
        "title_zh": "FEM 与解析地面",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-heightfield",
        "module": "test_2Dfem_heightfield_contact",
        "func": "test_2Dfem_heightfield_contact",
        "title_en": "FEM on height field",
        "title_zh": "FEM 高度场接触",
        "capture": 40,
        "max_frames": 60,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-voxel",
        "module": "test_2Dfem_voxel_contact",
        "func": "test_2Dfem_voxel_contact",
        "title_en": "FEM in voxel pit",
        "title_zh": "FEM 体素坑接触",
        "capture": 40,
        "max_frames": 60,
        "kwargs": {"headless": False},
    },
    {
        "slug": "fem-rigid-mixed",
        "module": "test_2dfemrigid_contact",
        "func": "test_2Dfemrigid_contact",
        "title_en": "FEM + rigid bodies",
        "title_zh": "FEM 与刚体混合",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "spring-mass",
        "module": "test_2Dspring_bcs",
        "func": "test_2Dspring",
        "title_en": "Spring-mass lattice",
        "title_zh": "弹簧质点网格",
        "capture": 8,
        "max_frames": 11,
        "kwargs": {"headless": False},
    },
    {
        "slug": "spring-mass-10k",
        "module": "test_2Dspring_10000",
        "func": "test_2Dspring",
        "title_en": "Large spring mesh (10k)",
        "title_zh": "大规模弹簧网格",
        "capture": 4,
        "max_frames": 6,
        "kwargs": {"headless": False},
    },
    {
        "slug": "mesh-rigid",
        "module": "test_2Dmeshrigid",
        "func": "test_2Drigid_contact",
        "title_en": "Mesh rigid bodies",
        "title_zh": "网格刚体",
        "capture": 25,
        "max_frames": 45,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-balls",
        "module": "test_2Drigid_contact",
        "func": "test_2Drigid_contact",
        "title_en": "Ball-ball contact",
        "title_zh": "球体接触",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-100-balls",
        "module": "test_2Drigid_contact100",
        "func": "test_2Drigid_contact",
        "title_en": "100 balls",
        "title_zh": "百球堆积",
        "capture": 8,
        "max_frames": 12,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-25-boxes",
        "module": "test_2Drigid_contact100_boxes",
        "func": "test_2Drigid_contact",
        "title_en": "25 boxes",
        "title_zh": "25 盒子碰撞",
        "capture": 6,
        "max_frames": 10,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-capsules",
        "module": "test_2Drigid_contact100_capsules",
        "func": "test_2Drigid_contact",
        "title_en": "Capsule crowd",
        "title_zh": "胶囊刚体群",
        "capture": 8,
        "max_frames": 12,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-ground",
        "module": "test_2Drigidanalytic_contact",
        "func": "test_2Drigidanal_contact",
        "title_en": "Rigid on analytical planes",
        "title_zh": "刚体解析地面",
        "capture": 12,
        "max_frames": 17,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-bcs",
        "module": "test_2Drigid_bcs",
        "func": "test_2Drigid_bcs",
        "title_en": "Rigid body BCs",
        "title_zh": "刚体边界条件",
        "capture": 5,
        "max_frames": 8,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-rotation",
        "module": "test_2Drigid_rot",
        "func": "test_2Drigidrot",
        "title_en": "Rigid rotation",
        "title_zh": "刚体旋转",
        "capture": 25,
        "max_frames": 45,
        "kwargs": {"headless": False},
    },
    {
        "slug": "box-friction-dynamic",
        "module": "test_2Dbox_friction",
        "func": "_run_2d_box_friction",
        "title_en": "Box friction (dynamic)",
        "title_zh": "盒子摩擦（滑动）",
        "capture": 50,
        "max_frames": 65,
        "kwargs": {"force_x": 4.0, "end_time": 1.1, "headless": False},
    },
    {
        "slug": "box-friction-inclined",
        "module": "test_2Dbox_friction",
        "func": "_run_2d_box_friction_inclined",
        "title_en": "Box on inclined plane",
        "title_zh": "斜面盒子摩擦",
        "capture": 50,
        "max_frames": 65,
        "kwargs": {"angle": 30.0, "end_time": 1.1, "headless": False},
    },
    {
        "slug": "domino",
        "module": "test_2Ddomino",
        "func": "test_2drigid_domino",
        "title_en": "Domino chain",
        "title_zh": "多米诺骨牌",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "pendulum",
        "module": "test_2Dpendulum_simple",
        "func": "test_two_body_joint",
        "title_en": "Revolute pendulum",
        "title_zh": "铰接摆",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "revolute-joint",
        "module": "test_2Drevolute_joint",
        "func": "test_revolute_joint",
        "title_en": "Revolute joint",
        "title_zh": "旋转关节",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False, "kernel_profile": False},
    },
    {
        "slug": "joint-bcs",
        "module": "test_2Djoint_bcs",
        "func": "test_2Djoint_rotation",
        "title_en": "Joint rotation BCs",
        "title_zh": "关节旋转边界",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "joint-pd-velocity",
        "module": "test_2Djoint_bcs_pdvel",
        "func": "test_2Djoint_rotation",
        "title_en": "Joint PD velocity",
        "title_zh": "关节 PD 速度控制",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "linkage",
        "module": "test_2Dlinkage",
        "func": "test_linkage",
        "title_en": "Four-bar linkage",
        "title_zh": "四连杆机构",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False, "kernel_profile": False},
    },
    {
        "slug": "robot",
        "module": "test_2Drobot",
        "func": "test_robot",
        "title_en": "2D robot arm",
        "title_zh": "二维机械臂",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False, "kernel_profile": False},
    },
    {
        "slug": "wheel-rolling",
        "module": "test_2D_wheel_rolling",
        "func": "test_cylinder_wheel_rolling",
        "title_en": "Cylinder wheel rolling",
        "title_zh": "圆柱轮滚动",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "wheel-rolling-initials",
        "module": "test_2Dwheel_rolling_initials",
        "func": "test_cylinder_wheel_rolling",
        "title_en": "Wheel rolling (initial spin)",
        "title_zh": "带初速的滚轮",
        "capture": 35,
        "max_frames": 55,
        "kwargs": {"headless": False},
    },
    {
        "slug": "solar-system",
        "module": "test_2Dsolar_system",
        "func": "test_2Dsolar_system",
        "title_en": "Solar-system rigid orbits",
        "title_zh": "刚体轨道（太阳系）",
        "capture": 25,
        "max_frames": 45,
        "kwargs": {"headless": False},
    },
    {
        "slug": "analytical-bcs",
        "module": "test_2Danalytical_bcs",
        "func": "test_2DAnalytical_bcs",
        "title_en": "Analytical domain BCs",
        "title_zh": "解析域边界条件",
        "capture": 1,
        "max_frames": 1,
        "kwargs": {"headless": False},
    },
    {
        "slug": "joint-pd-force",
        "module": "test_2Djoint_bcs_pdforce",
        "func": "test_2Djoint_rotation",
        "title_en": "Joint PD force control",
        "title_zh": "关节 PD 力控制",
        "capture": 30,
        "max_frames": 50,
        "kwargs": {"headless": False},
    },
    {
        "slug": "rigid-rotation-benchmark",
        "module": "test_2Drigid_rotation_benchmark",
        "func": "test_2Drigid_rotation_benchmark",
        "title_en": "Rigid rotation benchmark",
        "title_zh": "刚体旋转基准",
        "capture": 20,
        "max_frames": 35,
        "kwargs": {"headless": False},
    },
    {
        "slug": "analytical-rotation-benchmark",
        "module": "test_2Danalytical_rotation_benchmark",
        "func": "test_2Danalytical_rotation",
        "title_en": "Analytical rotation benchmark",
        "title_zh": "解析域旋转基准",
        "capture": 20,
        "max_frames": 35,
        "kwargs": {"headless": False},
    },
]


def capture_one(entry: dict) -> bool:
    slug = entry["slug"]
    output = GALLERY_DIR / f"{slug}.png"
    print(f"[gallery] Capturing {slug} -> {output.name}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "capture_gallery_single.py"),
        "--output",
        str(output),
        "--capture-frame",
        str(entry["capture"]),
        "--max-frames",
        str(entry["max_frames"]),
        "--module",
        entry["module"],
        "--func",
        entry["func"],
        "--kwargs",
        json.dumps(entry.get("kwargs", {})),
    ]

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        return False
    if not output.exists():
        print(f"[gallery] Missing output for {slug}")
        return False
    print(f"[gallery] OK {slug}")
    return True


def write_manifest(entries: list[dict], succeeded: list[str] | None = None) -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for entry in entries:
        png = GALLERY_DIR / f"{entry['slug']}.png"
        if succeeded is not None and entry["slug"] not in succeeded:
            if not png.exists():
                continue
        elif not png.exists():
            continue
        manifest.append(
            {
                "slug": entry["slug"],
                "file": f"{entry['slug']}.png",
                "title_en": entry["title_en"],
                "title_zh": entry["title_zh"],
                "source": f"test2D/{entry['module']}.py",
            }
        )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Capture test2D gallery screenshots")
    parser.add_argument("--only", type=str, default="", help="Comma-separated slugs to capture")
    args = parser.parse_args()

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    entries = [e for e in GALLERY_ENTRIES if not only or e["slug"] in only]

    succeeded: list[str] = []
    failed: list[str] = []
    for entry in entries:
        try:
            if capture_one(entry):
                succeeded.append(entry["slug"])
            else:
                failed.append(entry["slug"])
        except Exception:
            traceback.print_exc()
            failed.append(entry["slug"])

    write_manifest(GALLERY_ENTRIES, succeeded=None)
    print(f"\n[gallery] Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print("[gallery] Failed:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
