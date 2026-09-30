"""Dump all recorded eval episodes to MP4 (one file per episode)."""

import os

from learning import replay as replay_mod
from learning.replay import dump_mp4s, load_pack

TASKS = ("two_room", "push_t", "reacher")


def _wide_view_init(self, width: int, height: int, hidden: bool = False, fps: int = 0):
    replay_mod.ReplayView.__dict__["_orig_init"](self, width, height, hidden, fps)
    # Fit two-room walls and a Push-T EE that drifts past x=3.
    self.cam.scale = 300.0
    self.cam.ox = 40.0
    self.cam.oy = height - 56.0


def main():
    if "_orig_init" not in replay_mod.ReplayView.__dict__:
        replay_mod.ReplayView._orig_init = replay_mod.ReplayView.__init__
        replay_mod.ReplayView.__init__ = _wide_view_init
    packs = []
    for name in TASKS:
        path = os.path.join("learning", "results", name, "task_eval.json")
        pack = load_pack(path, collect_missing=False)
        for e in pack.episodes:
            e.tag = name
        print(f"{name}: {len(pack.episodes)} episodes")
        packs.append(pack)
    dump_mp4s(
        packs,
        out_dir=os.path.join("learning", "results", "replays"),
        size="1280x720",
        fps=30,
        skip=2,
        hold=24,
        vel_ticks=True,
    )


if __name__ == "__main__":
    main()
