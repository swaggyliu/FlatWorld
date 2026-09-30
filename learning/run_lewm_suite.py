"""Collect, train, evaluate, and report the three StateLeWM scenes.

Usage (repo root):
    python -m learning.run_lewm_suite --stage all
    python -m learning.run_lewm_suite --stage collect --task two_room
"""

import argparse
import os
import subprocess
import sys

TASKS = ("two_room", "push_t", "reacher")


def _run(args):
    print("+", " ".join(args), flush=True)
    subprocess.check_call(args)


def collect(task, n, episode_len):
    _run([sys.executable, "-m", "learning.data.collect",
          "--task", task, "--num-rollouts", str(n),
          "--episode-len", str(episode_len)])


def train(task, epochs, ensemble):
    data = f"learning/data/{task}_500"
    ckpt = f"learning/checkpoints/{task}"
    res = f"learning/results/{task}"
    _run([sys.executable, "-m", "learning.train",
          "--data", data, "--epochs", str(epochs),
          "--ensemble", str(ensemble),
          "--out", ckpt, "--results", res])


def evaluate(task, episodes):
    ckpt = f"learning/checkpoints/{task}"
    res = f"learning/results/{task}"
    _run([sys.executable, "-m", "learning.eval_task",
          "--task", task, "--checkpoint", ckpt,
          "--episodes", str(episodes), "--results", res,
          "--no-replay-fails"])


def report():
    _run([sys.executable, "-m", "learning.make_lewm_report"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all",
                        choices=("collect", "train", "eval", "report", "all"))
    parser.add_argument("--task", type=str, default="all",
                        help="all | two_room | push_t | reacher")
    parser.add_argument("--num-rollouts", type=int, default=500)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=25)
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    for t in tasks:
        if args.stage in ("collect", "all"):
            collect(t, args.num_rollouts, args.episode_len)
        if args.stage in ("train", "all"):
            train(t, args.epochs, args.ensemble)
        if args.stage in ("eval", "all"):
            evaluate(t, args.eval_episodes)
    if args.stage in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
