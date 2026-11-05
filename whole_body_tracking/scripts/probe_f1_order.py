#!/usr/bin/env python3
"""Diagnostic: print the PhysX joint/body order IsaacLab builds for F1.

Run on gradmotion (IsaacLab image):
    gm-run Beyondminic-Weilai-X1/whole_body_tracking/scripts/probe_f1_order.py
Prints robot.joint_names / body_names — the authoritative order the policy
was trained in (obs normalizer is keyed to it). ~2 min, 1 env, no training.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "source", "whole_body_tracking"))

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, num_envs=1).app  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402  (register tasks)
import whole_body_tracking.tasks  # noqa: F401,E402

env = gym.make("Tracking-Flat-F1-v0")
robot = env.unwrapped.scene["robot"]
print("[PROBE] joint_names:", list(robot.data.joint_names))
print("[PROBE] body_names:", list(robot.data.body_names))
print("[PROBE] n_joints:", len(robot.data.joint_names), "n_bodies:", len(robot.data.body_names))
env.close()
app.close()
print("[PROBE] done")
