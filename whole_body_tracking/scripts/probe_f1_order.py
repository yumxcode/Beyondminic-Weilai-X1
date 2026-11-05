#!/usr/bin/env python3
"""Diagnostic: print the PhysX joint/body order IsaacLab builds for F1.

Run on gradmotion (IsaacLab image), ~1 min, 1 env, no training:
    gm-run Beyondminic-Weilai-X1/whole_body_tracking/scripts/probe_f1_order.py
Builds the F1 articulation the same way csv_to_npz.py builds X1's (raw
InteractiveScene over the UrdfFileCfg spawn) and prints the authoritative
joint/body order the training policy speaks.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "source", "whole_body_tracking"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from whole_body_tracking.robots.f1 import F1_CYLINDER_CFG  # noqa: E402


class ProbeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = F1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
robot = scene["robot"]
print("[PROBE] joint_names:", list(robot.data.joint_names))
print("[PROBE] body_names:", list(robot.data.body_names))
print("[PROBE] n_joints:", len(robot.data.joint_names), "n_bodies:", len(robot.data.body_names))
print("[PROBE] done")
