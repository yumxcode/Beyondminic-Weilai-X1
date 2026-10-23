#!/usr/bin/env python3
"""Render reference motion npz kinematically in MuJoCo (offscreen MP4).

Mirrors the camera + loop protocol of play_mujoco.py --record so the output
can be hstack-compared frame-by-frame with the policy playback video.

Usage:
    .venv/bin/python tools_render_ref_x1.py \
        --motion whole_body_tracking/motions/dance_x1_train.npz \
        --loops 5 --out dance_x1_reference.mp4
"""
import argparse
import os

import numpy as np
import mujoco
import imageio.v2 as imageio

REPO = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(
    REPO, "whole_body_tracking", "source", "whole_body_tracking",
    "whole_body_tracking", "assets", "xyber_x1", "mjcf", "xyber_x1_flat.xml",
)
POLICY_DT = 0.02  # 50 Hz, matches npz fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--loops", type=int, default=5)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mjcf", default=MJCF)
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    # MuJoCo serial tree order joint qpos/dof addresses (skip free joint)
    joint_qpos_adr = [model.jnt_qposadr[i] for i in range(1, model.njnt)]

    md = np.load(args.motion)
    ref_joint_pos = md["joint_pos"].astype(np.float64)
    ref_body_pos_w = md["body_pos_w"].astype(np.float64)
    ref_body_quat_w = md["body_quat_w"].astype(np.float64)
    n_steps = ref_joint_pos.shape[0]
    print(f"[motion] {n_steps} frames @ {float(md['fps'])} fps, loops={args.loops}")

    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(model.jnt_bodyid[0])  # base_link
    cam.distance = 4.0
    cam.elevation = -10.0
    cam.azimuth = 90.0

    writer = imageio.get_writer(args.out, fps=int(round(1.0 / POLICY_DT)),
                                quality=9, macro_block_size=1)
    total = n_steps * args.loops
    for i in range(total):
        t = i % n_steps
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = ref_body_pos_w[t, 0]
        data.qpos[3:7] = ref_body_quat_w[t, 0]
        for j, adr in enumerate(joint_qpos_adr):
            data.qpos[adr] = ref_joint_pos[t, j]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
        if (i + 1) % n_steps == 0:
            print(f"[loop] {(i + 1) // n_steps}/{args.loops} done ({i + 1}/{total})")
    writer.close()
    print(f"[saved] {args.out} ({total} frames)")


if __name__ == "__main__":
    main()
