from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from beyondmimic_g1.math_utils import (
    initial_motion_to_world_yaw,
    quat_to_matrix,
    relative_orientation_6d,
)
from beyondmimic_g1.motion import MotionData
from beyondmimic_g1.policy import PolicySpec
from beyondmimic_g1.runtime import ObservationBuilder, permutation


JOINTS = tuple(f"joint_{index}" for index in range(29))


class RuntimeTest(unittest.TestCase):
    def test_permutation(self):
        source = ["b", "a", "c"]
        np.testing.assert_array_equal(permutation(source, ["a", "b", "c"]), [1, 0, 2])

    def test_identity_relative_orientation(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        alignment = initial_motion_to_world_yaw(identity, identity)
        result = relative_orientation_6d(identity, identity, alignment)
        np.testing.assert_allclose(result, np.eye(3)[:, :2].reshape(6), atol=1e-6)
        np.testing.assert_allclose(quat_to_matrix(alignment), np.eye(3), atol=1e-6)

    def test_motion_and_observation_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            frames, bodies = 3, 10
            quats = np.zeros((frames, bodies, 4), dtype=np.float32)
            quats[..., 0] = 1.0
            np.savez(
                path,
                fps=np.array([50]),
                joint_pos=np.zeros((frames, 29), dtype=np.float32),
                joint_vel=np.zeros((frames, 29), dtype=np.float32),
                body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float32),
                body_quat_w=quats,
                body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
                body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
            )
            motion = MotionData(path)
            spec = PolicySpec(
                joint_names=JOINTS,
                default_joint_pos=np.zeros(29, dtype=np.float32),
                stiffness=np.ones(29, dtype=np.float32),
                damping=np.ones(29, dtype=np.float32),
                action_scale=np.ones(29, dtype=np.float32),
                observation_names=(),
                anchor_body_name="torso_link",
                num_observations=154,
                num_actions=29,
            )
            builder = ObservationBuilder(spec, JOINTS, 9)
            observation = builder.build(
                motion.frame(0),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(29),
                np.zeros(29),
            )
            self.assertEqual(observation.shape, (154,))
            self.assertEqual(builder.target_runtime(np.full(29, 2.0), 1.0).shape, (29,))

    def test_joint_order_mapping(self):
        runtime_names = tuple(reversed(JOINTS))
        spec = PolicySpec(
            joint_names=JOINTS,
            default_joint_pos=np.arange(29, dtype=np.float32),
            stiffness=np.ones(29, dtype=np.float32),
            damping=np.ones(29, dtype=np.float32),
            action_scale=np.ones(29, dtype=np.float32),
            observation_names=(),
            anchor_body_name="torso_link",
            num_observations=154,
            num_actions=29,
        )
        builder = ObservationBuilder(spec, runtime_names, 0)
        target = builder.target_runtime(np.zeros(29), 1.0)
        np.testing.assert_array_equal(target, np.arange(29, dtype=np.float32)[::-1])


if __name__ == "__main__":
    unittest.main()
