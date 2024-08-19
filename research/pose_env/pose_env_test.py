# coding=utf-8
# Copyright 2024 The Tensor2Robot Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tensor2robot.research.pose_env.pose_env."""

import os
from absl.testing import absltest
from six.moves import range
from tensor2robot.research.pose_env import pose_env


class PoseEnvTest(absltest.TestCase):

  def test_PoseEnv(self):
    urdf_root = pose_env.get_pybullet_urdf_root()
    self.assertTrue(os.path.exists(urdf_root))
    env = pose_env.PoseToyEnv(urdf_root=urdf_root)
    obs = env.reset()
    policy = pose_env.PoseEnvRandomPolicy()
    action, _ = policy.sample_action(obs, 0)
    for _ in range(3):
      obs, _, done, _ = env.step(action)
      if done:
        obs = env.reset()

if __name__ == '__main__':
  absltest.main()
