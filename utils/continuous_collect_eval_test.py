# coding=utf-8
# Copyright 2019 The Tensor2Robot Authors.
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

"""Tests for run_collect_eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import gin
from tensor2robot.research.pose_env import pose_env
from tensor2robot.utils import continuous_collect_eval
import tensorflow as tf
FLAGS = flags.FLAGS


class PoseEnvModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      (pose_env.PoseEnvRandomPolicy,),
  )
  def test_run_pose_env_collect(self, demo_policy_cls):
    urdf_root = pose_env.get_pybullet_urdf_root()

    config_dir = 'research/pose_env/configs'
    gin_config = os.path.join(
        FLAGS.test_srcdir, config_dir, 'run_random_collect.gin')
    gin.parse_config_file(gin_config)
    tmp_dir = absltest.get_default_test_tmpdir()
    root_dir = os.path.join(tmp_dir, str(demo_policy_cls))
    gin.bind_parameter('PoseToyEnv.urdf_root', urdf_root)
    gin.bind_parameter(
        'collect_eval_loop.root_dir', root_dir)
    gin.bind_parameter('run_meta_env.num_tasks', 2)
    gin.bind_parameter('run_meta_env.num_episodes_per_adaptation', 1)
    gin.bind_parameter(
        'collect_eval_loop.policy_class', demo_policy_cls)
    continuous_collect_eval.collect_eval_loop()
    output_files = tf.io.gfile.glob(os.path.join(
        root_dir, 'policy_collect', '*.tfrecord'))
    self.assertLen(output_files, 2)


if __name__ == '__main__':
  absltest.main()
