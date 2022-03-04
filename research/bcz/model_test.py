# coding=utf-8
# Copyright 2022 The Tensor2Robot Authors.
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

"""Tests BCZ models with placeholder data."""

import itertools
import os
from absl import flags
from absl.testing import parameterized
import gin
from tensor2robot.research.bcz import model
from tensor2robot.utils.t2r_test_fixture import T2RModelFixture
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
TRAIN = tf.estimator.ModeKeys.TRAIN
_POSE_COMPONENTS_LIST = list(itertools.product(*[
    [True, False], [True, False], ['axis_angle', 'quaternion'], [False]]))


class BCZModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    gin.clear_config()
    super(BCZModelTest, self).setUp()
    self._fixture = T2RModelFixture(test_case=self, use_tpu=False)

  @parameterized.parameters(
      (model.spatial_softmax_network),
      (model.resnet_film_network),
  )
  def test_network_fn(self, network_fn):
    model_name = 'BCZModel'
    gin.bind_parameter(
        'BCZModel.network_fn', network_fn)
    gin.parse_config('BCZPreprocessor.mock_subtask = True')
    gin.parse_config(
        'resnet_film_network.film_generator_fn = @linear_film_generator')
    self._fixture.random_train(model, model_name)

  def test_all_components(self):
    """Train with all pose components."""
    model_name = 'BCZModel'
    pose_components = [
        ('xyz', 3, True, 100.),
        ('quaternion', 4, False, 10.),
        ('axis_angle', 3, True, 10.),
        ('arm_joints', 7, True, 1.),
        ('target_close', 1, False, 1.),
    ]
    gin.bind_parameter(
        'BCZModel.action_components', pose_components)
    gin.parse_config('BCZPreprocessor.mock_subtask = True')
    gin.parse_config(
        'resnet_film_network.film_generator_fn = @linear_film_generator')
    self._fixture.random_train(model, model_name)

  @parameterized.parameters(*_POSE_COMPONENTS_LIST)
  def test_pose_components(self,
                           residual_xyz,
                           residual_angle,
                           angle_format,
                           residual_gripper):
    """Tests with different action configurations."""
    model_name = 'BCZModel'
    if angle_format == 'axis_angle':
      angle_size = 3
    elif angle_format == 'quaternion':
      angle_size = 4
    action_components = [
        ('xyz', 3, residual_xyz, 100.),
        (angle_format, angle_size, residual_angle, 10.),
        ('target_close', 1, residual_gripper, 1.),
    ]
    gin.bind_parameter(
        'BCZModel.action_components', action_components)
    gin.bind_parameter(
        'BCZModel.state_components', [])
    gin.parse_config('BCZPreprocessor.mock_subtask = True')
    gin.parse_config(
        'resnet_film_network.film_generator_fn = @linear_film_generator')
    self._fixture.random_train(model, model_name)

  def test_random_train(self):
    base_dir = 'tensor2robot'

    gin_config = os.path.join(
        FLAGS.test_srcdir, base_dir, 'research/bcz/configs',
        'run_train_bc_langcond_trajectory.gin')
    model_name = 'BCZModel'
    gin_bindings = [
        'train_eval_model.eval_steps = 1',
        'EVAL_INPUT_GENERATOR=None',
    ]
    gin.parse_config_files_and_bindings(
        [gin_config], gin_bindings, finalize_config=False)
    self._fixture.random_train(model, model_name)

  def tearDown(self):
    gin.clear_config()
    super(BCZModelTest, self).tearDown()


if __name__ == '__main__':
  tf.test.main()
