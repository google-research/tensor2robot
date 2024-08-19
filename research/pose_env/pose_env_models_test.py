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

"""Integration tests for training pose_env models."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import gin
from tensor2robot.input_generators import default_input_generator
from tensor2robot.meta_learning import meta_policies
from tensor2robot.meta_learning import preprocessors
from tensor2robot.predictors import checkpoint_predictor
from tensor2robot.research.pose_env import pose_env
from tensor2robot.research.pose_env import pose_env_maml_models
from tensor2robot.research.pose_env import pose_env_models
from tensor2robot.utils import train_eval
from tensor2robot.utils import train_eval_test_utils
import tensorflow.compat.v1 as tf  # tf


BATCH_SIZE = 1
MAX_TRAIN_STEPS = 1
EVAL_STEPS = 1

NUM_TRAIN_SAMPLES_PER_TASK = 1
NUM_VAL_SAMPLES_PER_TASK = 1

FLAGS = tf.app.flags.FLAGS


class PoseEnvModelsTest(parameterized.TestCase):

  def setUp(self):
    super(PoseEnvModelsTest, self).setUp()
    base_dir = 'tensor2robot'
    test_data = os.path.join(FLAGS.test_srcdir,
                             base_dir,
                             'test_data/pose_env_test_data.tfrecord')
    self._train_log_dir = FLAGS.test_tmpdir
    if tf.io.gfile.exists(self._train_log_dir):
      tf.io.gfile.rmtree(self._train_log_dir)
    gin.bind_parameter('train_eval_model.max_train_steps', 3)
    gin.bind_parameter('train_eval_model.eval_steps', 2)

    self._record_input_generator = (
        default_input_generator.DefaultRecordInputGenerator(
            batch_size=BATCH_SIZE, file_patterns=test_data))

    self._meta_record_input_generator_train = (
        default_input_generator.DefaultRandomInputGenerator(
            batch_size=BATCH_SIZE))
    self._meta_record_input_generator_eval = (
        default_input_generator.DefaultRandomInputGenerator(
            batch_size=BATCH_SIZE))

  def test_mc(self):
    train_eval.train_eval_model(
        t2r_model=pose_env_models.PoseEnvContinuousMCModel(),
        input_generator_train=self._record_input_generator,
        input_generator_eval=self._record_input_generator,
        create_exporters_fn=None)

  def test_regression(self):
    train_eval.train_eval_model(
        t2r_model=pose_env_models.PoseEnvRegressionModel(),
        input_generator_train=self._record_input_generator,
        input_generator_eval=self._record_input_generator,
        create_exporters_fn=None)

  def test_regression_maml(self):
    maml_model = pose_env_maml_models.PoseEnvRegressionModelMAML(
        base_model=pose_env_models.PoseEnvRegressionModel())
    train_eval.train_eval_model(
        t2r_model=maml_model,
        input_generator_train=self._meta_record_input_generator_train,
        input_generator_eval=self._meta_record_input_generator_eval,
        create_exporters_fn=None)

  def _test_policy_interface(self, policy, restore=True):
    urdf_root = pose_env.get_pybullet_urdf_root()
    self.assertTrue(os.path.exists(urdf_root))
    env = pose_env.PoseToyEnv(
        urdf_root=urdf_root, render_mode='DIRECT')
    env.reset_task()
    obs = env.reset()
    if restore:
      policy.restore()
    policy.reset_task()
    action = policy.SelectAction(obs, None, 0)

    new_obs, rew, done, env_debug = env.step(action)
    episode_data = [[(obs, action, rew, new_obs, done, env_debug)]]
    policy.adapt(episode_data)

    policy.SelectAction(new_obs, None, 1)

  def test_regression_maml_policy_interface(self):
    t2r_model = pose_env_maml_models.PoseEnvRegressionModelMAML(
        base_model=pose_env_models.PoseEnvRegressionModel(),
        preprocessor_cls=preprocessors.FixedLenMetaExamplePreprocessor)
    predictor = checkpoint_predictor.CheckpointPredictor(t2r_model=t2r_model)
    predictor.init_randomly()
    policy = meta_policies.MAMLRegressionPolicy(t2r_model, predictor=predictor)
    self._test_policy_interface(policy, restore=False)

  @parameterized.parameters(
      ('run_train_reg_maml.gin',),
      ('run_train_reg.gin',))
  def test_train_eval_gin(self, gin_file):
    base_dir = 'tensor2robot'
    full_gin_path = os.path.join(
        FLAGS.test_srcdir, base_dir, 'research/pose_env/configs', gin_file)
    model_dir = os.path.join(FLAGS.test_tmpdir, 'test_train_eval_gin', gin_file)
    train_eval_test_utils.test_train_eval_gin(
        test_case=self,
        model_dir=model_dir,
        full_gin_path=full_gin_path,
        max_train_steps=MAX_TRAIN_STEPS,
        eval_steps=EVAL_STEPS)


if __name__ == '__main__':
  absltest.main()
