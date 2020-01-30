# coding=utf-8
# Copyright 2020 The Tensor2Robot Authors.
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

"""Test fixture for T2R models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import mock
from tensor2robot.input_generators import default_input_generator
from tensor2robot.utils import train_eval
from tensor2robot.utils import train_eval_test_utils
import tensorflow as tf

from tensorflow.python.tpu import tpu  # pylint: disable=g-direct-tensorflow-import

TRAIN = tf.estimator.ModeKeys.TRAIN
_MAX_TRAIN_STEPS = 2
_BATCH_SIZE = 2
_USE_TPU_WRAPPER = True


class T2RModelFixture(object):
  """Fixture to quickly test estimator models."""

  def __init__(self, test_case, use_tpu=False, extra_bindings=None):
    self._test_case = test_case
    self._use_tpu = use_tpu
    if self._use_tpu:
      gin.bind_parameter('AbstractT2RModel.device_type', 'tpu')
      gin.bind_parameter('tf.contrib.tpu.TPUConfig.iterations_per_loop', 1)
    gin.bind_parameter('tf.estimator.RunConfig.save_checkpoints_steps', 1)

    if extra_bindings:
      for parameter, binding in extra_bindings.items():
        gin.bind_parameter(parameter, binding)

  def random_train(self, module_name, model_name, **module_kwargs):
    """Instantiates and trains a T2R model with random inputs."""
    tf_model = getattr(module_name, model_name)(**module_kwargs)
    self.random_train_model(tf_model, **module_kwargs)

  def random_train_model(self, tf_model, **module_kwargs):
    """Trains a T2R model with random inputs."""
    params = self._get_params(
        model_dir=self._test_case.create_tempdir().full_path, **module_kwargs)
    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=params['batch_size'])

    initialize_system = tpu.initialize_system
    with mock.patch.object(
        tpu, 'initialize_system', autospec=True) as mock_init:
      mock_init.side_effect = initialize_system
      train_eval.train_eval_model(
          t2r_model=tf_model,
          input_generator_train=input_generator,
          max_train_steps=params['max_train_steps'],
          model_dir=params['model_dir'],
          use_tpu_wrapper=params['use_tpu_wrapper'])
      if self._use_tpu:
        mock_init.assert_called()
      train_eval_test_utils.assert_output_files(
          test_case=self._test_case,
          model_dir=params['model_dir'],
          expected_output_filename_patterns=train_eval_test_utils
          .DEFAULT_TRAIN_FILENAME_PATTERNS)

  def recordio_train(self, module_name, model_name, file_patterns,
                     **module_kwargs):
    """Trains the model with a RecordIO dataset for a few steps."""
    tf_model = getattr(module_name, model_name)(**module_kwargs)
    params = self._get_params(
        model_dir=self._test_case.create_tempdir().full_path, **module_kwargs)
    input_generator = default_input_generator.DefaultRecordInputGenerator(
        file_patterns, batch_size=params['batch_size'])
    initialize_system = tpu.initialize_system
    with mock.patch.object(
        tpu, 'initialize_system', autospec=True) as mock_init:
      mock_init.side_effect = initialize_system
      train_eval.train_eval_model(
          t2r_model=tf_model,
          input_generator_train=input_generator,
          max_train_steps=params['max_train_steps'],
          model_dir=params['model_dir'],
          use_tpu_wrapper=params['use_tpu_wrapper'])
      if self._use_tpu:
        mock_init.assert_called()
      train_eval_test_utils.assert_output_files(
          test_case=self._test_case,
          model_dir=params['model_dir'],
          expected_output_filename_patterns=train_eval_test_utils
          .DEFAULT_TRAIN_FILENAME_PATTERNS)

  def random_predict(self, module_name, model_name, **module_kwargs):
    """Runs predictions through a model with random inputs."""
    tf_model = getattr(module_name, model_name)(**module_kwargs)

    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=1)
    for prediction in train_eval.predict_from_model(
        t2r_model=tf_model,
        input_generator_predict=input_generator,
        model_dir=self._test_case.create_tempdir().full_path):
      return prediction
    return None

  def _get_params(self, model_dir, **params):
    default_params = dict(
        batch_size=_BATCH_SIZE,
        max_train_steps=_MAX_TRAIN_STEPS,
        use_tpu_wrapper=_USE_TPU_WRAPPER,
        model_dir=model_dir)
    default_params.update(params)
    return default_params
