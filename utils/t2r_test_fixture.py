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

"""Test fixture for T2R models."""


import glob
import os
import shutil
from typing import List
from absl import logging
import gin
import mock
import numpy as np

from tensor2robot.input_generators import default_input_generator
from tensor2robot.utils import train_eval
from tensor2robot.utils import train_eval_test_utils
import tensorflow.compat.v1 as tf

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
          input_generator_eval=input_generator,
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
    return params['model_dir']

  def random_predict(self,
                     module_name,
                     model_name,
                     batch_size=1,
                     yield_single_examples=True,
                     **module_kwargs):
    """Runs predictions through a model with random inputs."""
    tf_model = getattr(module_name, model_name)(**module_kwargs)

    input_generator = default_input_generator.DefaultRandomInputGenerator(
        batch_size=batch_size)
    for prediction in train_eval.predict_from_model(
        t2r_model=tf_model,
        input_generator_predict=input_generator,
        model_dir=self._test_case.create_tempdir().full_path,
        yield_single_examples=yield_single_examples):
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

  def train_and_check_golden_predictions(self,
                                         module_name,
                                         model_name,
                                         file_patterns,
                                         golden_data_filename,
                                         generate_golden_data,
                                         python2_data):
    """Verify model predictions / train outputs have remained constant.

    Args:
      module_name: Module containing T2R model.
      model_name: Name of T2R model.
      file_patterns: List of data to train model on. Should contain 1 data
        point for maximall determinism.
      golden_data_filename: Name of cached golden data.
      generate_golden_data: Re-generate golden golden data and checkpoint.
      python2_data: Set True if golden data was generated in Python2.
    """
    # To get deterministic random preprocessing. Note that model init does
    # not seem to respect this so initialize from the same checkpoint when
    # using recordio_train.
    tf.set_random_seed(123)
    model_dir = self.recordio_train(
        module_name, model_name, file_patterns)
    model_data_arr = np.load(
        os.path.join(model_dir, 'golden_values.npy'), allow_pickle=True)
    if generate_golden_data:
      with open(golden_data_filename, 'wb') as golden_data_file:
        np.save(golden_data_file, model_data_arr)
      # Update the golden checkpoint in the same directory as the golden
      # predictions.
      golden_dir = os.path.dirname(golden_data_filename)
      for filename in glob.glob(os.path.join(model_dir, 'model.ckpt-0*')):
        shutil.copy2(filename, os.path.join(
            golden_dir, os.path.basename(filename)))
    else:
      # load from python2-generated data.
      if python2_data:
        encoding = 'latin1'
      else:
        encoding = 'ASCII'
      golden_data_arr = np.load(
          golden_data_filename, allow_pickle=True, encoding=encoding)
      # For every value in the golden data, make sure that the model has not
      # changed it.
      golden_data = golden_data_arr[0]
      model_data = model_data_arr[0]
      for key, golden_value in golden_data.items():
        if key not in model_data:
          logging.info('key=%s, status=not in model dict', key)
          continue
        logging.info('key=%s', key)
        np.testing.assert_almost_equal(
            model_data[key], golden_value, decimal=5)
