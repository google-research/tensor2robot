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

# Lint as: python2, python3
"""Utility functions for train_eval tests for new models."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
from typing import Callable, Optional, Text, List
import gin
from tensor2robot.utils import train_eval
import tensorflow.compat.v1 as tf

DEFAULT_TRAIN_FILENAME_PATTERNS = [
    'operative_config-0.gin', 'model.ckpt-0.data-*', 'model.ckpt-0.meta',
    'model.ckpt-0.index', 'checkpoint'
]
DEFAULT_EVAL_FILENAME_PATTERNS = ['eval/events.*']


def assert_output_files(
    test_case,
    model_dir,
    expected_output_filename_patterns = None):
  """Verify that the expected output files are generated.

  Args:
    test_case: The instance of the test used to assert that the output files are
      generated.
    model_dir: The path where the model should be stored.
    expected_output_filename_patterns: All patterns of files which should exist
      after train_and_eval, train, or eval. If None, the default expected
      filename patterns are used.
  """
  if expected_output_filename_patterns is None:
    expected_output_filename_patterns = (
        DEFAULT_TRAIN_FILENAME_PATTERNS + DEFAULT_EVAL_FILENAME_PATTERNS)

  # Check that expected files have been written.
  for pattern in expected_output_filename_patterns:
    filename_pattern = os.path.join(model_dir, pattern)
    print('file_pattern', filename_pattern)
    filenames = tf.io.gfile.glob(filename_pattern)
    print('filenames', filenames)
    test_case.assertNotEmpty(
        filenames, msg='No files found with pattern "%s"' % filename_pattern)
    for filename in filenames:
      with tf.io.gfile.GFile(filename) as f:
        test_case.assertGreater(f.size(), 0, msg='%s is empty' % filename)


def test_train_eval_gin(test_case,
                        model_dir,
                        full_gin_path,
                        max_train_steps,
                        eval_steps,
                        gin_overwrites_fn = None,
                        assert_train_output_files = True,
                        assert_eval_output_files = True):
  """Train and eval a runnable gin config.

  Until we have a proper gen_rule to create individual targets for every gin
  file automatically, gin files can be tested using the pattern below.
  Please, use 'test_train_eval_gin' as the test function name such that it
  is easy to convert these tests as soon as the gen_rule is available.

  @parameterized.parameters(
      ('first.gin',),
      ('second.gin',),
      ('third.gin',),
  )
  def test_train_eval_gin(self, gin_file):
    full_gin_path = os.path.join(FLAGS.test_srcdir, BASE_GIN_PATH, gin_file)
    model_dir = os.path.join(FLAGS.test_tmpdir, 'test_train_eval_gin', gin_file)
    train_eval_test_utils.test_train_eval_gin(
        test_case=self,
        model_dir=model_dir,
        full_gin_path=full_gin_path,
        max_train_steps=MAX_TRAIN_STEPS,
        eval_steps=EVAL_STEPS)

  Args:
    test_case: The instance of the test used to assert that the output files are
      generated.
    model_dir: The path where the model should be stored.
    full_gin_path: The path of the gin file which parameterizes train_eval.
    max_train_steps: The maximum number of training steps, should be small since
      this is just for testing.
    eval_steps: The number of eval steps, should be small since this is just for
      testing.
    gin_overwrites_fn: Optional function which binds gin parameters to
      overwrite.
    assert_train_output_files: If True, the expected output files of the
      training run are checked, otherwise this check is skipped. If only
      evaluation is performed this should be set to False.
    assert_eval_output_files: If True, the output expected files of the
      evaluation run are checked, otherwise this check is skipped. If only
      training is performed this should be set to False. Note, if
      assert_train_output_files is set to False the model_dir is not deleted
      in order to load the model from training.
  """
  # We clear all prior parameters set by gin to ensure that we can call this
  # function sequentially for all parameterized tests.
  gin.clear_config(clear_constants=True)

  gin.parse_config_file(full_gin_path)
  gin.bind_parameter('train_eval_model.model_dir', model_dir)

  if gin_overwrites_fn is not None:
    gin_overwrites_fn()

  # Make sure that the model dir is empty. This is important for running
  # tests locally.
  if tf.io.gfile.exists(model_dir) and assert_train_output_files:
    tf.io.gfile.rmtree(model_dir)

  train_eval.train_eval_model(
      model_dir=model_dir,
      max_train_steps=max_train_steps,
      eval_steps=eval_steps,
      create_exporters_fn=None)
  if assert_train_output_files:
    assert_output_files(
        test_case=test_case,
        model_dir=model_dir,
        expected_output_filename_patterns=DEFAULT_TRAIN_FILENAME_PATTERNS)
  if assert_eval_output_files:
    assert_output_files(
        test_case=test_case,
        model_dir=model_dir,
        expected_output_filename_patterns=DEFAULT_EVAL_FILENAME_PATTERNS)
