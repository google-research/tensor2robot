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

# Lint as: python3
"""Tests for tensor2robot.utils.global_step_functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2robot.utils import global_step_functions
import tensorflow.compat.v1 as tf


class GlobalStepFunctionsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters({
      'testcase_name': 'constant',
      'boundaries': [1],
      'values': [5.0],
      'test_inputs': [0, 1, 10],
      'expected_outputs': [5.0, 5.0, 5.0]
  }, {
      'testcase_name': 'ramp_up',
      'boundaries': [10, 20],
      'values': [1.0, 11.0],
      'test_inputs': [0, 10, 13, 15, 18, 20, 25],
      'expected_outputs': [1.0, 1.0, 4.0, 6.0, 9.0, 11.0, 11.0]
  })
  def test_piecewise_linear(self, boundaries, values, test_inputs,
                            expected_outputs):
    global_step = tf.train.get_or_create_global_step()
    global_step_value = tf.placeholder(tf.int64, [])
    set_global_step = tf.assign(global_step, global_step_value)

    test_function = global_step_functions.piecewise_linear(boundaries, values)
    with tf.Session() as sess:
      for x, y_expected in zip(test_inputs, expected_outputs):
        sess.run(set_global_step, {global_step_value: x})
        y = sess.run(test_function)
        self.assertEqual(y, y_expected)

    # Test the same with tensors as inputs
    test_function = global_step_functions.piecewise_linear(
        tf.convert_to_tensor(boundaries), tf.convert_to_tensor(values))
    with tf.Session() as sess:
      for x, y_expected in zip(test_inputs, expected_outputs):
        sess.run(set_global_step, {global_step_value: x})
        y = sess.run(test_function)
        self.assertEqual(y, y_expected)


if __name__ == '__main__':
  tf.test.main()
