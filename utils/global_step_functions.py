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

# Lint as: python3
"""Gin configurable functions returning tf.Tensors based on the global_step.
"""

from typing import Optional, Sequence, Text, Union

import gin
import numpy as np
import tensorflow.compat.v1 as tf


@gin.configurable
def piecewise_linear(boundaries,
                     values,
                     name = None):
  """Piecewise linear function assuming given values at given boundaries.

  Args:
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries. The first entry must be 0.
    values: A list of `Tensor`s or float`s or `int`s that specifies the values
      at the `boundaries`. It must have the same number of elements as
      `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` if `x < boundaries[0]` and
    `values[-1]` if `x >= boundaries[-1]. If `boundaries[i] <= x <
    boundaries[i+1]` it is the linear interpolation between `values[i]` and
    `values[i+1]`: `values[i] + (values[i+1]-values[i]) * (x-boundaries[i]) /
    (boundaries[i+1]-boundaries[i])`.

  Raises:
    AssertionError: if values or boundaries is empty, or not the same size.
  """
  global_step = tf.train.get_or_create_global_step()
  with tf.name_scope(name, 'PiecewiseLinear', [global_step, boundaries, values,
                                               name]) as name:
    values = tf.convert_to_tensor(values)
    x = tf.cast(tf.convert_to_tensor(global_step), values.dtype)
    boundaries = tf.cast(tf.convert_to_tensor(boundaries), values.dtype)

    num_boundaries = np.prod(boundaries.shape.as_list())
    num_values = np.prod(values.shape.as_list())
    assert num_boundaries > 0, 'Need more than 0 boundaries'
    assert num_values > 0, 'Need more than 0 values'
    assert num_values == num_boundaries, ('boundaries and values must be of '
                                          'same size')

    # Make sure there is an unmet last boundary with the same value as the
    # last one that was passed in, and at least one boundary was met.
    values = tf.concat([values, tf.reshape(values[-1], [1])], 0)
    boundaries = tf.concat(
        [boundaries,
         tf.reshape(tf.maximum(x + 1, boundaries[-1]), [1])], 0)

    # Make sure there is at least one boundary that was already met, with the
    # same value as the first one that was passed in.
    values = tf.concat([tf.reshape(values[0], [1]), values], 0)
    boundaries = tf.concat(
        [tf.reshape(tf.minimum(x - 1, boundaries[0]), [1]), boundaries], 0)

    # Identify index of the last boundary that was passed.
    unreached_boundaries = tf.reshape(
        tf.where(tf.greater(boundaries, x)), [-1])
    unreached_boundaries = tf.concat(
        [unreached_boundaries, [tf.cast(tf.size(boundaries), tf.int64)]], 0)
    index = tf.reshape(tf.reduce_min(unreached_boundaries), [1])

    # Get values at last and next boundaries.
    value_left = tf.reshape(tf.slice(values, index - 1, [1]), [])
    left_boundary = tf.reshape(tf.slice(boundaries, index - 1, [1]), [])
    value_right = tf.reshape(tf.slice(values, index, [1]), [])
    right_boundary = tf.reshape(tf.slice(boundaries, index, [1]), [])

    # Calculate linear interpolation.
    a = (value_right - value_left) / (right_boundary - left_boundary)
    b = value_left - a * left_boundary
    return a * x + b


@gin.configurable
def exponential_decay(initial_value = 0.0001,
                      decay_steps = 10000,
                      decay_rate = 0.9,
                      staircase = True):
  """Create a value that decays exponentially with global_step.

  Args:
    initial_value: A scalar float32 or float64 Tensor or a Python
      number. The initial value returned for global_step == 0.
    decay_steps: A scalar int32 or int64 Tensor or a Python number. Must be
      positive. See the decay computation in `tf.exponential_decay`.
    decay_rate: A scalar float32 or float64 Tensor or a Python number. The decay
      rate.
    staircase: Boolean. If True, decay the value at discrete intervals.

  Returns:
    value: Scalar tf.Tensor with the value decaying based on the global_step.
  """
  global_step = tf.train.get_or_create_global_step()
  value = tf.compat.v1.train.exponential_decay(
      learning_rate=initial_value,
      global_step=global_step,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=staircase)
  return value
