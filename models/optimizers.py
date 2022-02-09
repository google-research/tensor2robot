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

"""Optimizer factory functions to be used with tensor2robot models."""

from typing import Callable

import gin
import gin.tf
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import opt as contrib_opt


@gin.configurable
def create_constant_learning_rate(initial_learning_rate = 0.0001):
  """Returns the configured constant initial_learning_rate."""
  return initial_learning_rate


@gin.configurable
def create_exp_decaying_learning_rate(initial_learning_rate = 0.0001,
                                      decay_steps = 10000,
                                      decay_rate = 0.9,
                                      staircase = True):
  """Create a learning rate that decays exponentially with global_steps.

  Args:
    initial_learning_rate: A scalar float32 or float64 Tensor or a Python
      number. The initial learning rate.
    decay_steps: A scalar int32 or int64 Tensor or a Python number. Must be
      positive. See the decay computation in `tf.exponential_decay`.
    decay_rate: A scalar float32 or float64 Tensor or a Python number. The decay
      rate.
    staircase: Boolean. If True decay the learning rate at discrete intervals.

  Returns:
    learning_rate: Scaler tf.Tensor with the learning rate depending on the
      globat_step.
  """
  learning_rate = tf.exponential_decay(
      learning_rate=initial_learning_rate,
      global_step=tf.get_or_create_global_step(),
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      staircase=staircase)
  return learning_rate


@gin.configurable
def default_create_optimizer_fn(use_summaries, learning_rate=1e-4):
  if use_summaries:
    tf.summary.scalar('learning_rate', learning_rate)
  return tf.train.AdamOptimizer(learning_rate)


@gin.configurable
def create_adam_optimizer(
    learning_rate_fn = create_constant_learning_rate):
  """Creates a function that returns a configured Adam optimizer."""
  def create_optimizer_fn(use_summaries):
    """Creates an Adam optimizer with an optional learning rate schedule."""
    learning_rate = learning_rate_fn()
    if use_summaries:
      tf.summary.scalar('learning_rate', learning_rate)
    return tf.train.AdamOptimizer(learning_rate=learning_rate)

  return create_optimizer_fn


@gin.configurable
def create_gradient_descent_optimizer(
    learning_rate_fn = create_constant_learning_rate):
  """Creates a function that returns a configured Gradient Descent Optimizer.

  Args:
    learning_rate_fn: Callable that returns a scalar Tensor evaluating to the
      current learning rate.

  Returns:
    A parameterless function that returns the configured Gradient Descent
    Optimizer.
  """

  def create_optimizer_fn(use_summaries):
    """Creates a gradient descent optimizer with an optional lr schedule."""
    learning_rate = learning_rate_fn()
    if use_summaries:
      tf.summary.scalar('learning_rate', learning_rate)
    return tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)

  return create_optimizer_fn


@gin.configurable
def create_momentum_optimizer(
    learning_rate_fn = create_constant_learning_rate,
    momentum=0.9):
  """Creates a function that returns a configured Momentum Optimizer.

  Args:
    learning_rate_fn: Callable that returns a scalar Tensor evaluating to the
      current learning rate.
    momentum: Momentum for Momentum Optimizer.

  Returns:
    A parameterless function that returns the configured Momentum Optimizer.
  """
  def create_optimizer_fn(use_summaries):
    """Creates a momentum optimizer with an optional learning rate schedule."""
    learning_rate = learning_rate_fn()
    if use_summaries:
      tf.summary.scalar('learning_rate', learning_rate)
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)

  return create_optimizer_fn


@gin.configurable
def create_moving_average_optimizer(optimizer,
                                    average_decay = 0.999
                                   ):
  """Creates a function that returns a configured MovingAverageOptimizer.

  Args:
    optimizer: The original tf.Optimizer.
    average_decay: Exponentional decay factor for the variable averaging.

  Returns:
    A parameterless function that returns the configured Momentum Optimizer.
  """
  return contrib_opt.MovingAverageOptimizer(
      optimizer, average_decay=average_decay)


@gin.configurable(denylist=['optimizer'])
def create_swapping_saver(
    optimizer,
    keep_checkpoint_every_n_hours = 1.0,
    save_relative_paths = False,
    max_to_keep = 5,
):
  # TODO(T2R_CONTRIBUTORS): Switch to using gin config for all saver params.
  return optimizer.swapping_saver(
      keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      save_relative_paths=save_relative_paths, max_to_keep=max_to_keep)
