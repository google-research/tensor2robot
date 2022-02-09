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

# Lint as python3
"""Utility functions for vr gripper.

Includes utils needed for discrete actions.
"""

from typing import List, Optional

import gin
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim


def GetDiscreteBins(num_bins, output_min, output_max):
  """Compute bin centers for discretizing the provided range into bins.

  Args:
    num_bins: number of bins to discretize into
    output_min: numpy array of size [action_dim,], min action values for each
      dimension
    output_max: numpy array of size [action_dim,], max action values for each
      dimension
  Returns:
    A numpy array of size [num_bins, action_dim] corresponding to the bin
      centers for each dimension.
  """
  action_range = output_max - output_min
  bin_sizes = action_range / float(num_bins)
  return np.array([output_min + (bin_sizes) * (bin_i + 0.5)
                   for bin_i in range(num_bins)])


def GetDiscreteActions(logits, action_size, num_bins, bin_centers):
  """Compute the discrete actions corresponding to the input logits.

  Args:
    logits: Tensor of size [batch_size, 1, timesteps, action_dim*num_bins]
      corresponding to the logits for all actions.
    action_size: dimensionality of the action space.
    num_bins: number of discrete bins.
    bin_centers: numpy array of size [num_bins, action_dim] with the centers of
      each bin for each dimension.

  Returns:
    Action corresponding to the maximum, tensor of size
      [batch_size, 1, timesteps, action_dim]
  """
  # Could compute the max right away without using a softmax, but this code
  # will be useful later when sampling from the distribution instead.
  action_probabilities = tf.nn.softmax(
      tf.reshape(logits, (-1, action_size, num_bins)))
  actions_onehot = tf.one_hot(tf.argmax(action_probabilities, -1), num_bins)
  bin_centers = tf.constant(np.transpose(bin_centers),
                            dtype=tf.float32)
  # Can change actions_onehot to action_probabilies below to get mean rather
  # than mode.
  actions = tf.reduce_sum(actions_onehot*bin_centers, -1)
  actions = tf.reshape(actions,
                       (-1, logits.shape[1], logits.shape[2], action_size))
  return actions


def GetDiscreteActionLoss(logits, action_labels, bin_centers, num_bins):
  """Convert labels to one-hot, compute cross-entropy loss, and return loss.

  Args:
    logits: Tensor corresponding to the predicted action logits, with size
      [batch_size, timesteps, action_dim*num_bins]
    action_labels: Tensor corresponding to the real valued action labels, with
      size [batch_size, 1, timesteps, action_dim]
    bin_centers: numpy array of size [num_bins, action_dim] corresponding to
      the centers of each bin for each dimension.
    num_bins: number of discrete bins.
  Returns:
    Scalar tensor, cross entropy loss between the predicted actions and labels.
  """
  action_labels = tf.expand_dims(action_labels, -2)
  bin_centers = tf.constant(bin_centers, dtype=tf.float32)
  while len(bin_centers.shape) < len(action_labels.shape):
    bin_centers = tf.expand_dims(bin_centers, 0)
  discrete_labels = tf.argmin((action_labels - bin_centers)**2, -2)
  onehot_labels = tf.one_hot(discrete_labels, num_bins)
  onehot_labels = tf.reshape(onehot_labels, (-1, num_bins))
  logits = tf.reshape(logits, (-1, num_bins))
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(onehot_labels, logits)
  loss = tf.reduce_mean(loss)
  return loss


@gin.configurable
class DiscreteDecoder(object):
  """Decoder module for predicting discrete actions."""

  def __init__(self,
               num_bins = 1,
               output_min = None,
               output_max = None):
    """Initialize the decoder.

    Args:
      num_bins: If action_discrete, the number of bins to discretize into.
      output_min: If action_discrete, min value of the action to discretize at.
      output_max: If action_discrete, max value of the action to discretize at.
    Returns:
      Tensor corresponding to discrete action.
    """
    self._num_bins = num_bins
    self._bin_centers = GetDiscreteBins(
        num_bins, np.array(output_min), np.array(output_max))

  def __call__(self, params, output_size):
    self._action_logits = slim.fully_connected(
        params, output_size * self._num_bins, activation_fn=None,
        scope='action_logits')
    predictions = GetDiscreteActions(
        self._action_logits, output_size, self._num_bins, self._bin_centers)
    return predictions

  def loss(self, labels):  # pylint: disable=invalid-name
    return GetDiscreteActionLoss(
        self._action_logits, labels.action, self._bin_centers, self._num_bins)
