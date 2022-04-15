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

"""Reusable network modules.
"""

import functools
from typing import List

import gin
from six.moves import range
import sonnet as snt
from tensor2robot.layers import snail
from tensor2robot.layers import vision_layers
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers


@gin.configurable
def SpatialSoftmaxTorso(image, aux_input, is_training):
  feature_points, end_points = vision_layers.BuildImagesToFeaturesModel(
      image, is_training=is_training, normalizer_fn=layers.layer_norm)
  end_points["feature_points"] = feature_points
  if aux_input is not None:
    feature_points = tf.concat([feature_points, aux_input], axis=1)
  return feature_points, end_points


@gin.configurable
def LinearHead(net, output_size):
  return layers.fully_connected(net, output_size, activation_fn=None)


@gin.configurable
def ConvLSTM(image,
             aux_input,
             conv_torso_fn,
             is_training,
             lstm_num_units,
             output_size,
             condition_sequence_length=20,
             inference_sequence_length=20):
  """Shared Convolutional-Spatial Softmax on images, LSTM body, shared FC head.

  Args:
    image: Batch of [batch_size, sequence_length, H, W, C].
    aux_input: Batch of aux features [batch_size, sequence_length, D].
    conv_torso_fn: Function that performs shared convolution across inputs.
    is_training: If this is a training graph or not.
    lstm_num_units: Number of hidden units in LSTM.
    output_size: Int specifying size of output layer.
    condition_sequence_length: Length of conditioning sequence. Unused.
    inference_sequence_length: Length of inference sequence. Unused.
  Returns:
    output: Tensor of shape [batch_size, sequence_length, output_size].
  """
  del condition_sequence_length, inference_sequence_length
  conv_torso_fn = functools.partial(conv_torso_fn, is_training=is_training)
  feature_points, end_points = snt.BatchApply(conv_torso_fn)(image, aux_input)
  lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=lstm_num_units)
  lstm_outputs, _ = tf.nn.dynamic_rnn(
      lstm_cell, feature_points, dtype=tf.float32)
  head_fn = functools.partial(LinearHead, output_size=output_size)
  estimated_pose = snt.BatchApply(head_fn)(lstm_outputs)
  return estimated_pose, end_points


@gin.configurable
def SNAIL(image,
          aux_input,
          conv_torso_fn,
          is_training,
          output_size,
          num_blocks = 2,
          tc_filters = 32,
          attention_size = 16,
          condition_sequence_length=20,
          inference_sequence_length=20):
  """SNAIL sequence encoder described in https://arxiv.org/abs/1707.03141."""
  with tf.variable_scope("snail"):
    conv_torso_fn = functools.partial(conv_torso_fn, is_training=is_training)
    feature_points, end_points = snt.BatchApply(conv_torso_fn)(image, aux_input)
    sequence_length = condition_sequence_length + inference_sequence_length
    x = feature_points
    for i in range(num_blocks):
      x = snail.TCBlock(x, sequence_length, tc_filters, scope="tc%d" % i)
      x, ep = snail.AttentionBlock(
          x, attention_size, attention_size, scope="attn%d" % i)
      end_points["attn_probs/%d" % i] = ep["attn_prob"]
    estimated_pose = LinearHead(x, output_size)
    return estimated_pose, end_points


@gin.configurable
def MultiHeadMLP(net,
                 action_sizes,
                 num_waypoints,
                 fc_layers,
                 is_training,
                 stop_gradient_future_waypoints = True):
  """Multihead MLP outputs."""
  timesteps = net.shape[1].value if len(net.shape) == 3 else 1
  def MLPFn(x, num_waypoints):
    """Helper function for building MLP."""
    head_outputs = []
    for action_size in action_sizes:
      head = layers.stack(
          x, layers.fully_connected, fc_layers, activation_fn=tf.nn.relu)
      head = layers.fully_connected(
          head, action_size * num_waypoints, activation_fn=None)
      if timesteps != 1:
        head_outputs.append(
            tf.reshape(head, [-1, timesteps, num_waypoints, action_size]))
      else:
        head_outputs.append(
            tf.reshape(head, [-1, num_waypoints, action_size]))
    return head_outputs
  # Predict remaining waypoints with a separate model and block gradients
  # from flowing back to `net`.
  if num_waypoints > 1 and stop_gradient_future_waypoints:
    with tf.variable_scope("action_trajectory", reuse=tf.AUTO_REUSE):
      components_1 = MLPFn(net, 1)
    with tf.variable_scope("auxiliary_trajectory", reuse=tf.AUTO_REUSE):
      if is_training:
        # We only stop gradient during training, so we can still compute
        # saliencies using the eval graph.
        net = tf.stop_gradient(net)
      components_2 = MLPFn(net, num_waypoints-1)
    # Concatenate across time dimension.
    return [tf.concat([c1, c2], axis=-2)
            for c1, c2 in zip(components_1, components_2)]
  else:
    return MLPFn(net, num_waypoints)
