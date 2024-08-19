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

"""Implementation of building blocks from https://arxiv.org/abs/1707.03141.

Implementation here is designed to match pseudocode in the paper.
"""

from typing import Text

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers


def CausalConv(x, dilation_rate, filters, kernel_size=2, scope = ""):
  """Performs causal dilated 1D convolutions.

  Args:
    x : Tensor of shape (batch_size, steps, input_dim).
    dilation_rate: Dilation rate of convolution.
    filters: Number of convolution filters.
    kernel_size: Width of convolution kernel. SNAIL paper uses 2 for all
      experiments.
    scope: Variable scope for this layer.
  Returns:
    y: Tensor of shape (batch_size, new_steps, D).
  """
  with tf.variable_scope(scope):
    causal_pad_size = (kernel_size - 1) * dilation_rate
    # Pad sequence dimension.
    x = tf.pad(x, [[0, 0], [causal_pad_size, 0], [0, 0]])
    return layers.conv1d(
        x,
        filters,
        kernel_size=kernel_size,
        padding="VALID",
        rate=dilation_rate)


def DenseBlock(x, dilation_rate, filters, scope = ""):
  r"""SNAIL \'dense block\' with gated activation and concatenation.

  Args:
    x : Tensor of shape [batch, time, channels].
    dilation_rate: Dilation rate of convolution.
    filters: Number of convolution filters.
    scope: Variable scope for this layer.
  Returns:
    y: Tensor of shape [batch, time, channels + filters].
  """
  with tf.variable_scope(scope):
    xf = CausalConv(x, dilation_rate, filters, scope="xf")
    xg = CausalConv(x, dilation_rate, filters, scope="xg")
    activations = tf.nn.tanh(xf) * tf.nn.sigmoid(xg)
    return tf.concat([x, activations], axis=2)


def TCBlock(x, sequence_length, filters, scope = ""):
  """A stack of DenseBlocks with exponentially increasing dilations.

  Args:
    x : Tensor of shape [batch, sequence_length, channels].
    sequence_length: Sequence length of x.
    filters: Number of convolution filters.
    scope: Variable scope for this layer.
  Returns:
    y: Tensor of shape [batch, sequence_length, channels + filters].
  """
  with tf.variable_scope(scope):
    for i in range(1, int(np.ceil(np.log2(sequence_length)))+1):
      x = DenseBlock(x, 2**i, filters, scope="DenseBlock_%d" % i)
    return x


def CausallyMaskedSoftmax(x):
  """Causally masked Softmax. Zero out probabilities before and after norm.

  pre-softmax logits are masked by setting upper diagonal to -inf:

  |a  0, 0|    |0, -inf, -inf|
  |b, d, 0|  + |0,   0,  -inf|
  |c, e, f|    |0,   0,    0 |

  Args:
    x: Batched tensor of shape [batch_size, T, T].
  Returns:
    Softmax where each row corresponds to softmax vector for each query.
  """
  lower_diag = tf.linalg.band_part(x, -1, 0)
  upper_diag = -np.inf * tf.ones_like(x)
  upper_diag = tf.linalg.band_part(upper_diag, 0, -1)
  upper_diag = tf.linalg.set_diag(
      upper_diag, tf.zeros_like(tf.linalg.diag_part(x)))
  x = lower_diag + upper_diag
  softmax = tf.nn.softmax(x)
  return tf.linalg.band_part(softmax, -1, 0)


def AttentionBlock(x, key_size, value_size, scope = ""):
  """Self-attention key-value lookup, styled after Vaswani et al. '17.

  query and key are of shape [T, K]. query * transpose(key) yields logits of
  shape [T, T]. logits[i, j] corresponds to unnormalized attention vector over
  values [T, V] for each timestep i. Because this attention is over a set of
  temporal values, we causally mask the pre-softmax logits[i, j] := 0, for all
  j > i.

  Citations:
    Vaswani et al. '17: Attention is All you need
      https://arxiv.org/abs/1706.03762.

  Args:
    x: Input tensor of shape [batch, sequence_length, channels].
    key_size: Integer key dimensionality.
    value_size: Integer value dimensionality.
    scope: Variable scope for this layer.
  Returns:
    result: Tensor of shape [batch, sequence_length, channels + value_size]
    end_points: Dictionary of intermediate values (e.g. debugging).
  """
  end_points = {}
  with tf.variable_scope(scope):
    key = layers.fully_connected(x, key_size, activation_fn=None)  # [T, K]
    query = layers.fully_connected(x, key_size, activation_fn=None)  # [T, K]
    logits = tf.matmul(query, key, transpose_b=True)  # [T, T]
    # Useful for visualizing attention alignment matrices.
    probs = CausallyMaskedSoftmax(logits/np.sqrt(key_size))  # [T, T]
    end_points["attn_prob"] = probs
    values = layers.fully_connected(x, value_size, activation_fn=None)  # [T, V]
    read = tf.matmul(probs, values)  # [T, V]
    result = tf.concat([x, read], axis=2)  # [T, K + V]
    return result, end_points
