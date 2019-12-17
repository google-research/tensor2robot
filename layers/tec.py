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

"""Functions for building Task-Embedded Control Networks: https://arxiv.org/abs/1810.03237."""  # pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
from tensor2robot.layers import vision_layers
import tensorflow as tf  # tf
from typing import Optional, Text, Tuple
from tensorflow.contrib import layers
from tensorflow.contrib import losses as contrib_losses


def embed_fullstate(
    fullstate, embed_size, scope, reuse=tf.AUTO_REUSE, fc_layers=(100,)):
  """Embed full state pose (non-image) observations.

  Args:
    fullstate: A rank 2 tensor: [N, F].
    embed_size: Integer, the output embedding size.
    scope: Name of the tf variable scope.
    reuse: The variable_scope reuse setting.
    fc_layers: A tuple of ints describing the number of units in each hidden
      layer.
  Returns:
    A rank 2 tensor: [N, embed_size].
  """
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    embedding = layers.stack(
        fullstate, layers.fully_connected, fc_layers,
        activation_fn=tf.nn.relu, normalizer_fn=layers.layer_norm)
    embedding = layers.fully_connected(
        embedding, embed_size, activation_fn=None)
  return embedding


@gin.configurable
def embed_condition_images(
    condition_image,
    scope,
    reuse=tf.AUTO_REUSE,
    fc_layers = None):
  """Independently embed a (meta)-batch of images.

  Args:
    condition_image: A rank 4 tensor of images: [N, H, W, C].
    scope: Name of the tf variable_scope.
    reuse: The variable_scope reuse setting.
    fc_layers: An optional tuple of ints describing the number of units in each
      fully-connected hidden layer.
  Returns:
    A rank 2 tensor of embeddings: [N, embedding size].
  Raises:
    ValueError if `condition_image` has incorrect rank.
  """
  if len(condition_image.shape) != 4:
    raise ValueError(
        'Image has unexpected shape {}.'.format(condition_image.shape))
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    image_embedding, _ = vision_layers.BuildImagesToFeaturesModel(
        condition_image)
    if fc_layers is not None:
      image_embedding = layers.stack(
          image_embedding,
          layers.fully_connected,
          fc_layers[:-1],
          activation_fn=tf.nn.relu,
          normalizer_fn=layers.layer_norm)
      image_embedding = layers.fully_connected(
          image_embedding, fc_layers[-1], activation_fn=None)
  return image_embedding


@gin.configurable
def reduce_temporal_embeddings(
    temporal_embedding,
    output_size,
    scope,
    reuse=tf.AUTO_REUSE,
    conv1d_layers = (64,),
    fc_hidden_layers = (100,)):
  """Combine embedding across the episode temporal dimension.

  Args:
    temporal_embedding: A rank 3 tensor: [N, time dim, feature dim].
    output_size: The dimension of the output embedding.
    scope: Name of the tf variable_scope.
    reuse: The variable_scope reuse setting.
    conv1d_layers: An optional tuple of ints describing the number of feature
      maps in each 1D conv layer.
    fc_hidden_layers: A tuple of ints describing the number of units in each
      fully-connected hidden layer.
  Returns:
    A rank 2 tensor: [N, output_size].
  Raises:
    ValueError if `temporal_embedding` has incorrect rank.
  """
  if len(temporal_embedding.shape) != 3:
    raise ValueError('Temporal embedding has unexpected shape {}.'.format(
        temporal_embedding.shape))
  embedding = temporal_embedding
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    if conv1d_layers is not None:
      for num_filters in conv1d_layers:
        embedding = tf.layers.conv1d(
            embedding, num_filters, 10, activation=tf.nn.relu, use_bias=False)
        embedding = layers.layer_norm(embedding)
    embedding = layers.flatten(embedding)
    embedding = layers.stack(
        embedding,
        layers.fully_connected,
        fc_hidden_layers,
        activation_fn=tf.nn.relu,
        normalizer_fn=layers.layer_norm)
    embedding = layers.fully_connected(
        embedding, output_size, activation_fn=None)
  return embedding


def compute_embedding_contrastive_loss(
    inf_embedding,
    con_embedding,
    successes = None):
  """Compute triplet loss between inference and condition_embeddings.

  Expects embeddings to be L2-normalized.

  Args:
    inf_embedding: A rank 3 tensor: [num_tasks, num_inf_episodes, K].
    con_embedding: A rank 3 tensor: [num_tasks, num_con_episodes, K].
    successes: (Optional). A rank 2 tensor: [num_tasks, num_inf_episodes]. If
      provided, only con_embedding with corresponding successes=1.0 are assigned
      as positives (all failures are negatives). When not provided,
      con_embedding is always treated as a positive example.
  Returns:
    The contrastive loss computed using the task zero inf_embedding and
    each of the `num_tasks` con_embeddings.
  """
  if len(inf_embedding.shape) != 3:
    raise ValueError(
        'Unexpected inf_embedding shape: {}.'.format(inf_embedding.shape))
  if len(con_embedding.shape) != 3:
    raise ValueError(
        'Unexpected con_embedding shape: {}.'.format(con_embedding.shape))
  avg_inf_embedding = tf.reduce_mean(inf_embedding, axis=1)
  avg_con_embedding = tf.reduce_mean(con_embedding, axis=1)
  anchor = avg_inf_embedding[0:1]
  labels = tf.math.equal(tf.range(tf.shape(avg_con_embedding)[0]), 0)
  if successes is not None:
    inference_success = tf.math.equal(tf.reduce_mean(successes, axis=1), 1.0)
    labels = tf.logical_and(labels, inference_success)
  # Unlike TEC paper, use standard contrastive loss.
  embed_loss = contrib_losses.metric_learning.contrastive_loss(
      labels, anchor, avg_con_embedding)
  return embed_loss
