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

"""Functions for building Task-Embedded Control Networks: https://arxiv.org/abs/1810.03237."""  # pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
from tensor2robot.layers import vision_layers
import tensorflow.compat.v1 as tf  # tf
from tf_slim import losses as slim_losses
from typing import Optional, Text, Tuple
from tensorflow.contrib import layers


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
    fc_layers = None,
    use_spatial_softmax = True):
  """Independently embed a (meta)-batch of images.

  Args:
    condition_image: A rank 4 tensor of images: [N, H, W, C].
    scope: Name of the tf variable_scope.
    reuse: The variable_scope reuse setting.
    fc_layers: An optional tuple of ints describing the number of units in each
      fully-connected hidden layer, or 1x1 conv layer when excluding spatial
      softmax.
    use_spatial_softmax: Whether to use a spatial softmax or not.
  Returns:
    A rank 2 tensor of embeddings: [N, embedding size] if spatial_softmax is
    True. Otherwise, a rank 4 tensor of visual features [N, H, W, embedding size]
  Raises:
    ValueError if `condition_image` has incorrect rank.
  """
  if len(condition_image.shape) != 4:
    raise ValueError(
        'Image has unexpected shape {}.'.format(condition_image.shape))
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    image_embedding, _ = vision_layers.BuildImagesToFeaturesModel(
        condition_image, use_spatial_softmax=use_spatial_softmax)
    if fc_layers is not None:
      if len(image_embedding.shape) == 2:
        image_embedding = layers.stack(
            image_embedding,
            layers.fully_connected,
            fc_layers[:-1],
            activation_fn=tf.nn.relu,
            normalizer_fn=layers.layer_norm)
        image_embedding = layers.fully_connected(
            image_embedding, fc_layers[-1], activation_fn=None)
      else:
        image_embedding = layers.stack(
            image_embedding,
            layers.conv2d,
            fc_layers[:-1],
            kernel_size=[1, 1],
            activation_fn=tf.nn.relu,
            normalizer_fn=layers.layer_norm)
        image_embedding = layers.conv2d(
            image_embedding, fc_layers[-1], activation_fn=None)
  return image_embedding


@gin.configurable
def reduce_temporal_embeddings(
    temporal_embedding,
    output_size,
    scope,
    reuse=tf.AUTO_REUSE,
    conv1d_layers = (64,),
    fc_hidden_layers = (100,),
    combine_mode = 'temporal_conv'):
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
    combine_mode: How to reduce across time to get a fixed length vector.
  Returns:
    A rank 2 tensor: [N, output_size].
  Raises:
    ValueError if `temporal_embedding` has incorrect rank.
  """
  if len(temporal_embedding.shape) == 5:
    temporal_embedding = tf.reduce_mean(temporal_embedding, axis=[2, 3])
  if len(temporal_embedding.shape) != 3:
    raise ValueError('Temporal embedding has unexpected shape {}.'.format(
        temporal_embedding.shape))
  embedding = temporal_embedding
  with tf.variable_scope(scope, reuse=reuse, use_resource=True):
    if 'temporal_conv' not in combine_mode:
      # Just average
      embedding = tf.reduce_mean(embedding, axis=1)
    else:
      if conv1d_layers is not None:
        for num_filters in conv1d_layers:
          embedding = tf.layers.conv1d(
              embedding, num_filters, 10, activation=tf.nn.relu, use_bias=False)
          embedding = layers.layer_norm(embedding)
      if combine_mode == 'temporal_conv_avg_after':
        embedding = tf.reduce_mean(embedding, axis=1)
      else:
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


@gin.configurable
def compute_embedding_contrastive_loss(
    inf_embedding,
    con_embedding,
    positives = None,
    contrastive_loss_mode='both_directions'):
  """Compute triplet loss between inference and condition_embeddings.

  Expects embeddings to be L2-normalized.

  Args:
    inf_embedding: A rank 3 tensor: [num_tasks, num_inf_episodes, K].
    con_embedding: A rank 3 tensor: [num_tasks, num_con_episodes, K].
    positives: (Optional). A rank 1 bool tensor: [num_tasks]. If provided, instead
      of assigning positives to just the 1st task in the batch, it uses the positives
      given. Positives should be defined as if the 1st task was the anchor. When not
      provided, the 1st con_embedding is positive and every other con_embedding is
      negatives.
    contrastive_loss_mode: Which contrastive loss function to use.
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
  if positives is not None:
    labels = positives
  else:
    labels = tf.math.equal(tf.range(tf.shape(avg_con_embedding)[0]), 0)
  # Unlike TEC paper, use standard contrastive loss.
  # This does L2 distance in space
  if contrastive_loss_mode == 'default':
    # anchor_inf --> con embeddings
    embed_loss = slim_losses.metric_learning.contrastive_loss(
        labels, anchor, avg_con_embedding)
  elif contrastive_loss_mode == 'both_directions':
    # anchor_inf --> con embeddings and anchor_con --> inf embeddings.
    # Since data is paired, we know we can reuse the labels.
    # Seems to perform best.
    embed_loss1 = slim_losses.metric_learning.contrastive_loss(
        labels, anchor, avg_con_embedding)
    anchor_cond = avg_con_embedding[0:1]
    embed_loss2 = slim_losses.metric_learning.contrastive_loss(
        labels, anchor_cond, avg_inf_embedding)
    embed_loss = embed_loss1 + embed_loss2
  elif contrastive_loss_mode == 'reverse_direction':
    # anchor_con --> inf embeddings.
    anchor_cond = avg_con_embedding[0:1]
    embed_loss = slim_losses.metric_learning.contrastive_loss(
        labels, anchor_cond, avg_inf_embedding)
  elif contrastive_loss_mode == 'cross_entropy':
    # softmax(temperature * z^T c), does both directions by default.
    #
    # This should be similar to the InfoNCE contrastive loss, but is slightly
    # different because entries in the same batch may be for the same task.
    #
    # Performance untested.
    temperature = 2
    anchor_cond = avg_con_embedding[0:1]
    cosine_sim = tf.reduce_sum(anchor * avg_con_embedding, axis=1)
    loss1 = tf.keras.losses.binary_crossentropy(
        labels, temperature * cosine_sim, from_logits=True)
    cosine_sim_2 = tf.reduce_sum(anchor_cond * avg_inf_embedding, axis=1)
    loss2 = tf.keras.losses.binary_crossentropy(
        labels, temperature * cosine_sim_2, from_logits=True)
    embed_loss = loss1 + loss2
  elif contrastive_loss_mode == 'triplet':
    # Triplet loss
    labels = tf.tile(labels, [2])
    embeds = tf.concat([avg_inf_embedding, avg_con_embedding], axis=0)
    embed_loss = slim_losses.metric_learning.triplet_semihard_loss(
        labels, embeds, margin=3.0)
  else:
    raise ValueError('Did not understand contrastive_loss_mode')
  return embed_loss
