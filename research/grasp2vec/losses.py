# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""Losses for feature learning.

Implements several loss functions for training models.
"""

import gin
from six.moves import range
import tensorflow.compat.v1 as tf  # tf
from tf_slim import losses as slim_losses

from tensorflow.keras import layers as contrib_layers


def L2ArithmeticLoss(pregrasp_embedding, goal_embedding, postgrasp_embedding,
                     mask):
  """Calculates the L2 Arithmetic loss.

  ||pregrasp_embedding*signs[0] + goal_embedding*signs[1] +
  postgrasp_embedding*signs[2] ||_2

  Args:
    pregrasp_embedding: batch of embeddings of the pregrasp image.
    goal_embedding: batch of embeddings of the goal image.
    postgrasp_embedding: batch of embeddings of the postgrasp image.
    mask: tensor of 0s and 1s indicating which inputs should be used to
    calculate loss.
  Returns:
    A scalar loss
  """
  def _ComputeLoss():
    raw_distances = pregrasp_embedding - goal_embedding - postgrasp_embedding
    distances = tf.reduce_sum(raw_distances ** 2, axis=1)
    _, mask1_data = tf.dynamic_partition(distances, mask, 2)
    loss = tf.cast(tf.reduce_mean(mask1_data), tf.float32)
    return loss
  return tf.cond(tf.reduce_sum(mask) > 0, _ComputeLoss,
                 lambda: tf.zeros(1, tf.float32))


@gin.configurable
def TripletLoss(pregrasp_embedding, goal_embedding, postgrasp_embedding):
  """Uses semi-hard mining triplet loss.

  ||pregrasp_embedding*signs[0] - postgrasp_embedding*signs[1] +
  goal_embedding*signs[2] ||_2

  Args:
    pregrasp_embedding: batch of embeddings of the pregrasp image
    goal_embedding: batch of embeddings of the goal image
    postgrasp_embedding: batch of embeddings of the postgrasp image
  Returns:
    A scalar loss
  """

  pair_a = tf.nn.l2_normalize(pregrasp_embedding-postgrasp_embedding, axis=1)
  pair_b = tf.nn.l2_normalize(goal_embedding, axis=1)
  labels = tf.range(pregrasp_embedding.shape[0], dtype=tf.int32)
  labels = tf.tile(labels, [2])
  pairs = tf.concat([pair_a, pair_b], axis=0)
  loss = slim_losses.metric_learning.triplet_semihard_loss(
      labels, pairs, margin=3.0)
  return loss, pairs, labels


def CosineArithmeticLoss(pregrasp_embedding, goal_embedding,
                         postgrasp_embedding, mask):
  """Calculates the cosine arithmetic loss.

  ||pregrasp_embedding*signs[0] - postgrasp_embedding*signs[1] +
  goal_embedding*signs[2] ||_2

  Args:
    pregrasp_embedding: batch of embeddings of the pregrasp image.
    goal_embedding: batch of embeddings of the goal image.
    postgrasp_embedding: batch of embeddings of the postgrasp image.
    mask: tensor of 0s and 1s indicating which inputs should be used to
    calculate loss.
  Returns:
    A scalar loss.
  """
  mask = tf.cast(mask, tf.int32)
  mask = tf.reshape(mask, (-1,))
  def _ComputeLoss():
    pair_a = tf.nn.l2_normalize(pregrasp_embedding-postgrasp_embedding, axis=1)
    pair_b = tf.nn.l2_normalize(goal_embedding, axis=1)
    distances = tf.losses.cosine_distance(
        pair_a, pair_b, axis=1, reduction=tf.losses.Reduction.NONE)
    _, mask1_data = tf.dynamic_partition(distances, mask, 2)
    loss = tf.cast(tf.reduce_mean(mask1_data), tf.float32)
    return loss
  return tf.cond(tf.reduce_sum(mask) > 0, _ComputeLoss,
                 lambda: tf.zeros(1, tf.float32))


def KeypointAccuracy(keypoints, labels):
  """Calculates the quadrant accuracy of keypoints.

  Only for use with Shapes dataset.

  Args:
    keypoints: batch x 2 tensor of floats from spatial softmax.
    labels: batch x 1 integer label of correct quadrant. Quadrants correspond to
      quadrant_centers defined below.
  Returns:
    accuracy of predictions.
    cross entropy loss.
  """
  keypoints = tf.reshape(keypoints, (-1, 2))
  quadrant_centers = tf.constant([[0.5, -0.5],
                                  [-0.5, -0.5],
                                  [0.5, 0.5],
                                  [-0.5, 0.5]], dtype=tf.float32)
  logits = tf.matmul(keypoints, quadrant_centers, transpose_b=True)
  predictions = tf.nn.softmax(logits)
  correct = tf.cast(tf.equal(labels, tf.argmax(predictions, 1)), tf.float32)
  labels_onehot = tf.one_hot(labels, 4, on_value=1.0, off_value=0.0,
                             dtype=tf.float32)
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels_onehot, logits=logits))
  return tf.reduce_mean(correct), loss


def SendToZeroLoss(tensor, mask):
  """Calculates the distance of the inputs from zero.

  Args:
    tensor: batch x _ tensor.
    mask: binary tensor indicating which inputs in the batch should be counted
    in the loss.
  Returns:
    A scalar loss.
  """
  mask = tf.cast(mask, tf.int32)
  mask = tf.reshape(mask, (-1,))
  def _ComputeLoss():
    distances = tf.norm(tensor, axis=1)
    _, mask1_data = tf.dynamic_partition(distances, mask, 2)
    loss = tf.cast(tf.reduce_mean(mask1_data), tf.float32)
    return loss
  return tf.cond(tf.reduce_sum(mask) > 0, _ComputeLoss,
                 lambda: tf.zeros(1, tf.float32))


@gin.configurable
def NPairsLoss(pregrasp_embedding, goal_embedding, postgrasp_embedding,
               non_negativity_constraint=False):
  """Uses npairs_loss in both directions.

  Args:
    pregrasp_embedding: Batch of embeddings of the pregrasp image
    goal_embedding: Batch of embeddings of the goal image
    postgrasp_embedding: Batch of embeddings of the postgrasp image
    non_negativity_constraint: If True, passes pre - post through ReLU to
      represent valid set of objects. This is used to handle situations where
      postgrasp embedding contains objects that are not contained in the
      pregrasp embedding (e.g. object was dumped into the bin during the grasp).
  Returns:
    A scalar loss
  """
  pair_a = pregrasp_embedding - postgrasp_embedding
  if non_negativity_constraint:
    pair_a = tf.nn.relu(pair_a)

  pair_b = goal_embedding
  labels = tf.range(pregrasp_embedding.shape[0], dtype=tf.int32)
  loss_1 = slim_losses.metric_learning.npairs_loss(labels, pair_a, pair_b)
  loss_2 = slim_losses.metric_learning.npairs_loss(labels, pair_b, pair_a)
  tf.summary.scalar('npairs_loss1', loss_1)
  tf.summary.scalar('npairs_loss2', loss_2)
  return loss_1+loss_2


def NPairsLossMultilabel(pregrasp_embedding, goal_embedding,
                         postgrasp_embedding, grasp_success, params):
  """Uses npairs_loss in both directions.

  Args:
    pregrasp_embedding: Batch of embeddings of the pregrasp image
    goal_embedding: Batch of embeddings of the goal image
    postgrasp_embedding: Batch of embeddings of the postgrasp image
    grasp_success: Batch of 1s and 0s indicating grasp success.
    params: Parameters for loss. Currently unused.
  Returns:
    A scalar loss
  """
  del params
  pair_a = pregrasp_embedding - postgrasp_embedding
  pair_b = goal_embedding
  grasp_success = tf.cast(tf.squeeze(grasp_success), tf.int32)
  range_tensor = (tf.range(pregrasp_embedding.shape[0],
                           dtype=tf.int32))*grasp_success
  labels = tf.one_hot(range_tensor, pregrasp_embedding.shape[0]+1,
                      on_value=1, off_value=0)
  sparse_labels = [
      contrib_layers.dense_to_sparse(labels[i]) for i in range(labels.shape[0])
  ]
  loss_1 = slim_losses.metric_learning.npairs_loss_multilabel(
      sparse_labels, pair_a, pair_b)
  loss_2 = slim_losses.metric_learning.npairs_loss_multilabel(
      sparse_labels, pair_b, pair_a)

  tf.summary.scalar('npairs_loss1', loss_1)
  tf.summary.scalar('npairs_loss2', loss_2)
  return loss_1+loss_2


def MatchNormsLoss(anchor_tensors, paired_tensors):
  """A norm on the difference between the norms of paired tensors.

  Gradients are only applied to the paired_tensor.
  Args:
    anchor_tensors: batch of embeddings deemed to have a "correct" norm.
    paired_tensors: batch of embeddings that will be pushed to the norm of
      anchor_tensors.
  Returns:
    A scalar loss
  """
  anchor_norms = tf.stop_gradient(tf.norm(anchor_tensors, axis=1))
  paired_norms = tf.norm(paired_tensors, axis=1)
  tf.summary.histogram('norms_difference', tf.nn.l2_loss(anchor_norms
                                                         -paired_norms))
  loss = tf.reduce_mean(tf.nn.l2_loss(anchor_norms-paired_norms))
  return loss


def _GetSoftMaxResponse(goal_embedding, scene_spatial):
  """Max response of an embeddings across a spatial feature map.

  The goal_embedding is multiplied across the spatial dimensions of the
  scene_spatial to generate a heatmap. Then the spatial softmax-pooled value of
  this heatmap is returned. If the goal_embedding and scene_spatial are aligned
  to the same space, then _GetSoftMaxResponse returns larger values if the
  object is present in the scene, and smaller values if the object is not.

  Args:
    goal_embedding: A batch x D tensor embedding of the goal image.
    scene_spatial: A batch x H x W x D spatial feature map tensor.
  Returns:
    max_heat: A tensor of length batch.
    max_soft: The max value of the softmax (ranges between 0 and 1.0)
  """
  batch, dim = goal_embedding.shape
  reshaped_query = tf.reshape(goal_embedding, (int(batch), 1, 1, int(dim)))
  scene_heatmap = tf.reduce_sum(tf.multiply(scene_spatial,
                                            reshaped_query), axis=3,
                                keep_dims=True)
  scene_heatmap_flat = tf.reshape(scene_heatmap, (batch, -1))
  max_heat = tf.reduce_max(scene_heatmap_flat, axis=1)
  scene_softmax = tf.nn.softmax(scene_heatmap_flat, axis=1)
  max_soft = tf.reduce_max(scene_softmax, axis=1)
  return max_heat, max_soft


def TYloss(pregrasp_spatial, postgrasp_spatial, goal_embedding):
  """Encourages the response of the pregrasp scene to be higher than postgrasp.

  If an object is in a scene, the max response of the goal embedding in the
  scene should be high (something was detected). If the object is not in the
  scene, the max response should be low (nothing was detected).

  A likelihood ratio based approach seems to be more stable than independent
  sigmoid cross entropy losses for positives and negatives. In other words,
  if p(object) is pregrasp similarity and q is postgrasp similartiy, the
  objective is: max p(object)/q(object), or equivalently,

  max log p(object) - log q(object)
  min log q(object) - log p(object)

  Args:
    pregrasp_spatial: A batch x H x W x D spatial feature map tensor of the
      pregrasp scene.
    postgrasp_spatial: A batch x H x W x D spatial feature map tensor of the
      postgrasp scene.
    goal_embedding: A batch x D tensor embedding of the goal image.
  Returns:
    loss: A scalar value.
  """
  pregrasp_spatial = tf.math.l2_normalize(pregrasp_spatial, axis=-1)
  postgrasp_spatial = tf.math.l2_normalize(postgrasp_spatial, -1)
  goal_embedding = tf.math.l2_normalize(goal_embedding, -1)
  goal_embedding = tf.expand_dims(tf.expand_dims(goal_embedding, 1), 1)
  pre_similarity_map = tf.reduce_sum(
      pregrasp_spatial * goal_embedding, axis=-1)
  pre_similarity_max = tf.reduce_max(pre_similarity_map, axis=[1, 2])
  post_similarity_map = tf.reduce_sum(
      postgrasp_spatial * goal_embedding, axis=-1)
  post_similarity_max = tf.reduce_max(post_similarity_map, axis=[1, 2])
  return tf.reduce_mean(post_similarity_max - pre_similarity_max)
