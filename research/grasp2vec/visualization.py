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

# Lint as: python2, python3
"""Visualization utilities for models, mostly for tensorboard summaries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys

import gin
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers

try:
  from cvx2 import latest as cv2  # pylint: disable=g-import-not-at-top
except ImportError:


def plot_labels(labels, max_label=1, predictions=None, name=''):
  """Plots integer labels and optionally predictions as images.

  By default takes the first 3 in the batch.

  Args:
    labels: Batch x 1 size tensor of labels
    max_label:  An integer indicating the largest possible label
    predictions: Batch x max_label size tensor of predictions (range 0-1.0)
    name: string to name tensorflow summary
  """
  if max_label > 1:
    labels = tf.one_hot(
        labels, max_label, on_value=1.0, off_value=0.0, dtype=tf.float32)
  labels_image = tf.reshape(labels[:3], (1, 3, max_label, 1))
  empty_image = tf.zeros((1, 3, max_label, 1))
  image = tf.concat([labels_image, empty_image, empty_image], axis=-1)
  if predictions is not None:
    pred_image = tf.reshape(predictions[:3], (1, 3, 4, 1))
    image2 = tf.concat([empty_image, pred_image, empty_image], axis=-1)
    image = tf.concat([image, image2], axis=1)
  tf.summary.image('labels_' + six.ensure_str(name), image, max_outputs=1)


def plot_distances(pregrasp, goal, postgrasp):
  """Plot evaluation metrics for grasp2vec."""
  correct_distances = tf.norm(pregrasp - (goal + postgrasp), axis=1)
  incorrect_distances = tf.norm(pregrasp - pregrasp[::-1], axis=1)
  goal_distances = tf.norm(goal - goal[::-1], axis=1)
  tf.summary.histogram('correct_distances', correct_distances)
  tf.summary.histogram('goal_distances', goal_distances)
  tf.summary.histogram('incorrect_distances', incorrect_distances)
  tf.summary.histogram('pregrasp_sizes', tf.norm(pregrasp, axis=1))
  tf.summary.histogram('postgrasp_sizes', tf.norm(postgrasp, axis=1))
  tf.summary.histogram('goal_sizes', tf.norm(goal, axis=1))
  # Cosine similarity metric between adjacent minibatch elements.
  goal_normalized = goal / (1e-7 + tf.norm(goal, axis=1, keep_dims=True))
  similarity = tf.reduce_sum(
      goal_normalized[:-1] * goal_normalized[1:], axis=1)
  tf.summary.histogram('goal_cosine_similarity', similarity)


def add_heatmap_summary(feature_query, feature_map, name):
  """Plots dot produce of feature_query on feature_map.

  Args:
    feature_query: Batch x embedding size tensor of goal embeddings
    feature_map: Batch x h x w x embedding size of pregrasp scene embeddings
    name: string to name tensorflow summaries
  Returns:
     Batch x h x w x 1 heatmap
  """
  batch, dim = feature_query.shape
  reshaped_query = tf.reshape(feature_query, (int(batch), 1, 1, int(dim)))
  heatmaps = tf.reduce_sum(
      tf.multiply(feature_map, reshaped_query), axis=3, keep_dims=True)
  tf.summary.image(name, heatmaps)
  shape = tf.shape(heatmaps)
  softmaxheatmaps = tf.nn.softmax(tf.reshape(heatmaps, (int(batch), -1)))
  tf.summary.image(
      six.ensure_str(name) + 'softmax', tf.reshape(softmaxheatmaps, shape))
  return heatmaps


def add_spatial_softmax(heatmaps, images):
  locations_ij = contrib_layers.spatial_softmax(
      heatmaps, temperature=0.1, trainable=False)
  # spatial_softmax.BuildSpatialSoftmax returns [x1, ..., xN, y1, ..., yN] in
  # the inner dimension while layers.spatial_softmax returns
  # [i1, j1, ... iN, jN].
  y, x = tf.split(locations_ij, 2, axis=-1)
  locations_xy = tf.expand_dims(tf.concat([x, y], axis=-1), axis=1)
  add_spatial_soft_argmax_viz(images, heatmaps, locations_xy)
  return locations_xy


def np_render_keypoints(image, locations, num_images=3, dot_radius=3):
  """Computes rasterized spatial soft argmax locations overlaid on an image.

  Args:
    image: np.array of shape (N, H, W, 3) where N is the batch size.
    locations: np.array of shape (N, C, 2), where locations[n,c] are the i,j
      argmax coordinates of the cth softmax channel of the nth minibatch item.
    num_images: How many images to return in the batch dimension. Must be
      greater than the max_outputs value passed to tf.summary.image.
    dot_radius: Radius of the dots to paint at each location.

  Returns:
    Rasterized image in the form of a np.array of shape
    (num_images, h, w, 3).
  """
  # Make sure we do not attempt to index images which do not exist.
  num_images = np.minimum(num_images, image.shape[0])
  _, h, w, _ = image.shape
  mx, my = np.meshgrid(np.arange(w), np.arange(h))
  num_points = locations.shape[1]
  images = []
  for i in range(num_images):
    # Convert image to grey and reduce contrast, so we see the dots.
    img = np.tile(np.mean(image[i], axis=2, keepdims=True), [1, 1, 3])
    img = img / 2.0 + 0.4
    # Point colors.
    hues = np.linspace(0, 1, num_points + 1)[:-1]
    colors = [np.array(colorsys.hsv_to_rgb(hue, 1.0, 0.9)) for hue in hues]
    # (i, j) = (-1, -1) is the top left corner of the image. -1 -> 0
    xs = np.round((locations[i, :, 0] + 1.0) * w / 2.0).astype(np.int)
    ys = np.round((locations[i, :, 1] + 1.0) * h / 2.0).astype(np.int)
    for x, y, color in zip(xs, ys, colors):
      # Paint a dot of color at (x, y).
      dist_x = x - mx
      dist_y = y - my
      dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
      weight = np.maximum(np.minimum(dot_radius - dist, 1.0), 0.0)
      weight = np.tile(np.expand_dims(weight, 2), [1, 1, 3])
      img = img * (1 - weight) + weight * color.reshape([1, 1, 3])
    img = (img*255).astype(np.uint8)
    images.append(img)
  # Concatenate along batch dimensions.
  return np.stack(images, 0)


@gin.configurable
def add_spatial_soft_argmax_viz(image,
                                softmax,
                                locations,
                                max_outputs=3,
                                num_groups=1,
                                num_rows=1):
  """Generates TensorBoard visualization summaries for spatial softmax models.

  Args:
    image: Image tensor of shape (N, H, W, 3).
    softmax: Image features tensor of shape [N, H, W, C], where C is the number
      of spatial softmax maps.
    locations: Tensor of shape (N, C, 2), where locations[n,c] are the i,j
      argmax coordinates of the cth softmax map of the nth minibatch item.
    max_outputs: Maximum number of minibatch items to output in each summary.
    num_groups: Number of groups to subdivide the softmax channels into for
      visualization.
    num_rows: Number of rows per softmax layer visualization.
  """
  # Compute batch histogram summaries for mean x, y.
  tf.summary.histogram('x', locations[:, :, 0])
  tf.summary.histogram('y', locations[:, :, 1])

  # Average softmax maps.
  softmax_avg_channel = tf.reduce_mean(softmax, 3, keep_dims=True)
  tf.summary.image('SpatialSoftmax/softmax_avg', softmax_avg_channel)

  # Overlay of soft argmax locations on image.
  softmax_keypoints_image = tf.py_func(np_render_keypoints,
                                       [image, locations, max_outputs],
                                       [tf.uint8])[0]
  tf.summary.image(
      'SpatialSoftmax/locations',
      softmax_keypoints_image,
      max_outputs=max_outputs)

  if num_groups > 1:
    channel_groups = tf.split(softmax, num_groups, axis=3)
    for i, channel_group in enumerate(channel_groups):
      tf.summary.image('SpatialSoftmax/softmax_group_{}'.format(i),
                       get_softmax_viz(image, channel_group, num_rows))
  else:
    tf.summary.image('SpatialSoftmax/softmax',
                     get_softmax_viz(image, softmax, num_rows))


def get_softmax_viz(image, softmax, nrows=None):
  """Arrange softmax maps in a grid and superimpose them on the image."""
  softmax_shape = tf.shape(softmax)
  batch_size = softmax_shape[0]
  target_height = softmax_shape[1] * 2
  target_width = softmax_shape[2] * 2
  num_points = softmax_shape[3]

  if nrows is None:
    # Find a number of rows such that the arrangement is as square as possible.
    num_points_float = tf.cast(num_points, tf.float32)
    nfsqrt = tf.cast(tf.floor(tf.sqrt(num_points_float)), tf.int32)
    divs = tf.range(1, nfsqrt + 1)
    remainders = tf.mod(num_points_float, tf.cast(divs, tf.float32))
    divs = tf.gather(divs, tf.where(tf.equal(remainders, 0)))
    nrows = tf.reduce_max(divs)
  ncols = tf.cast(num_points / nrows, tf.int32)
  nrows = tf.cast(nrows, tf.int32)
  # Normalize per channel
  img = softmax / tf.reduce_max(softmax, axis=[1, 2], keepdims=True)
  # Use softmax as hue and saturation and original image as value of HSV image.
  greyimg = tf.image.rgb_to_grayscale(image)
  greyimg = tf.image.resize_images(greyimg, [target_height, target_width])
  greyimg = tf.tile(greyimg, [1, 1, 1, num_points])
  greyimg = tf.reshape(greyimg,
                       [batch_size, target_height, target_width, num_points, 1])
  img = tf.image.resize_images(img, [target_height, target_width])
  img = tf.reshape(img,
                   [batch_size, target_height, target_width, num_points, 1])
  img = tf.concat([img / 2.0 + 0.5, img, greyimg * 0.7 + 0.3], axis=4)

  # Rearrange channels into a ncols x nrows grid.
  img = tf.reshape(img,
                   [batch_size, target_height, target_width, nrows, ncols, 3])
  img = tf.transpose(img, [0, 3, 1, 4, 2, 5])
  img = tf.reshape(img,
                   [batch_size, target_height * nrows, target_width * ncols, 3])

  img = tf.image.hsv_to_rgb(img)
  return img


def _put_text(imgs, texts):
  """Python function that renders text onto a image."""
  result = np.empty_like(imgs)
  for i in range(imgs.shape[0]):
    text = texts[i]
    if isinstance(text, bytes):
      text = six.ensure_text(text)
    # You may need to adjust text size and position and size.
    # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
    result[i, :, :, :] = cv2.putText(
        imgs[i, :, :, :], str(text), (0, 30),
        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 1), 2)
  return result


def tf_put_text(imgs, texts):
  """Adds text to an image tensor."""
  return tf.py_func(_put_text, [imgs, texts], Tout=imgs.dtype)
