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

"""Common configurable image manipulation methods for use in preprocessors."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import tensorflow as tf
from typing import Sequence


def RandomCropImages(images, input_shape,
                     target_shape):
  """Crop a part of given shape from a random location in a list of images.

  Args:
    images: List of tensors of shape [batch_size, h, w, c].
    input_shape: Shape [h, w, c] of the input images.
    target_shape: Shape [h, w] of the cropped output.

  Raises:
    ValueError: In case the either the input_shape or the target_shape have a
      wrong length.

  Returns:
    crops: List of cropped tensors of shape [batch_size] + target_shape.
  """
  if len(input_shape) != 3:
    raise ValueError(
        'The input shape has to be of the form (height, width, channels) '
        'but has len {}'.format(len(input_shape)))
  if len(target_shape) != 2:
    raise ValueError('The target shape has to be of the form (height, width) '
                     'but has len {}'.format(len(target_shape)))
  max_y = int(input_shape[0]) - int(target_shape[0])
  max_x = int(input_shape[1]) - int(target_shape[1])
  with tf.control_dependencies(
      [tf.assert_greater_equal(max_x, 0),
       tf.assert_greater_equal(max_y, 0)]):
    offset_y = tf.random_uniform((), maxval=max_y + 1, dtype=tf.int32)
    offset_x = tf.random_uniform((), maxval=max_x + 1, dtype=tf.int32)
    return [
        tf.image.crop_to_bounding_box(img, offset_y, offset_x,
                                      int(target_shape[0]),
                                      int(target_shape[1])) for img in images
    ]


def CenterCropImages(images, input_shape,
                     target_shape):
  """Take a central crop of given size from a list of images.

  Args:
    images: List of tensors of shape [batch_size, h, w, c].
    input_shape: Shape [h, w, c] of the input images.
    target_shape: Shape [h, w, c] of the cropped output.

  Returns:
    crops: List of cropped tensors of shape [batch_size] + target_shape.
  """
  if len(input_shape) != 3:
    raise ValueError(
        'The input shape has to be of the form (height, width, channels) '
        'but has len {}'.format(len(input_shape)))
  if len(target_shape) != 2:
    raise ValueError('The target shape has to be of the form (height, width) '
                     'but has len {}'.format(len(target_shape)))
  if input_shape[0] == target_shape[0] and input_shape[1] == target_shape[1]:
    return [image for image in images]

  # Assert all images have the same shape.
  assert_ops = []
  for image in images:
    assert_ops.append(
        tf.assert_equal(
            input_shape[:2],
            tf.shape(image)[1:3],
            message=('All images must have same width and height'
                     'for CenterCropImages.')))
  offset_y = int(input_shape[0] - target_shape[0]) // 2
  offset_x = int(input_shape[1] - target_shape[1]) // 2
  with tf.control_dependencies(assert_ops):
    crops = [
        tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                      target_shape[0], target_shape[1])
        for image in images
    ]
  return crops


@gin.configurable
def ApplyPhotometricImageDistortions(images,
                                     random_brightness=False,
                                     max_delta_brightness=0.125,
                                     random_saturation=False,
                                     lower_saturation=0.5,
                                     upper_saturation=1.5,
                                     random_hue=False,
                                     max_delta_hue=0.2,
                                     random_contrast=False,
                                     lower_contrast=0.5,
                                     upper_contrast=1.5,
                                     random_noise_level=0.0,
                                     random_noise_apply_probability=0.5):
  """Apply photometric distortions to the input images.

  Args:
    images: Tensor of shape [batch_size, h, w, 3] containing a batch of images
      to apply the random photometric distortions to.
    random_brightness: Boolean; whether to randomly adjust the brightness.
    max_delta_brightness: Float; maximum delta for the random value by which to
      adjust the brightness.
    random_saturation: Boolean; whether to randomly adjust the saturation.
    lower_saturation: Float; lower bound of the range from which to chose a
      random value for the saturation.
    upper_saturation: Float; upper bound of the range from which to chose a
      random value for the saturation.
    random_hue: Boolean; whether to randomly adjust the hue.
    max_delta_hue: Float; maximum delta for the random value by which to adjust
      the hue.
    random_contrast: Boolean; whether to randomly adjust the contrast.
    lower_contrast: Float; lower bound of the range from which to chose a random
      value for the contrast.
    upper_contrast: Float; upper bound of the range from which to chose a random
      value for the contrast.
    random_noise_level: Standard deviation of the gaussian from which to sample
      random noise to be added to the images. If 0.0, no noise is added.
    random_noise_apply_probability: Probability of applying additive random
      noise to the images.

  Returns:
    images: Tensor of shape [batch_size, h, w, 3] containing a batch of images
      resulting from applying random photometric distortions to the inputs.
  """
  with tf.variable_scope('photometric_distortions'):
    # Adjust brightness to a random level.
    if random_brightness:
      delta = tf.random_uniform([], -max_delta_brightness, max_delta_brightness)
      for i, image in enumerate(images):
        images[i] = tf.image.adjust_brightness(image, delta)

    # Adjust saturation to a random level.
    if random_saturation:
      lower = lower_saturation
      upper = upper_saturation
      saturation_factor = tf.random_uniform([], lower, upper)
      for i, image in enumerate(images):
        images[i] = tf.image.adjust_saturation(image, saturation_factor)

    # Randomly shift the hue.
    if random_hue:
      delta = tf.random_uniform([], -max_delta_hue, max_delta_hue)
      for i, image in enumerate(images):
        images[i] = tf.image.adjust_hue(image, delta)

    # Adjust contrast to a random level.
    if random_contrast:
      lower = lower_contrast
      upper = upper_contrast
      contrast_factor = tf.random_uniform([], lower, upper)
      for i, image in enumerate(images):
        images[i] = tf.image.adjust_contrast(image, contrast_factor)

    # Add random Gaussian noise.
    if random_noise_level:
      for i, image in enumerate(images):
        rnd_noise = tf.random_normal(tf.shape(image), stddev=random_noise_level)
        img_shape = tf.shape(image)
        def ImageClosure(value):
          return lambda: value
        image = tf.cond(
            tf.reduce_all(
                tf.greater(
                    tf.random.uniform([1]), random_noise_apply_probability)),
            ImageClosure(image), ImageClosure(image + rnd_noise))
        images[i] = tf.reshape(image, img_shape)

    # Clip to valid range.
    for i, image in enumerate(images):
      images[i] = tf.clip_by_value(image, 0.0, 1.0)
  return images
