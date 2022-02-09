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

"""Shared utils for image distorton and cropping."""

from tensor2robot.preprocessors import image_transformations
import tensorflow.compat.v1 as tf


def maybe_distort_image_batch(images, mode):
  """Applies data augmentation to given images.

  Args:
    images: 4D Tensor (batch images) or 5D Tensor (batch of image sequences).
    mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

  Returns:
    Distorted images. Image distortion is identical for every image in the
      batch.
  """
  if mode == tf.estimator.ModeKeys.TRAIN:
    images = image_transformations.ApplyPhotometricImageDistortions([images])[0]
  return images


def maybe_distort_and_flip_image_batch(images, mode):
  """Applies data augmentation and random flips to given images.

  Args:
    images: 4D Tensor (batch images) or 5D Tensor (batch of image sequences).
    mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

  Returns:
    Distorted images. Image distortion is identical for every image in the
      batch.
  """
  if mode == tf.estimator.ModeKeys.TRAIN:
    images = image_transformations.ApplyPhotometricImageDistortions([images])[0]
    images = image_transformations.ApplyRandomFlips(images)
  return images


def preprocess_image(image,
                     mode,
                     is_sequence,
                     input_size,
                     target_size,
                     crop_size=None,
                     image_distortion_fn=maybe_distort_image_batch):
  """Shared preprocessing function for images.

  Args:
    image: A tf.Tensor for the input images, which is either a 4D Tensor (batch
      of images) or 5D Tensor (batch of sequences). It is assumed that all
      dimensions are constant, except the batch dimension.
    mode: (modekeys) specifies if this is training, evaluation or prediction.
    is_sequence: Should be True if input is a batch of sequences, and False
      otherwise.
    input_size: [h, w] of the input image
    target_size: [h, w] of the output image, expected to be equal or smaller
      than input size. If smaller, we do a crop of the image.
    crop_size: [h, w] of crop size. If None, defaults to target_size.
    image_distortion_fn: A function that takes an image tensor and the training
      mode as input and returns an image tensor of the same size as the input.

  Returns:
    A tf.Tensor for the batch of images / batch of sequences. If mode == TRAIN,
    this applies image distortion and crops the image randomly. Otherwise, it
    does not add image distortion and takes a crop from the center of the image.
  """
  leading_shape = tf.shape(image)[:-3]

  # Must be tf.float32 to distort.
  image = tf.image.convert_image_dtype(image, tf.float32)

  if is_sequence:
    # Flatten batch dimension.
    image = tf.reshape(image, [-1] + image.shape[-3:].as_list())

  crop_size = crop_size or target_size
  image = crop_image(
      image, mode, input_size=input_size, target_size=crop_size)
  # Reshape to target size.
  image = tf.image.resize_images(image, target_size)

  # Convert dtype and distort.
  image = image_distortion_fn(image, mode=mode)

  # Flatten back into a sequence.
  if is_sequence:
    tail_shape = tf.constant(list(target_size) + [3])
    full_final_shape = tf.concat([leading_shape, tail_shape], axis=0)
    image = tf.reshape(image, full_final_shape)
  return image


def crop_image(img, mode, input_size=(512, 640), target_size=(472, 472)):
  """Takes a crop of the image, either randomly or from the center.

  The crop is consistent across all images given in the batch.

  Args:
    img: 4D image Tensor [batch, height, width, channels].
    mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
    input_size: (height, width) of input.
    target_size: (height, width) of desired crop.

  Returns:
    img cropped to the desired size, randomly if mode == TRAIN and from the
    center otherwise.
  """
  if input_size == target_size:
    # Don't even bother adding the ops.
    return img
  input_height, input_width = input_size
  input_shape = (input_height, input_width, 3)
  target_shape = target_size

  if mode == tf.estimator.ModeKeys.TRAIN:
    crops = image_transformations.RandomCropImages([img],
                                                   input_shape=input_shape,
                                                   target_shape=target_shape)[0]

  else:
    crops = image_transformations.CenterCropImages([img],
                                                   input_shape=input_shape,
                                                   target_shape=target_shape)[0]
  return crops
