# coding=utf-8
# Copyright 2021 The Tensor2Robot Authors.
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
"""Tests for tensor2robot image_transformations."""

from absl.testing import parameterized
import numpy as np
from six.moves import range
from tensor2robot.preprocessors import image_transformations
import tensorflow.compat.v1 as tf  # tf


class ImageTransformationsTest(tf.test.TestCase, parameterized.TestCase):

  def _CreateRampTestImages(self, batch_size, height, width):
    """Creates a batch of test images of given size.

    Args:
      batch_size: Number of images to stack into a batch.
      height: Height of the image.
      width: Width of the image.

    Returns:
      images: Tensor of shape [batch_size, height, width, 3]. In each image
        the R-channel values are equal to the x coordinate of the pixel, in G-
        and B-channel values are equal to the y coordinate.
    """
    mesh_x, mesh_y = tf.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    mesh_x = tf.expand_dims(mesh_x, 2)
    mesh_y = tf.expand_dims(mesh_y, 2)
    image = tf.concat([mesh_x, mesh_y, mesh_y], 2)
    image = tf.expand_dims(image, 0)
    images = tf.tile(image, [batch_size, 1, 1, 1])
    return images

  def _CreateTestDepthImages(self, batch_size, height, width):
    """Creates a batch of test depth images of given size.

    Args:
      batch_size: Number of depth images to stack into a batch.
      height: Height of the depth image.
      width: Width of the depth image.

    Returns:
      depth_images: Tensor of shape [batch_size, height, width, 1]. In each
        depth image, depth value are uniformly sampled from 0.25 ~ 0.5.
    """
    tensor_shape = [batch_size, height, width, 1]
    depth_images = tf.random.uniform(tensor_shape, 0.25, 2.5)
    return depth_images

  @parameterized.parameters(([20, 20],), ([32, 32],))
  def testRandomCrop(self, output_shape):
    with tf.Graph().as_default():
      input_shape = [32, 32, 3]
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      cropped = image_transformations.RandomCropImages([images], input_shape,
                                                       output_shape)[0]
      with tf.Session() as sess:
        cropped_image = sess.run(cropped)
        self.assertAllEqual(cropped_image.shape,
                            [batch_size] + output_shape + [3])
        self.assertEqual(cropped_image[0, -1, 0, 1] - cropped_image[0, 0, 0, 1],
                         output_shape[0] - 1)
        self.assertEqual(cropped_image[0, 0, -1, 0] - cropped_image[0, 0, 0, 0],
                         output_shape[1] - 1)

  def testFaultyRandomCrop(self):
    with tf.Graph().as_default():
      input_shape = [32, 32, 3]
      output_shape = [20, 64]
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      with tf.Session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          cropped = image_transformations.RandomCropImages([images],
                                                           input_shape,
                                                           output_shape)[0]
          sess.run(cropped)

  def testWrongRandomCropImages(self):
    """Tests that all ValueErrors are triggered for RandomCropImages."""
    with tf.Graph().as_default():
      input_shape = [32, 32, 3]
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      with self.assertRaises(ValueError):
        # The input shape is (height, width) but (height, width, channels) is
        # required.
        image_transformations.RandomCropImages([images], [32, 32], [20, 64])

      with self.assertRaises(ValueError):
        # The input shape is (height, width, channel, random) but
        # (height, width, channels) is required.
        image_transformations.RandomCropImages([images], [32, 32, 3, 4],
                                               [20, 64])

      with self.assertRaises(ValueError):
        # The output shape is (height, ) but (height, width) is required.
        image_transformations.RandomCropImages([images], [32, 32, 3], [20])

      with self.assertRaises(ValueError):
        # The output shape is (height, width, random) but (height, width) is
        # required.
        image_transformations.RandomCropImages([images], [32, 32, 3],
                                               [20, 32, 64])

  def testWrongCenterCropImages(self):
    """Tests that all ValueErrors are triggered for CenterCropImages."""
    with tf.Graph().as_default():
      input_shape = [32, 32, 3]
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      with self.assertRaises(ValueError):
        # The input shape is (height, width) but (height, width, channels) is
        # required.
        image_transformations.CenterCropImages([images], [32, 32], [20, 64])

      with self.assertRaises(ValueError):
        # The input shape is (height, width, channel, random) but
        # (height, width, channels) is required.
        image_transformations.CenterCropImages([images], [32, 32, 3, 4],
                                               [20, 64])

      with self.assertRaises(ValueError):
        # The output shape is (height, ) but (height, width) is required.
        image_transformations.CenterCropImages([images], [32, 32, 3], [20])

      with self.assertRaises(ValueError):
        # The output shape is (height, width, random) but (height, width) is
        # required.
        image_transformations.CenterCropImages([images], [32, 32, 3],
                                               [20, 32, 64])

  @parameterized.parameters(([32, 32], [20, 20]), ([512, 640], [472, 472]))
  def testCenterCrop(self, input_shape, output_shape):
    input_shape = input_shape + [3]
    with tf.Graph().as_default():
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      cropped = image_transformations.CenterCropImages([images], input_shape,
                                                       output_shape)[0]
      with tf.Session() as sess:
        cropped_image = sess.run(cropped)
        # Check cropped shape.
        self.assertAllEqual(cropped_image.shape,
                            [batch_size] + output_shape + [3])
        # Check top-left corner on G-channel (y-coordinates).
        self.assertEqual(cropped_image[0, 0, 0, 1],
                         (input_shape[0] - output_shape[0]) // 2)
        # Check bottom-left corner on G-channel (y-coordinates).
        self.assertEqual(cropped_image[0, -1, 0, 1],
                         (input_shape[0] - output_shape[0]) // 2 +
                         output_shape[0] - 1)
        # Check top-left corner on R-channel (x-coordinates).
        self.assertEqual(cropped_image[0, 0, 0, 0],
                         (input_shape[1] - output_shape[1]) // 2)
        # Check bottom-left corner on R-channel (x-coordinates).
        self.assertEqual(cropped_image[0, 0, -1, 0],
                         (input_shape[1] - output_shape[1]) // 2 +
                         output_shape[1] - 1)

  @parameterized.parameters(([20, 20],), ([32, 32],))
  def testPhotometricImageDistortions(self, input_shape):
    input_shape = input_shape + [3]
    with tf.Graph().as_default():
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      tensor_list = []
      for i in range(batch_size):
        tensor_list.append(images[i])
      distorted = image_transformations.ApplyPhotometricImageDistortions(
          tensor_list, random_noise_apply_probability=1.0)
      delta = tf.reduce_sum(tf.square(images - distorted))
      with tf.Session() as sess:
        images_delta = sess.run(delta)
        # Check if any distortion applied.
        self.assertGreater(images_delta, 0)

  @parameterized.parameters(([20, 20],), ([32, 32],))
  def testPhotometricImageDistortionsParallel(self, input_shape):
    input_shape = input_shape + [3]
    with tf.Graph().as_default():
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      distorted = image_transformations.ApplyPhotometricImageDistortionsParallel(
          images, random_noise_apply_probability=1.0)
      delta = tf.reduce_sum(tf.square(images - distorted))
      with tf.Session() as sess:
        images_delta = sess.run(delta)
        # Check if any distortion applied.
        self.assertGreater(images_delta, 0)

  @parameterized.parameters(([20, 20],), ([32, 32],))
  def testDepthImageDistortions(self, input_shape):
    input_shape = input_shape + [1]
    with tf.Graph().as_default():
      batch_size = 4
      depth_images = self._CreateTestDepthImages(batch_size, input_shape[0],
                                                 input_shape[1])
      tensor_list = []
      for i in range(batch_size):
        tensor_list.append(depth_images[i])
      distorted = image_transformations.ApplyDepthImageDistortions(
          tensor_list, random_noise_apply_probability=1.0)
      depth_delta = tf.reduce_sum(tf.square(depth_images - distorted))
      with tf.Session() as sess:
        depth_images_delta = sess.run(depth_delta)
        # Check if any distortion applied.
        self.assertGreater(depth_images_delta, 0)

  @parameterized.parameters(([20, 20],), ([32, 32],))
  def testCustomCrop(self, target_shape):
    with tf.Graph().as_default():
      input_shape = [32, 32, 3]
      batch_size = 4
      target_locations = tf.tile(tf.constant([[10, 10]]), [batch_size, 1])
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      cropped = image_transformations.CustomCropImages([images],
                                                       input_shape,
                                                       target_shape,
                                                       [target_locations])[0]
      with tf.Session() as sess:
        cropped_image = sess.run(cropped)
        self.assertAllEqual(cropped_image.shape,
                            [batch_size] + target_shape + [3])
        self.assertEqual(cropped_image[0, -1, 0, 1] - cropped_image[0, 0, 0, 1],
                         target_shape[0] - 1)
        self.assertEqual(cropped_image[0, 0, -1, 0] - cropped_image[0, 0, 0, 0],
                         target_shape[1] - 1)

  @parameterized.parameters(([32, 32, 3], [53, 8], [10, 10]))
  def testFaultyCustomCrop(self, input_shape, target_shape, target_location):
    """Test that wrong crop parameters lead to failure."""
    with tf.Graph().as_default():
      batch_size = 4
      images = self._CreateRampTestImages(batch_size, input_shape[0],
                                          input_shape[1])
      target_locations = tf.constant(target_location)
      target_locations = tf.tile(tf.expand_dims(target_locations, 0),
                                 [batch_size, 1])

      with tf.Session() as sess:
        with self.assertRaises(ValueError):
          cropped = image_transformations.CustomCropImages(
              [images], input_shape, target_shape, [target_locations])[0]
          sess.run(cropped)


if __name__ == '__main__':
  tf.test.main()
