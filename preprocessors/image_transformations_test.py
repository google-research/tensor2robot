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

"""Tests for tensor2robot image_transformations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensor2robot.preprocessors import image_transformations
import tensorflow as tf  # tf


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
      cropped = image_transformations.RandomCropImages([images], input_shape,
                                                       output_shape)[0]
      with tf.Session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
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


if __name__ == '__main__':
  tf.test.main()
