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

# Lint as python3
"""TensorFlow impl of Spatial Softmax layers. (spatial soft arg-max).

TODO(T2R_CONTRIBUTORS) - consider replacing with contrib version.
"""

import gin
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


@gin.configurable
def BuildSpatialSoftmax(features, spatial_gumbel_softmax=False):
  """Computes the spatial softmax of the input features.

  Args:
    features: A tensor of size [batch_size, num_rows, num_cols, num_features]
    spatial_gumbel_softmax: If set to True, samples locations stochastically
      rather than computing expected coordinates with respect to heatmap.
  Returns:
    A tuple of (expected_feature_points, softmax).
    expected_feature_points: A tensor of size
      [batch_size, num_features * 2]. These are the expected feature
      locations, i.e., the spatial softmax of feature_maps. The inner
      dimension is arranged as [x1, x2, x3 ... xN, y1, y2, y3, ... yN].
    softmax: A Tensor which is the softmax of the features.
      [batch_size, num_rows, num_cols, num_features].
  """
  _, num_rows, num_cols, num_features = features.get_shape().as_list()

  with tf.name_scope('SpatialSoftmax'):
    # Create tensors for x and y positions, respectively
    x_pos = np.empty([num_rows, num_cols], np.float32)
    y_pos = np.empty([num_rows, num_cols], np.float32)

    # Assign values to positions
    for i in range(num_rows):
      for j in range(num_cols):
        x_pos[i, j] = 2.0 * j / (num_cols - 1.0) - 1.0
        y_pos[i, j] = 2.0 * i / (num_rows - 1.0) - 1.0

    x_pos = tf.reshape(x_pos, [num_rows * num_cols])
    y_pos = tf.reshape(y_pos, [num_rows * num_cols])

    # We reorder the features (norm3) into the following order:
    # [batch_size, NUM_FEATURES, num_rows, num_cols]
    # This lets us merge the batch_size and num_features dimensions, in order
    # to compute spatial softmax as a single batch operation.
    features = tf.reshape(
        tf.transpose(features, [0, 3, 1, 2]), [-1, num_rows * num_cols])

    if spatial_gumbel_softmax:
      # Temperature is hard-coded for now, make this more flexible if results
      # are promising.
      dist = tfp.distributions.RelaxedOneHotCategorical(
          temperature=1.0, logits=features)
      softmax = dist.sample()
    else:
      softmax = tf.nn.softmax(features)
    # Element-wise multiplication
    x_output = tf.multiply(x_pos, softmax)
    y_output = tf.multiply(y_pos, softmax)
    # Sum per out_size x out_size
    x_output = tf.reduce_sum(x_output, [1], keep_dims=True)
    y_output = tf.reduce_sum(y_output, [1], keep_dims=True)
    # Concatenate x and y, and reshape.
    expected_feature_points = tf.reshape(
        tf.concat([x_output, y_output], 1), [-1, num_features*2])
    softmax = tf.transpose(
        tf.reshape(softmax, [-1, num_features, num_rows,
                             num_cols]), [0, 2, 3, 1])
    return expected_feature_points, softmax
