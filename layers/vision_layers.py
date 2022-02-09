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

# Lint as python3
"""Implements image-to-pose regression model from WTL paper.

Colloquially referred to as 'Berkeley-Net'.
"""
import gin
from six.moves import range
from tensor2robot.layers import spatial_softmax
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


@gin.configurable
def BuildImagesToFeaturesModel(images,
                               filter_size=3,
                               num_blocks=5,
                               num_output_maps=32,
                               is_training=False,
                               normalizer_fn=slim.layer_norm,
                               normalizer_params=None,
                               weight_regularization=0.00001,
                               film_output_params=None,
                               use_spatial_softmax=True):
  """Builds the pose regression model.

  Args:
    images: A float32 Tensor with shape [batch_size, height, width, channels]
      representing the camera image. Its values range from 0.0 to 1.0.
    filter_size: The width and height of the conv filters.
    num_blocks: The number of pool-conv-conv_1x1 blocks to repeat.
    num_output_maps: Number of output feature maps.
    is_training: True if training.
    normalizer_fn: Function to use for normalization. Defaults to layer norm.
    normalizer_params: Dictionary of normalizer_fn parameters for batch_norm.
    weight_regularization: Weight regularization factor.
    film_output_params: If given, parse gamma and beta from given Tensor and
      scale feature maps as done in FILM (https://arxiv.org/abs/1709.07871). As
        recommended by the paper, instead of doing gamma * x + beta, this does
        (1 + gamma) * x + beta, to better handle the initial zero-centered
        gamma.
    use_spatial_softmax: If True, adds a spatial softmax layer on top of the
      final convnet features.

  Returns:
    expected_feature_points: If spatial_softmax is True, a tensor of size
      [batch_size, num_output_maps * 2]. These are the expected feature
      locations, i.e., the spatial softmax of feature_maps. The inner
      dimension is arranged as [x1, x2, x3 ... xN, y1, y2, y3, ... yN].

      If spatial_softmax is False, a tensor of size [batch_size, height, width,
      num_output_maps], where height / width will depend on the input size.
    extra: A dict containing the softmax if use_spatial_softmax is True,
      otherwise an empty dict.
  """

  if normalizer_params is None and normalizer_fn == slim.batch_norm:
    normalizer_params = {
        'is_training': is_training,
        'decay': 0.99,
        'scale': False,
        'epsilon': 0.0001,
    }
    batch_norm_params_with_scaling = {
        'is_training': is_training,
        'decay': 0.99,
        'scale': True,
        'epsilon': 0.0001,
    }
  else:
    batch_norm_params_with_scaling = None

  # Number of channels for each layer in intermediate conv layers.
  # Not configurable at the moment.
  num_channels_per_block = 32

  if film_output_params is not None:
    # Retrieves the gammas and betas for FILM.
    # Given an input z we wish to condition on, FILM learns an
    #   f_i(z) = gamma_i, beta_i
    # for each layer i we want to condition, then does.
    #   FILM(h_i) = gamma_i * h_i + beta_i
    # where h_i is the pre-activation of the network (right before ReLU).
    # This assume we are given a Tensor that's the concat of all gammas and
    # betas, and we want to condition each conv layer of the network.
    expected_size = 2 * num_blocks * num_channels_per_block
    # I bet there's a better way to assert this.
    film_shape = film_output_params.get_shape().as_list()
    if len(film_shape) != 2:
      raise ValueError('FILM shape is %s but is expected to be 2-D' %
                       str(film_shape))
    if film_shape[-1] != expected_size:
      raise ValueError('FILM shape is %s but final dimension should be %d' %
                       (str(film_shape), expected_size))

    # [batch, film_size] -> [batch, 1, 1, film_size] for broadcasting
    film_output_params = tf.expand_dims(film_output_params, axis=-2)
    film_output_params = tf.expand_dims(film_output_params, axis=-2)
    gammas_and_betas = tf.split(
        film_output_params, num_or_size_splits=2 * num_blocks, axis=-1)
    gammas, betas = gammas_and_betas[:num_blocks], gammas_and_betas[num_blocks:]
    for i in range(num_blocks):
      gammas[i] = 1.0 + gammas[i]

  net = images

  with slim.arg_scope([slim.conv2d], padding='VALID'):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(weight_regularization),
        biases_initializer=tf.constant_initializer(0.01),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params):
      for i in range(num_blocks):
        if i == 0 or i == 1:
          stride = 2
        else:
          stride = 1

        # Conv -> BN -> FILM -> ReLU.
        net = slim.conv2d(
            net,
            num_outputs=num_channels_per_block,
            activation_fn=None,
            kernel_size=[filter_size, filter_size],
            stride=stride,
            scope='conv{:d}'.format(i + 2))

        if film_output_params is not None:
          net = gammas[i] * net + betas[i]
        net = tf.nn.relu(net)

      net = slim.conv2d(
          net,
          num_output_maps, [1, 1],
          scope='final_conv_1x1',
          normalizer_params=batch_norm_params_with_scaling)
      if use_spatial_softmax:
        net, softmax = spatial_softmax.BuildSpatialSoftmax(net)
        return net, {'softmax': softmax}
      else:
        return net, {}


@gin.configurable
def BuildFILMParams(embedding, film_output_size=2 * 5 * 32):
  """Builds FILM output params from input embedding.

  This is just a linear layer, no activation function.

  Args:
    embedding: A rank 2 tensor: [N, embedding_size]
    film_output_size: The number of parameters to output for FILM. Should equal
      2 * total_number_of_channels across all conv layers we want to apply FILM
      to.

  Returns:
    A rank 2 tensor [N, film_output_size].
  """
  return slim.fully_connected(
      embedding,
      film_output_size,
      scope='film',
      normalizer_fn=None,
      activation_fn=None)


@gin.configurable
def BuildImagesToFeaturesModelHighRes(images,
                                      filter_size=3,
                                      num_blocks=5,
                                      num_output_maps=32,
                                      is_training=False,
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params=None,
                                      weight_regularization=0.00001):
  """Builds the pose regression model.

  Note: this is a variant of the above, used in the PI-GPS paper (Chebotar et
  al., 2016). We call it "HighRes" because it adds up features from multiple
  layers at different resolutions by scaling everything up, and the spatial
  softmax is computed at the highest of those resolutions. See
  https://arxiv.org/pdf/1610.00529.pdf for an architecture diagram.

  Args:
    images: A float32 Tensor with shape [batch_size, height, width, channels]
      representing the camera image. Its values range from 0.0 to 1.0.
    filter_size: The width and height of the conv filters.
    num_blocks: The number of pool-conv-conv_1x1 blocks to repeat.
    num_output_maps: Number of output feature maps.
    is_training: True if training.
    normalizer_fn: Function to use for normalization. Defaults to batch norm.
    normalizer_params: Dictionary of normalizer_fn parameters.
    weight_regularization: Weight regularization factor.

  Returns:
    expected_feature_points: A tensor of size
      [batch_size, num_features * 2]. These are the expected feature
      locations, i.e., the spatial softmax of feature_maps. The inner
      dimension is arranged as [x1, x2, x3 ... xN, y1, y2, y3, ... yN].
  """
  # Parameters for batch normalization.
  batch_norm_params_with_scaling = None
  if normalizer_fn == slim.batch_norm:
    if normalizer_params is None:
      normalizer_params = {
          'is_training': is_training,
          'decay': 0.99,
          'scale': False,
          'epsilon': 0.0001,
      }
    batch_norm_params_with_scaling = {
        'is_training': is_training,
        'decay': 0.99,
        'scale': True,
        'epsilon': 0.0001,
    }

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d], padding='VALID'):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_regularizer=slim.l2_regularizer(weight_regularization),
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params):
      block_outs = []
      net = slim.avg_pool2d(images, [2, 2], stride=2, scope='pool1')
      net = slim.conv2d(
          net, 16, [filter_size, filter_size], stride=2, scope='conv1')
      net = slim.conv2d(
          net, 32, [filter_size, filter_size], stride=1, scope='conv2')
      block_outs.append(slim.conv2d(net, 32, [1, 1], scope='conv2_1x1'))
      for i in range(1, num_blocks):
        net = slim.max_pool2d(
            net, [2, 2], stride=2, scope='pool{:d}'.format(i + 1))
        net = slim.conv2d(
            net,
            32, [filter_size, filter_size],
            stride=1,
            scope='conv{:d}'.format(i + 2))
        block_outs.append(
            slim.conv2d(net, 32, [1, 1], scope='conv{:d}_1x1'.format(i + 2)))
      final_image_shape = block_outs[0].get_shape().as_list()[1:3]

      def ResizeLayerToImage(layer):
        return tf.image.resize_images(
            layer, [final_image_shape[0], final_image_shape[1]],
            tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      net = tf.add_n([ResizeLayerToImage(layer) for layer in block_outs])
      net = slim.conv2d(
          net,
          num_output_maps, [1, 1],
          scope='final_conv_1x1',
          normalizer_params=batch_norm_params_with_scaling)
      net, softmax = spatial_softmax.BuildSpatialSoftmax(net)
      return net, {'softmax': softmax}


@gin.configurable
def BuildImageFeaturesToPoseModel(expected_feature_points,
                                  num_outputs,
                                  aux_input=None,
                                  aux_output_dim=0,
                                  hidden_dim=100,
                                  num_layers=2,
                                  is_training=True,
                                  normalizer_fn=slim.layer_norm,
                                  bias_transform_size=10):
  """Build a model to predict pose from image features.

  This is currently a single fully connected layer from feature points to the
  pose.

  Args:
    expected_feature_points: A tensor of size [batch_size, num_features * 2].
      These are the expected feature locations, i.e., the spatial softmax of
      feature_maps. The inner dimension is arranged as [x1, x2, x3 ... xN, y1,
      y2, y3, ... yN].
    num_outputs: The dimensionality of the output vector. If None, returns the
      last hidden layer.
    aux_input: auxiliary inputs, such as robot configuration
    aux_output_dim: dimension of auxiliary predictions to make, eg button pose
    hidden_dim: dimensionality of the fully connected hidden layers.
    num_layers: number of fully connected hidden layers.
    is_training: True if training the model, False otherwise.
    normalizer_fn: Function to use for normalization. Cannot be batch norm.
    bias_transform_size: Size of the bias transform variable vector.

  Returns:
    A tensor of size [batch_size, num_outputs] representing the output vector
    (predicted pose), and either None or a tensor of size
    [batch_size, aux_output_dim] if aux_output_dim > 0
  """
  del is_training  # Unused.
  if aux_input is not None:
    # (batch*40, 64), (batch*40, 7)
    net = tf.concat([expected_feature_points, aux_input], 1)
  else:
    net = expected_feature_points
  bias_init = tf.constant_initializer(0.01)

  with slim.arg_scope(
      [slim.fully_connected],
      biases_initializer=bias_init,
      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):

    # Add bias transformation variable.
    if bias_transform_size > 0:
      bt = tf.zeros([tf.shape(net)[0], bias_transform_size])
      bt = slim.bias_add(bt, initializer=bias_init)
      net = tf.concat([net, bt], 1)

    for layer_index in range(num_layers):
      net = slim.fully_connected(
          net,
          hidden_dim,
          scope='pose_fc{:d}'.format(layer_index),
          normalizer_fn=normalizer_fn)
    if num_outputs:
      net = slim.fully_connected(
          net,
          num_outputs,
          activation_fn=None,
          scope='pose_fc{:d}'.format(num_layers))
    if aux_output_dim > 0:
      aux_output = slim.fully_connected(
          expected_feature_points,
          aux_output_dim,
          activation_fn=None,
          scope='pose_fc_aux')
      return net, aux_output
    else:
      return net, None
