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

"""ResNet tower.
"""

from typing import List, Optional, Text
from absl import logging
import gin
from six.moves import range
from tensor2robot.layers import film_resnet_model as resnet_lib
import tensorflow.compat.v1 as tf
import tensorflow as contrib_framework
import tf_slim as contrib_slim

slim = contrib_slim


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, list(choices.keys())))
    raise ValueError(err)


def _model_output(inputs, data_format):
  """Maybe convert from channels_first (NCHW) back to channels_last (NHWC)."""
  if data_format == 'channels_first':
    return tf.transpose(a=inputs, perm=[0, 2, 3, 1])
  else:
    return inputs


def _get_resnet_scope():
  scope = tf.get_default_graph().get_name_scope()
  if scope:
    scope += '/'
  return scope + 'resnet_model/'


def resnet_endpoints(model):
  """Extract intermediate values from ResNet model."""
  graph = tf.get_default_graph()
  scope = _get_resnet_scope()
  end_points = {}
  tensors = ['initial_conv', 'initial_max_pool', 'pre_final_pool',
             'final_reduce_mean', 'final_dense']
  tensors += [
      'block_layer{}'.format(i + 1) for i in range(len(model.block_sizes))]
  for name in tensors:
    tensor = graph.get_tensor_by_name('{}{}:0'.format(scope, name))
    if len(tensor.shape) == 4:
      tensor = _model_output(tensor, model.data_format)
    end_points[name] = tensor
  return end_points


@gin.configurable
def linear_film_generator(embedding,
                          block_sizes,
                          filter_sizes,
                          enabled_block_layers = None):
  """FiLM generator for all blocks in a ResNet.

  Args:
    embedding: Conditioning embedding. Passed in by ResNet model builder.
    block_sizes: Passed in by ResNet model builder. Number of blocks in each
      block layer.
    filter_sizes: Filter sizes for each block layer. Passed in by ResNet model
      builder.
    enabled_block_layers: Optional list specifying which block layers have
      conditioning. This can be gin-configured by the user.

  Returns:
    film_gamma_betas: List of lists of FiLM vectors for each block layer.
      film_gamma_betas[i][j] is the jth block of the ith block layer, and is
      either None or a [batch, 2*C] tensor,where C is the channel dimension of
      the ResNet activations it is modulating. As recommended by the paper,
      instead of doing gamma * x + beta, this does
      (1 + gamma) * x + beta, to better handle the initial zero-centered
      gamma.
  """
  if enabled_block_layers:
    if len(enabled_block_layers) != len(block_sizes):
      raise ValueError(
          'Got {} bools for enabled_block_layers, expected {}'.format(
              len(enabled_block_layers), len(block_sizes)))
  # FiLM generator - just a linear projection of embedding.
  film_gamma_betas = []
  for i, num_blocks in enumerate(block_sizes):
    if enabled_block_layers and not enabled_block_layers[i]:
      # Do not generate FiLM vectors for this block layer.
      film_gamma_betas.append([None]*num_blocks)
    else:
      num_filters = filter_sizes[i]
      film_output_size = num_blocks * num_filters * 2
      film_gamma_beta = slim.fully_connected(
          embedding,
          film_output_size,
          scope='film{}'.format(i),
          normalizer_fn=None,
          activation_fn=None)
      film_gamma_betas.append(tf.split(film_gamma_beta, num_blocks, axis=-1))
  return film_gamma_betas


@gin.configurable
def resnet_model(images,
                 is_training,
                 num_classes,
                 resnet_size=50,
                 weight_decay=None,
                 kernel_size=7,
                 num_filters=64,
                 return_intermediate_values=False,
                 film_generator_fn=None,
                 film_generator_input=None,
                 pretrain_checkpoint=None):
  """Returns resnet model, optionally returning intermediate endpoint tensors.

  Args:
    images: A Tensor representing a batch [N,H,W,C] of input images.
    is_training: A boolean. Set to True to add operations required only when
      training the classifier.
    num_classes: Dimensionality of output logits emitted by final dense layer.
    resnet_size: Size of resnet. One of [18, 34, 50, 101, 152, 200].
    weight_decay: L2 weight regularizer.
    kernel_size: Size of the convolution kernels used in the resnet model.
    num_filters: Number of filters used.
    return_intermediate_values: If True, returns a dictionary of output and
      intermediate activation values.
    film_generator_fn: Callable that returns a List (for each block layer) of
      Lists (per ResNet block) of FiLM conditioning vectors.
    film_generator_input: Embedding tensor to be passed to film_generator_fn.
    pretrain_checkpoint: String to initialize model weights from. Does *NOT*
      initialize final logits layer. ResNet checkpoints can be found here:
      https://github.com/tensorflow/models/tree/master/official/r1/resnet.
  """
  # For bigger models, we want to use "bottleneck" layers
  if resnet_size < 50:
    bottleneck = False
  else:
    bottleneck = True
  model = resnet_lib.Model(
      resnet_size=resnet_size,
      bottleneck=bottleneck,
      num_classes=num_classes,
      num_filters=num_filters,
      kernel_size=kernel_size,
      conv_stride=2,
      first_pool_size=3,
      first_pool_stride=2,
      block_sizes=_get_block_sizes(resnet_size),
      block_strides=[1, 2, 2, 2],
      resnet_version=resnet_lib.DEFAULT_VERSION,
      data_format='channels_last',
      weight_decay=weight_decay,
      dtype=resnet_lib.DEFAULT_DTYPE
  )
  final_dense = model(images, is_training,
                      film_generator_fn, film_generator_input)
  if pretrain_checkpoint:
    # Initialize variables in ResNet, excluding the final dense layer and any
    # optimization-specific variables (e.g. Momentum, Adam Beta).
    # When initializing on TPUs, use AbstractT2RModel.init_from_checkpoint_fn.
    resnet_init_from_checkpoint_fn(pretrain_checkpoint)
  if return_intermediate_values:
    return resnet_endpoints(model)
  else:
    return final_dense


@gin.configurable
def resnet_init_from_checkpoint_fn(checkpoint):
  """init_from_checkpoint_fn that can be used to init a model from a checkpoint.

  Args:
    checkpoint: String pointing to path of TF checkpoint.

  Raises:
    A ValueError if a variable(s) is missing and partial restore is not
    explicitly enabled.
  """
  logging.info('Initializing model weights from %s', checkpoint)
  assignment_map = {}
  resnet_scope = _get_resnet_scope()
  for var in contrib_framework.get_variables(
      scope=resnet_scope, collection=tf.GraphKeys.TRAINABLE_VARIABLES):
    if 'dense' not in var.op.name:
      # Remove the parent scope prefix.
      name_in_ckpt = var.op.name.replace(resnet_scope, 'resnet_model/')
      assignment_map[name_in_ckpt] = var
  tf.train.init_from_checkpoint(checkpoint, assignment_map)
