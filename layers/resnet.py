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

"""ResNet tower.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import tensorflow as tf
from tensorflow_models.official.resnet import resnet_model as resnet_lib


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
               resnet_size, choices.keys()))
    raise ValueError(err)


def _model_output(inputs, data_format):
  """Maybe convert from channels_first (NCHW) back to channels_last (NHWC)."""
  if data_format == 'channels_first':
    return tf.transpose(a=inputs, perm=[0, 2, 3, 1])
  else:
    return inputs


def resnet_endpoints(model):
  """Extract intermediate values from ResNet model."""
  graph = tf.get_default_graph()
  scope = tf.get_variable_scope().name
  if scope:
    scope += '/'
  prefix = 'resnet_model'
  end_points = {}
  tensors = ['initial_conv', 'initial_max_pool', 'pre_final_pool',
             'final_reduce_mean', 'final_dense']
  tensors += [
      'block_layer{}'.format(i + 1) for i in range(len(model.block_sizes))]
  for name in tensors:
    tensor = graph.get_tensor_by_name(
        '{}{}/{}:0'.format(scope, prefix, name))
    if len(tensor.shape) == 4:
      tensor = _model_output(tensor, model.data_format)
    end_points[name] = tensor
  return end_points


@gin.configurable
def resnet_model(images,
                 is_training,
                 num_classes,
                 resnet_size=50,
                 return_intermediate_values=False):
  """Returns resnet model, optionally returning intermediate endpoint tensors.

  Args:
    images: A Tensor representing a batch [N,H,W,C] of input images.
    is_training: A boolean. Set to True to add operations required only when
      training the classifier.
    num_classes: Dimensionality of output logits emitted by final dense layer.
    resnet_size: Size of resnet. One of [18, 34, 50, 101, 152, 200].
    return_intermediate_values: If True, returns a dictionary of output and
      intermediate activation values.
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
      num_filters=64,
      kernel_size=7,
      conv_stride=2,
      first_pool_size=3,
      first_pool_stride=2,
      block_sizes=_get_block_sizes(resnet_size),
      block_strides=[1, 2, 2, 2],
      resnet_version=resnet_lib.DEFAULT_VERSION,
      data_format='channels_last',
      dtype=resnet_lib.DEFAULT_DTYPE
  )
  final_dense = model(images, is_training)
  if return_intermediate_values:
    return resnet_endpoints(model)
  else:
    return final_dense
