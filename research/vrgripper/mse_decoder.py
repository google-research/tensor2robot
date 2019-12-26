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

"""Abstract decoder and MSE decoder.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin

import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim


@gin.configurable
class MSEDecoder(object):
  """Default MSE decoder."""

  def __call__(self, params, output_size):
    self._predictions = slim.fully_connected(
        params, output_size, activation_fn=None, scope='pose')
    return self._predictions

  def loss(self, labels):
    return tf.losses.mean_squared_error(labels=labels.action,
                                        predictions=self._predictions)
