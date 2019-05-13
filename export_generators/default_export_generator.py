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

"""Utilities for generating SavedModel exports based on AbstractT2RModels."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import copy
import gin

from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata
import tensorflow as tf
from typing import Optional, Dict, Text, Any

MODE = tf.estimator.ModeKeys.PREDICT


@gin.configurable
class DefaultExportGenerator(abstract_export_generator.AbstractExportGenerator):
  """Class to manage assets related to exporting a model.

  Args:
    export_raw_receivers: Whether to export receiver_fns which do not have
      preprocessing enabled. This is useful for serving using Servo, in
      conjunction with client-preprocessing.
  """

  def create_serving_input_receiver_numpy_fn(self, params=None):
    """Create a serving input receiver for numpy.

    Args:
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      serving_input_receiver_fn: A callable which creates the serving inputs.
    """
    del params

    def serving_input_receiver_fn():
      """Create the ServingInputReceiver to export a saved model.

      Returns:
        An instance of ServingInputReceiver.
      """
      # We have to filter our specs since only required tensors are
      # used for inference time.
      flat_feature_spec = tensorspec_utils.flatten_spec_structure(
          self._get_input_features_for_receiver_fn())
      required_feature_spec = (
          tensorspec_utils.filter_required_flat_tensor_spec(flat_feature_spec))
      receiver_tensors = tensorspec_utils.make_placeholders(
          required_feature_spec)

      # We want to ensure that our feature processing pipeline operates on a
      # copy of the features and does not alter the receiver_tensors.
      features = tensorspec_utils.flatten_spec_structure(
          copy.copy(receiver_tensors))

      if (not self._export_raw_receivers and self._preprocess_fn is not None):
        features, _ = self._preprocess_fn(features=features, labels=None)

      return tf.estimator.export.ServingInputReceiver(features,
                                                      receiver_tensors)

    return serving_input_receiver_fn

  def create_serving_input_receiver_tf_example_fn(
      self, params = None):
    """Create a serving input receiver for tf_examples.

    Args:
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      serving_input_receiver_fn: A callable which creates the serving inputs.
    """
    del params

    def serving_input_receiver_fn():
      """Create the ServingInputReceiver to export a saved model.

      Returns:
        An instance of ServingInputReceiver.
      """
      # We only assume one input, a string which containes the serialized proto.
      receiver_tensors = {
          'input_example_tensor':
              tf.placeholder(
                  dtype=tf.string, shape=[None], name='input_example_tensor')
      }
      feature_spec = self._get_input_features_for_receiver_fn()
      # We have to filter our specs since only required tensors are
      # used for inference time.
      flat_feature_spec = tensorspec_utils.flatten_spec_structure(feature_spec)
      required_feature_spec = (
          tensorspec_utils.filter_required_flat_tensor_spec(flat_feature_spec))

      tensor_dict, tensor_spec_dict = (
          tensorspec_utils.tensorspec_to_feature_dict(required_feature_spec))

      parse_tf_example_fn = tfdata.create_parse_tf_example_fn(
          tensor_dict=tensor_dict,
          tensor_spec_dict=tensor_spec_dict,
          feature_tspec=feature_spec)

      features = parse_tf_example_fn(receiver_tensors['input_example_tensor'])

      if (not self._export_raw_receivers and self._preprocess_fn is not None):
        features, _ = self._preprocess_fn(features=features, labels=None)

      return tf.estimator.export.ServingInputReceiver(features,
                                                      receiver_tensors)

    return serving_input_receiver_fn
