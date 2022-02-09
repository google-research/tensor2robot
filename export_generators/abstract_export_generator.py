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
"""Utilities for exporting savedmodels."""

import abc
import functools
import os
from typing import Any, Dict, List, Optional, Text

import gin
import six
from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf
from tensorflow.contrib import util as contrib_util

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

MODE = tf.estimator.ModeKeys.PREDICT


@gin.configurable
class AbstractExportGenerator(six.with_metaclass(abc.ABCMeta, object)):
  """Class to manage assets related to exporting a model.

  Attributes:
    export_raw_receivers: Whether to export receiver_fns which do not have
      preprocessing enabled. This is useful for serving using Servo, in
      conjunction with client-preprocessing.
  """

  def __init__(self, export_raw_receivers = False):
    self._export_raw_receivers = export_raw_receivers
    self._feature_spec = None
    self._out_feature_spec = None
    self._preprocess_fn = None
    self._model_name = None

  def set_specification_from_model(self,
                                   t2r_model):
    """Set the feature specifications and preprocess function from the model.

    Args:
      t2r_model: A T2R model instance.
    """
    preprocessor = t2r_model.preprocessor
    self._feature_spec = preprocessor.get_in_feature_specification(MODE)
    tensorspec_utils.assert_valid_spec_structure(self._feature_spec)
    self._out_feature_spec = (preprocessor.get_out_feature_specification(MODE))
    tensorspec_utils.assert_valid_spec_structure(self._out_feature_spec)
    self._preprocess_fn = functools.partial(preprocessor.preprocess, mode=MODE)
    self._model_name = type(t2r_model).__name__

  def _get_input_features_for_receiver_fn(self):
    """Helper function to return a input featurespec for reciver fns.

    Returns:
      The appropriate feature specification to use
    """
    if self._export_raw_receivers:
      return self._out_feature_spec
    else:
      return self._feature_spec

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  def create_warmup_requests_numpy(self, batch_sizes,
                                   export_dir):
    """Creates warm-up requests for a given feature specification.

    This writes an output file in
    `export_dir/assets.extra/tf_serving_warmup_requests` for use with Servo.

    Args:
      batch_sizes: Batch sizes of warm-up requests to write.
      export_dir: Base directory for the export.

    Returns:
      The filename written.
    """
    feature_spec = self._get_input_features_for_receiver_fn()

    flat_feature_spec = tensorspec_utils.flatten_spec_structure(feature_spec)
    tf.io.gfile.makedirs(export_dir)
    request_filename = os.path.join(export_dir, 'tf_serving_warmup_requests')
    with tf.python_io.TFRecordWriter(request_filename) as writer:
      for batch_size in batch_sizes:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self._model_name
        numpy_feature_specs = tensorspec_utils.make_constant_numpy(
            flat_feature_spec, constant_value=0, batch_size=batch_size)

        for key, numpy_spec in numpy_feature_specs.items():
          request.inputs[key].CopyFrom(
              contrib_util.make_tensor_proto(numpy_spec))

        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())
    return request_filename
