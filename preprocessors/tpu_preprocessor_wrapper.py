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

"""A preprocessor which wraps preprocessors for TPU usage.

This intended for usage as part of TPUT2RModelWrapper, which will automatically
wrap preprocessors for TPU usage.
"""

from typing import Optional, Tuple

import gin
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf

ModeKeys = tf.estimator.ModeKeys


@gin.configurable
class TPUPreprocessorWrapper(abstract_preprocessor.AbstractPreprocessor):
  """A preprocessor to create action images."""

  def __init__(self, preprocessor):
    """Creates a class instance.

    Note, additional arguments which would be passed to the parent class
    are on purpose not provided in order to ensure that all configuration
    is actually done in the wrapped preprocessor.

    Args:
      preprocessor: The preprocessor which will be wrapped.
    """
    super(TPUPreprocessorWrapper, self).__init__()
    self._preprocessor = preprocessor

  @property
  def preprocessor(self):
    """Returns the wrapped preprocessor."""
    return self._preprocessor

  def _filter_using_spec(self,
                         tensor_spec_struct,
                         output_spec
                        ):
    """Filters all optional tensors from the tensor_spec_struct.

    Args:
      tensor_spec_struct: The instance of TensorSpecStruct which contains the
        preprocessing tensors.
      output_spec: The reference TensorSpecStruct which allows to infer which
        tensors should be removed.

    Returns:
      A new instance which contains only required tensors.
    """
    filtered_spec_struct = tensorspec_utils.TensorSpecStruct()
    for key in output_spec.keys():
      filtered_spec_struct[key] = tensor_spec_struct[key]
    return filtered_spec_struct

  def get_in_feature_specification(
      self, mode):
    """The specification for the input features for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.replace_dtype(
        self._preprocessor.get_in_feature_specification(mode),
        from_dtype=tf.bfloat16,
        to_dtype=tf.float32)

  def get_in_label_specification(
      self, mode):
    """The specification for the input labels for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.replace_dtype(
        self._preprocessor.get_in_label_specification(mode),
        from_dtype=tf.bfloat16,
        to_dtype=tf.float32)

  def get_out_feature_specification(
      self, mode):
    """The specification for the output features after executing preprocess_fn.

    Note, we strip all optional specs to further reduce communication and
    computation overhead for feeding to TPUs.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.replace_dtype(
        tensorspec_utils.filter_required_flat_tensor_spec(
            self._preprocessor.get_out_feature_specification(mode)),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)

  def get_out_label_specification(
      self, mode):
    """The specification for the output labels after executing preprocess_fn.

    Note, we strip all optional specs to further reduce communication and
    computation overhead for feeding to TPUs.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.replace_dtype(
        tensorspec_utils.filter_required_flat_tensor_spec(
            self._preprocessor.get_out_label_specification(mode)),
        from_dtype=tf.float32,
        to_dtype=tf.bfloat16)

  def _preprocess_fn(
      self, features,
      labels, mode
  ):
    features, labels = self._preprocessor._preprocess_fn(features, labels, mode)  # pylint: disable=protected-access

    out_feature_spec = self.get_out_feature_specification(mode)
    features = self._filter_using_spec(features, out_feature_spec)
    features = tensorspec_utils.cast_float32_to_bfloat16(
        features, out_feature_spec)
    if labels is not None:
      out_label_spec = self.get_out_label_specification(mode)
      labels = self._filter_using_spec(labels, out_label_spec)
      labels = tensorspec_utils.cast_float32_to_bfloat16(labels, out_label_spec)
    return features, labels
