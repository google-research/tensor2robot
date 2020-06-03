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

"""A simple no operation preprocessor."""

from typing import Optional, Tuple

import gin
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf


@gin.configurable
class NoOpPreprocessor(abstract_preprocessor.AbstractPreprocessor):
  """A convenience preprocessor which does not perform any preprocessing.

  This prerpocessor provides convenience functionality in case we simply want
  to ensure that already single examples contain the right information for our
  model. This preprocessor does not perform any preprocessing, but allows
  existing models to initialize a preprocessor without any additional runtime
  overhead.
  """

  def get_in_feature_specification(
      self, mode):
    """The specification for the input features for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification
    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))

  def get_in_label_specification(
      self, mode):
    """The specification for the input labels for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification
    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def get_out_feature_specification(
      self, mode):
    """The specification for the output features after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification
    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))

  def get_out_label_specification(
      self, mode):
    """The specification for the output labels after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification
    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def _preprocess_fn(
      self, features,
      labels,
      mode
  ):
    """The preprocessing function which will be executed prior to the fn.

    As the name NoOpPreprocessor suggests, we do not perform any prerprocessing.

    Args:
      features: The input features extracted from a single example in our
        in_features_specification format.
      labels: (Optional None) The input labels extracted from a single example
        in our in_features_specification format.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      features: The preprocessed features, potentially adding
        additional tensors derived from the input features.
      labels: (Optional) The preprocessed labels, potentially
        adding additional tensors derived from the input features and labels.
    """
    return features, labels
