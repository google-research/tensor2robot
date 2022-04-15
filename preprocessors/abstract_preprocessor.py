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

"""The abstract preprocessor, handling boilerplate validation."""

import abc
from typing import Any, Callable, Optional, Tuple

import six
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator

ModeKeys = tf_estimator.ModeKeys


class AbstractPreprocessor(six.with_metaclass(abc.ABCMeta, object)):
  """A per example preprocessing function executed prior to the model_fn.

  Note, our preprocessor is invoked for a batch of features and labels.
  If the _preprocess_fn can only operate on batch_size one please use
  tf.map_fn as described in _preprocessor_fn.
  """

  def __init__(
      self,
      model_feature_specification_fn = None,
      model_label_specification_fn = None,
      is_model_device_tpu = False):
    """Initialize an instance.

    The provided specifications are used both for the in and out specification.
    The _preprocess_fn will not alter the provided tensors.

    Args:
      model_feature_specification_fn: (Optional) A function which takes mode as
        an argument and returns a valid spec structure for the features,
        preferablely a (hierarchical) namedtuple of TensorSpecs and
        OptionalTensorSpecs.
      model_label_specification_fn: (Optional) A function which takes mode as an
        argument and returns a valid spec structure for the labels, preferably a
        (hierarchical) namedtupel of TensorSpecs and OptionalTensorSpecs.
      is_model_device_tpu: True if the model is operating on TPU and otherwise
        False. This information is useful to do type conversions and strip
        unnecessary information from preprocessing since no summaries are
        generated on TPUs.
    """
    for spec_generator in [
        model_feature_specification_fn, model_label_specification_fn
    ]:
      for estimator_mode in [ModeKeys.TRAIN, ModeKeys.PREDICT, ModeKeys.EVAL]:
        if spec_generator:
          tensorspec_utils.assert_valid_spec_structure(
              spec_generator(estimator_mode))

    self._model_feature_specification_fn = model_feature_specification_fn
    self._model_label_specification_fn = model_label_specification_fn
    self._is_model_device_tpu = is_model_device_tpu

  @property
  def model_feature_specification_fn(self):
    return self._model_feature_specification_fn

  @model_feature_specification_fn.setter
  def model_feature_specification_fn(self, model_feature_specification_fn):
    self._model_feature_specification_fn = model_feature_specification_fn

  @property
  def model_label_specification_fn(self):
    return self._model_label_specification_fn

  @model_label_specification_fn.setter
  def model_label_specification_fn(self, model_label_specification_fn):
    self._model_label_specification_fn = model_label_specification_fn

  @abc.abstractmethod
  def get_in_feature_specification(
      self, mode):
    """The specification for the input features for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """

  @abc.abstractmethod
  def get_in_label_specification(
      self, mode):
    """The specification for the input labels for the preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """

  @abc.abstractmethod
  def get_out_feature_specification(
      self, mode):
    """The specification for the output features after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """

  @abc.abstractmethod
  def get_out_label_specification(
      self, mode):
    """The specification for the output labels after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors.
    """

  @abc.abstractmethod
  def _preprocess_fn(
      self, features,
      labels, mode
  ):
    """The preprocessing function which will be executed prior to the model_fn.

    Note, _preprocess_fn is invoked for a batch of features and labels.
    If the _preprocess_fn can only operate on batch_size one please use
    the following pattern.

    def _fn(features_single_batch, labels_single_batch):
      # The actual implementation

    return = tf.map_fn(
      _fn, # The single batch implementation
      (features, labels), # Our nested structure, the first dimension unpacked
      dtype=(self.get_out_feature_specification(),
             self.get_out_labels_specification()),
      back_prop=False,
      parallel_iterations=self._parallel_iterations)

    Args:
      features: The input features extracted from a single example in our
        in_feature_specification format.
      labels: (Optional None) The input labels extracted from a single example
        in our in_label_specification format.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      features_preprocessed: The preprocessed features, potentially adding
        additional tensors derived from the input features.
      labels_preprocessed: (Optional) The preprocessed labels, potentially
        adding additional tensors derived from the input features and labels.
    """

  def preprocess(
      self, features,
      labels, mode
  ):
    """The function which preprocesses the features and labels per example.

    Note, this function performs the boilerplate packing and flattening and
    verification of the features and labels according to our spec. The actual
    preprocessing is performed by _preprocess_fn.

    Args:
      features: The features of a single example.
      labels: (Optional None) The labels of a single example.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      features_preprocessed: The preprocessed and flattened features
        verified to fulfill our output specs.
      labels_preprocessed: (Optional None) The preprocessed and flattened labels
        verified to fulfill our output specs.
    """
    # First, we verify that the input features and labels fulfill our spec.
    # We further pack the flattened features and labels to our (hierarchical)
    # specification.:
    features = tensorspec_utils.validate_and_pack(
        expected_spec=self.get_in_feature_specification(mode),
        actual_tensors_or_spec=features,
        ignore_batch=True)
    if labels is not None:
      labels = tensorspec_utils.validate_and_pack(
          expected_spec=self.get_in_label_specification(mode),
          actual_tensors_or_spec=labels,
          ignore_batch=True)

    features_preprocessed, labels_preprocessed = self._preprocess_fn(
        features=features, labels=labels, mode=mode)

    features_preprocessed = tensorspec_utils.validate_and_flatten(
        expected_spec=self.get_out_feature_specification(mode),
        actual_tensors_or_spec=features_preprocessed,
        ignore_batch=True)
    if labels_preprocessed:
      labels_preprocessed = tensorspec_utils.validate_and_flatten(
          expected_spec=self.get_out_label_specification(mode),
          actual_tensors_or_spec=labels_preprocessed,
          ignore_batch=True)
    return features_preprocessed, labels_preprocessed
