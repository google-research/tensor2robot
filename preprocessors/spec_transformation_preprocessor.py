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

"""A convenience preprocessor which allows to easily transform specs."""

from typing import Text

from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf


class SpecTransformationPreprocessor(
    abstract_preprocessor.AbstractPreprocessor):
  """A convenience abstract preprocessor which allows to replace tensors.

  It is a common requirement for preprocessors to e.g. convert data types
  for images from tf.float32 to tf.uint8. For instance, models should only
  operate on tpu ready tensor types such as [tf.float32, tf.int32, tf.int64,
  tf.bfloat16,tf.string, and tf.bool]. However, images are typically stored in
  tf.uint8 format. Another common operation is changing the shape of the
  feature, for example, due to cropping. Many specs specified in models do not
  require any modification, therefore, the preprocessor simply has to request
  them and not alter them.

  This abstract preprocessor allows to perform these transformations very easily
  by merely updating the existing specs provided by the model. It reduces the
  requirement for boilerplate code to implement default model specific
  preprocessors. However, we encourage the implementation of "full"
  preprocessors for new model which specify the in as well as the out
  specifications completely in the preprocessor.
  """

  def update_spec(self, tensor_spec_struct,
                  key, **kwargs_for_tensorspec):
    """Helper function to allow to alter a specific tensorspec in the structure.

    Args:
      tensor_spec_struct: A TensorSpecStruct describing the required and
        optional tensors supporting both dictionary and hierarchical attribute
        like access.
      key: The key we use to index tensor_spec_struct.
      **kwargs_for_tensorspec: The {key: value} argumentes for
        ExtendedTensorSpec.from_spec to overwrite the existing spec.

    Returns:
      The updated tensor_spec_struct.
    """
    tensor_spec_struct[key] = tensorspec_utils.ExtendedTensorSpec.from_spec(
        spec=tensor_spec_struct[key], **kwargs_for_tensorspec)

  def get_in_feature_specification(
      self, mode):
    """The specification for the input features before executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors supporting
      both dictionary and hierarchical attribute like access.
    """
    # We transform the tensorspec into it's flat structure.
    # This allows us to easily replace specs and replace tensors during
    # preprocessing.
    # Note, this will not alter with the spec itself, it merely allows the
    # user of this function to easily apply the proper preprocessing.
    # Further, flattening the spec structure will make a copy, allowing
    # us to safely operate on the in_feature_specification without
    # altering a implicit state or changing the model_feature_specification.
    tensor_spec_struct = tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))
    return self._transform_in_feature_specification(tensor_spec_struct)

  def _transform_in_feature_specification(
      self, tensor_spec_struct
  ):
    """The specification for the input features for the preprocess_fn.

    Here we will transform the feature spec to represent the requirements
    for preprocessing.

    Args:
      tensor_spec_struct: A TensorSpecStruct describing the required and
        optional tensors supporting both dictionary and hierarchical attribute
        like access.

    Returns:
      tensor_spec_struct: The transformed TensorSpecStruct describing the
        required and optional tensors supporting both dictionary and
        hierarchical attribute like access.
    """
    return tensor_spec_struct

  def get_in_label_specification(
      self, mode):
    """The specification for the input labels before executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors supporting
      both dictionary and hierarchical attribute like access.
    """
    # We transform the tensorspec into it's flat structure.
    # This allows us to easily replace specs and replace tensors during
    # preprocessing.
    # Note, this will not alter with the spec itself, it merely allows the
    # user of this function to easily apply the proper preprocessing.
    # Further, flattening the spec structure will make a copy, allowing
    # us to safely operate on the in_label_specification without
    # altering a implicit state or changing the model_label_specification.
    tensor_spec_struct = tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))
    return self._transform_in_label_specification(tensor_spec_struct)

  def _transform_in_label_specification(
      self, tensor_spec_struct
  ):
    """The specification for the input labels for the preprocess_fn.

    Here we will transform the feature spec to represent the requirements
    for preprocessing.

    Args:
      tensor_spec_struct: A TensorSpecStruct describing the required and
        optional tensors supporting both dictionary and hierarchical attribute
        like access.

    Returns:
      tensor_spec_struct: The transformed TensorSpecStruct describing the
        required and optional tensors supporting both dictionary and
        hierarchical attribute like access.
    """
    return tensor_spec_struct

  def get_out_feature_specification(
      self, mode):
    """The specification for the output features after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors supporting
      both dictionary and hierarchical attribute like access.
    """
    return self._model_feature_specification_fn(mode)

  def get_out_label_specification(
      self, mode):
    """The specification for the output labels after executing preprocess_fn.

    Arguments:
      mode: mode key for this feature specification

    Returns:
      A TensorSpecStruct describing the required and optional tensors supporting
      both dictionary and hierarchical attribute like access.
    """
    return self._model_label_specification_fn(mode)
