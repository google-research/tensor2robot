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

"""The abstract base class for input generators."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import functools
import inspect
import gin
from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from typing import Callable, Optional, Text, Tuple, Union


@gin.configurable
class AbstractInputGenerator(object):
  """The abstract input generator responsible for creating the input pipeline.

  The main functionality for exporting models both for serialized tf.Example
  protos and numpy feed_dict's is implemented in a general way in this abstract
  class. The dataset pipeline used for training has to be overwritten in
  respective subclasses.
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self, batch_size = 32):
    """Create an instance.

    Args:
      batch_size: (Optional) This determines the batch size for each feature and
        label produced by the input pipeline.
    """
    self._feature_spec = None
    self._label_spec = None
    # The function used within tf.data.Dataset.map to preprocess the data which
    # has the mode set by the preprocessor.
    self._preprocess_fn = None
    self._batch_size = batch_size
    self._out_feature_spec = None
    self._out_label_spec = None

  @property
  def batch_size(self):
    return self._batch_size


  def set_feature_specifications(self, feature_spec, out_feature_spec):
    tensorspec_utils.assert_valid_spec_structure(feature_spec)
    tensorspec_utils.assert_valid_spec_structure(out_feature_spec)
    self._feature_spec = feature_spec
    self._out_feature_spec = out_feature_spec

  def set_label_specifications(self, label_spec, out_label_spec):
    tensorspec_utils.assert_valid_spec_structure(label_spec)
    tensorspec_utils.assert_valid_spec_structure(out_label_spec)
    self._label_spec = label_spec
    self._out_label_spec = out_label_spec

  def set_specification_from_model(self,
                                   t2r_model,
                                   mode):
    """Get all specifications to create and verify an input pipeline.

    Args:
      t2r_model: A T2RModel from which we get all necessary feature
        and label specifications.
      mode: A tf.estimator.ModelKeys object that specifies the mode for
        specification.
    """
    preprocessor = t2r_model.preprocessor
    self._feature_spec = preprocessor.get_in_feature_specification(mode)
    tensorspec_utils.assert_valid_spec_structure(self._feature_spec)
    self._label_spec = preprocessor.get_in_label_specification(mode)
    tensorspec_utils.assert_valid_spec_structure(self._label_spec)
    # It is necessary to verify that the output of the dataset inputs fulfill
    # our specification.
    self._out_feature_spec = (preprocessor.get_out_feature_specification(mode))
    tensorspec_utils.assert_valid_spec_structure(self._out_feature_spec)
    self._out_label_spec = (preprocessor.get_out_label_specification(mode))
    tensorspec_utils.assert_valid_spec_structure(self._out_label_spec)
    self._preprocess_fn = functools.partial(preprocessor.preprocess, mode=mode)

  def set_preprocess_fn(self, preprocess_fn):  # pytype: disable=invalid-annotation
    """Register the preprocess_fn used during the input data generation.

    Note, the preprocess_fn can only have `features` and optionally `labels` as
    inputs. The `mode` has to be abstracted by using a closure or
    functools.partial prior to passing a preprocessor.preprocess function.
    For example using functools:
    set_preprocess_fn(
      functools.partial(preprocessor.preprocess,
                        mode=tf.estimator.ModeKeys.TRAIN))

    Args:
      preprocess_fn: The function called during the input dataset generation to
        preprocess the data.
    """
    if isinstance(preprocess_fn, functools.partial):  # pytype: disable=wrong-arg-types
      # Note, we do not combine both conditions into one since
      # inspect.getargspec does not work for functools.partial objects.
      if 'mode' not in preprocess_fn.keywords:
        raise ValueError('The preprocess_fn mode has to be set if a partial'
                         'function has been passed.')
    elif 'mode' in inspect.getargspec(preprocess_fn).args:
      raise ValueError('The passed preprocess_fn has an open argument `mode`'
                       'which should be patched by a closure or with '
                       'functools.partial.')

    self._preprocess_fn = preprocess_fn

  def create_dataset_input_fn(self, mode):
    """Create the dataset input_fn used for train and eval.

    Args:
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      A valid input_fn for the estimator api.
    """
    self._assert_specs_initialized()
    self._assert_out_specs_initialized()

    def input_fn(params=None):
      """The input_fn callable which is used within tf.Estimator.

      Args:
        params: An optional dict of hyper parameters that will be passed into
          input_fn and model_fn. Keys are names of parameters, values are basic
          python types. There are reserved keys for TPUEstimator, including
          'batch_size'.

      Returns:
        features: All features according to our
          preprocessor.get_out_feature_specification().
        labels: All labels according to our
          preprocessor.get_out_label_specification().
      """
      return self._create_dataset(mode=mode, params=params)

    return input_fn

  @abc.abstractmethod
  def _create_dataset(self, mode, params=None):
    """The actual implementation to create the dataset input_fn.

    Args:
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      A valid input_fn for the estimator api.
    """

  def _assert_specs_initialized(self):
    """Ensure that all specs are initialized.

    Raises:
      ValueError: If either label_spec or feature_spec have not been set.
    """
    if self._label_spec is None:
      raise ValueError('No label spec set, please parameterize the input '
                       'generator using set_specification_from_model.')
    if self._feature_spec is None:
      raise ValueError('No label spec set, please parameterize the input '
                       'generator using set_specification_from_model.')

  def _assert_out_specs_initialized(self):
    """Ensure that all specs are initialized.

    Raises:
      ValueError: If either label_spec or feature_spec have not been set.
    """
    if self._out_label_spec is None:
      raise ValueError('No out label spec set, please parameterize the input '
                       'generator using set_specification_from_model.')
    if self._out_feature_spec is None:
      raise ValueError('No out label spec set, please parameterize the input '
                       'generator using set_specification_from_model.')
