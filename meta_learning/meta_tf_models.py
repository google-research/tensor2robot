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

# Lint as python3
"""Meta-Learning TFModels.

A standard TFModel encapsulates an `InputGenerator`, a `Preprocessor`, and a
`Model`. To turn this into a Meta-learning equivalent, we introduce a
MetaTFModel which encapsulates a `MetaInputGenerator`, a `MetaPreprocessor`, and
a `MetaModel`. The `Meta`-classes wrap instances of base classes and
allow us to turn models into their meta-learning equivalents.

"""

from typing import Optional, Text

from absl import logging
import gin
import numpy as np
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils as utils
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import framework as contrib_framework

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

framework = contrib_framework
nest = framework.nest
# pylint: disable=invalid-name
TensorSpec = utils.ExtendedTensorSpec
TrainValPair = meta_tfdata.TrainValPair
# pylint: enable=invalid-name


def select_mode(val_mode, train, val):
  # nest.map_structure requires `native` python dicts as input since it uses
  # the c interface.
  val_dict = utils.flatten_spec_structure(val).to_dict()
  train_dict = utils.flatten_spec_structure(train).to_dict()
  select_mode_fn = lambda train, val: tf.where(val_mode, x=val, y=train)
  return utils.TensorSpecStruct(
      list(nest.map_structure(select_mode_fn, train_dict, val_dict).items()))


def _create_meta_spec(
    tensor_spec, spec_type, num_train_samples_per_task,
    num_val_samples_per_task):
  """Create a TrainValPair from an existing spec.

  Note, the train spec will maintain the same name and thus mapping to the
  input. This is important to create the parse_tf_example_fn automatically.
  The validation data will have a val/ prefix such that we can feed different
  data to both inputs.

  Args:
    tensor_spec: A dict, (named)tuple, list or a hierarchy thereof filled by
      TensorSpecs(subclasses) or Tensors.
    spec_type: A string ['features', 'labels'] specifying which spec type we
      alter in order to introduce the corresponding val_mode.
    num_train_samples_per_task: Number of training samples to expect per task
      batch element.
    num_val_samples_per_task: Number of val examples to expect per task batch
      element.

  Raises:
    ValueError: If the spec_type is not in ['features', 'labels'].

  Returns:
    An instance of TensorSpecStruct representing a valid
    meta learning tensor_spec with .train and .val access.
  """
  if spec_type not in ['features', 'labels']:
    raise ValueError('We only support spec_type "features" or "labels" '
                     'but received {}.'.format(spec_type))
  train_tensor_spec = utils.flatten_spec_structure(
      utils.copy_tensorspec(
          tensor_spec, batch_size=num_train_samples_per_task, prefix='train'))
  # Since the train part is also required for inference, the specs cannot be
  # optional.
  for key, value in train_tensor_spec.items():
    train_tensor_spec[key] = utils.ExtendedTensorSpec.from_spec(
        value, is_optional=False)

  val_tensor_spec = utils.flatten_spec_structure(
      utils.copy_tensorspec(
          tensor_spec, batch_size=num_val_samples_per_task, prefix='val'))

  # Since the train part is also required for inference, the specs for
  # val cannot be optional because the inputs to a while loop have to be
  # the same for every step of the loop.
  for key, value in val_tensor_spec.items():
    val_tensor_spec[key] = utils.ExtendedTensorSpec.from_spec(
        value, is_optional=False)
  val_mode_shape = (1,)
  if num_train_samples_per_task is None:
    val_mode_shape = ()
  val_mode = TensorSpec(
      shape=val_mode_shape, dtype=tf.bool, name='val_mode/{}'.format(spec_type))
  return utils.flatten_spec_structure(
      TrainValPair(
          train=train_tensor_spec, val=val_tensor_spec, val_mode=val_mode))


@gin.configurable
class MetaPreprocessor(abstract_preprocessor.AbstractPreprocessor):
  """Initial version of a MetaPreprocessor."""

  def __init__(self, base_preprocessor, num_train_samples_per_task,
               num_val_samples_per_task):
    """Construct a Meta-learning feature/label Preprocessor.

    Used in conjunction with MetaInputGenerator. Takes a normal preprocessor's
      expected inputs and wraps its outputs into TrainVal pairs for
      meta-learning algorithms.

    Args:
      base_preprocessor: An instance of the base preprocessor class to convert
        into TrainValPairs.
      num_train_samples_per_task: Number of training samples to expect per task
        batch element.
      num_val_samples_per_task: Number of val examples to expect per task batch
        element.
    """
    super(MetaPreprocessor, self).__init__()
    self._base_preprocessor = base_preprocessor
    self._num_train_samples_per_task = num_train_samples_per_task
    self._num_val_samples_per_task = num_val_samples_per_task

  @property
  def num_train_samples_per_task(self):
    logging.warning('This function is only used for the legacy input pipeline.'
                    'Please use the new input_generator.')
    return self._num_train_samples_per_task

  @property
  def num_val_samples_per_task(self):
    logging.warning('This function is only used for the legacy input pipeline.'
                    'Please use the new input_generator.')
    return self._num_val_samples_per_task

  @property
  def base_preprocessor(self):
    return self._base_preprocessor

  def get_in_feature_specification(
      self):
    """See parent class."""
    return _create_meta_spec(
        self._base_preprocessor.get_in_feature_specification(),
        spec_type='features',
        num_train_samples_per_task=self._num_train_samples_per_task,
        num_val_samples_per_task=self._num_val_samples_per_task)

  def get_in_label_specification(self):
    """See parent class."""
    return _create_meta_spec(
        self._base_preprocessor.get_in_label_specification(),
        spec_type='labels',
        num_train_samples_per_task=self._num_train_samples_per_task,
        num_val_samples_per_task=self._num_val_samples_per_task)

  def get_out_feature_specification(
      self):
    """See parent class."""
    return _create_meta_spec(
        self._base_preprocessor.get_out_feature_specification(),
        spec_type='features',
        num_train_samples_per_task=self._num_train_samples_per_task,
        num_val_samples_per_task=self._num_val_samples_per_task)

  def get_out_label_specification(self):
    """See parent class."""
    return _create_meta_spec(
        self._base_preprocessor.get_out_label_specification(),
        spec_type='labels',
        num_train_samples_per_task=self._num_train_samples_per_task,
        num_val_samples_per_task=self._num_val_samples_per_task)

  def _preprocess_fn(self, features,
                     labels,
                     mode):
    """See base class."""
    if mode is None:
      raise ValueError('The mode should never be None.')

    # In order to unflatten the flattened examples later, we need to keep
    # track of the original shapes.
    features = meta_tfdata.flatten_batch_examples(features)

    # The original preprocessor can only operate on the flattened data.
    if labels is not None:
      labels = meta_tfdata.flatten_batch_examples(labels)

      # We invoke our original preprocessor on the flat batch.
      features.train, labels.train = self._base_preprocessor.preprocess(
          features=features.train, labels=labels.train, mode=mode)
      features.val, labels.val = self._base_preprocessor.preprocess(
          features=features.val, labels=labels.val, mode=mode)
    else:
      # We invoke our original preprocessor on the flat batch.
      features.train, _ = self._base_preprocessor.preprocess(
          features=features.train, labels=None, mode=mode)
      features.val, _ = self._base_preprocessor.preprocess(
          features=features.val, labels=None, mode=mode)

    # We need to unflatten with num_*_samples_per_task since the preprocessor
    # might introduce new tensors or reshape existing tensors.
    features.train = meta_tfdata.unflatten_batch_examples(
        features.train, self._num_train_samples_per_task)
    features.val = meta_tfdata.unflatten_batch_examples(
        features.val, self._num_val_samples_per_task)
    features.val_mode = tf.reshape(features.val_mode, [-1, 1])
    if labels is not None:
      labels.train = meta_tfdata.unflatten_batch_examples(
          labels.train, self._num_train_samples_per_task)
      labels.val = meta_tfdata.unflatten_batch_examples(
          labels.val, self._num_val_samples_per_task)
      labels.val_mode = tf.reshape(labels.val_mode, [-1, 1])

    return features, labels


class MetalearningModel(abstract_model.AbstractT2RModel):
  """Base class for Meta-Learning Models (e.g. MAML).

  Meta-Learning models operate over pairs of training and validation
  batches, minimizing some loss `L_val(model.update(L_train))` w.r.t. model and/
  or `L_train`.

  Inherit from this class to implement a custom RL^2 model for a given task.
  """

  def __init__(self,
               base_model,
               num_train_samples_per_task,
               num_val_samples_per_task,
               preprocessor_cls=None,
               **kwargs):
    super(MetalearningModel, self).__init__(
        preprocessor_cls=preprocessor_cls,
        **kwargs)
    self._base_model = base_model
    self._num_train_samples_per_task = num_train_samples_per_task
    self._num_val_samples_per_task = num_val_samples_per_task

  @property
  def default_preprocessor_cls(self):
    return MetaPreprocessor

  @property
  def preprocessor(self):
    preprocessor_cls = self._preprocessor_cls
    if preprocessor_cls is None:
      preprocessor_cls = self.default_preprocessor_cls
    self._preprocessor = preprocessor_cls(
        self._base_model.preprocessor,
        num_train_samples_per_task=self._num_train_samples_per_task,
        num_val_samples_per_task=self._num_val_samples_per_task)
    return self._preprocessor

  def get_feature_specification(self):
    """See parent class."""
    num_train_samples_per_task = self._num_train_samples_per_task
    num_val_samples_per_task = self._num_val_samples_per_task
    return _create_meta_spec(
        self._base_model.get_feature_specification(),
        spec_type='features',
        num_train_samples_per_task=num_train_samples_per_task,
        num_val_samples_per_task=num_val_samples_per_task)

  def get_label_specification(self):
    """See parent class."""
    num_train_samples_per_task = self._num_train_samples_per_task
    num_val_samples_per_task = self._num_val_samples_per_task
    return _create_meta_spec(
        self._base_model.get_label_specification(),
        spec_type='labels',
        num_train_samples_per_task=num_train_samples_per_task,
        num_val_samples_per_task=num_val_samples_per_task)

  def _flatten_and_add_meta_dim(self, train_data, val_data, val_mode):
    """Pack data into a flattened TrainValPair and add an extra dimension.

    Args:
      train_data: (Spec structure of) train data with a single batch dimension.
      val_data: (Spec structure of) val data with a single batch dimension.
      val_mode: The binary data corresponding to val_mode in a TrainValPair.

    Returns:
      An TensorSpecStruct with train, val, and val_mode attrs. The
      data will have an extra batch dimension.
    """
    flat_data = utils.flatten_spec_structure(
        TrainValPair(train_data, val_data, val_mode))
    # We need to expand the additional dimension to match the same structure
    # as train/eval. Up until here we only have the data for one batch
    # of data and one task [batch, ...]. However, meta learning assumes
    # [num_tasks, num_samples_per_task, ...], therefore we add one additional
    # dimension.
    for key, value in flat_data.train.items():
      flat_data.train[key] = np.expand_dims(value, 0)
    for key, value in flat_data.val.items():
      flat_data.val[key] = np.expand_dims(value, 0)
    return flat_data
