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

# Lint as: python2, python3
"""Specialized preprocessors for meta learning e.g. for MAML."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Optional, Text, Tuple

import gin
import six
from six.moves import range
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils as utils
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest
TSpecStructure = utils.TensorSpecStruct


def create_maml_feature_spec(feature_spec,
                             label_spec):
  """Create a meta feature from existing base_model specs.

  Note, the train spec will maintain the same name and thus mapping to the
  input. This is important to create the parse_tf_example_fn automatically.
  The validation data will have a val/ prefix such that we can feed different
  data to both inputs.

  Args:
    feature_spec: A hierarchy of TensorSpecs(subclasses) or Tensors.
    label_spec: A hierarchy of TensorSpecs(subclasses) or Tensors.

  Returns:
    An instance of TensorSpecStruct representing a valid
    meta learning tensor_spec with .condition and .inference access.
  """
  condition_spec = TSpecStructure()
  condition_spec.features = utils.flatten_spec_structure(
      utils.copy_tensorspec(
          feature_spec, batch_size=-1, prefix='condition_features'))
  condition_spec.labels = utils.flatten_spec_structure(
      utils.copy_tensorspec(
          label_spec, batch_size=-1, prefix='condition_labels'))
  inference_spec = TSpecStructure()
  inference_spec.features = utils.flatten_spec_structure(
      utils.copy_tensorspec(
          feature_spec, batch_size=-1, prefix='inference_features'))

  meta_feature_spec = TSpecStructure()
  meta_feature_spec.condition = condition_spec
  meta_feature_spec.inference = inference_spec
  return meta_feature_spec


def create_maml_label_spec(label_spec):
  """Create a meta feature from existing base_model specs.

  Args:
    label_spec: A hierarchy of TensorSpecs(subclasses) or Tensors.

  Returns:
    An instance of TensorSpecStruct representing a valid
    meta learning tensor_spec for computing the outer loss.
  """
  return utils.flatten_spec_structure(
      utils.copy_tensorspec(label_spec, batch_size=-1, prefix='meta_labels'))


@gin.configurable
class MAMLPreprocessorV2(abstract_preprocessor.AbstractPreprocessor):
  """Initial version of a MetaPreprocessor."""

  def __init__(self,
               base_preprocessor):
    """Construct a Meta-learning feature/label Preprocessor.

    Used in conjunction with MetaInputGenerator. Takes a normal preprocessor's
      expected inputs and wraps its outputs into meta TensorSpecs for
      meta-learning algorithms.

    Args:
      base_preprocessor: An instance of the base preprocessor class to convert
        into TrainValPairs.
    """
    super(MAMLPreprocessorV2, self).__init__()
    self._base_preprocessor = base_preprocessor

  @property
  def base_preprocessor(self):
    return self._base_preprocessor

  def get_in_feature_specification(
      self, mode):
    """See parent class."""
    return create_maml_feature_spec(
        self._base_preprocessor.get_in_feature_specification(mode),
        self._base_preprocessor.get_in_label_specification(mode))

  def get_in_label_specification(self,
                                 mode):
    """See parent class."""
    return create_maml_label_spec(
        self._base_preprocessor.get_in_label_specification(mode))

  def get_out_feature_specification(
      self, mode):
    """See parent class."""
    return create_maml_feature_spec(
        self._base_preprocessor.get_out_feature_specification(mode),
        self._base_preprocessor.get_out_label_specification(mode))

  def get_out_label_specification(
      self, mode):
    """See parent class."""
    return create_maml_label_spec(
        self._base_preprocessor.get_out_label_specification(mode))

  def create_meta_map_fn(self,
                         num_condition_samples_per_task,
                         num_inference_samples_per_task):
    """Creates a map function to construct meta features/labels.

    Args:
      num_condition_samples_per_task: The number of samples for conditioning per
        batch.
      num_inference_samples_per_task: The number of samples for evaluation per
        batch.

    Raises:
      ValueError: If num_*_samples_per_task is None or not greater than 0.

    Returns:
      A callable map function which takes a batch of features and labels
      defined by the base_preprocessor and groups them into meta features and
      meta labels.
    """
    if (num_condition_samples_per_task is None or
        num_condition_samples_per_task <= 0):
      raise ValueError(
          'num_condition_samples_per_task cannot be None and has '
          'to be positve but is {}.'.format(num_condition_samples_per_task))
    if (num_inference_samples_per_task is None or
        num_inference_samples_per_task <= 0):
      raise ValueError(
          'num_inference_samples_per_task cannot be None and has '
          'to be positve but is {}.'.format(num_inference_samples_per_task))

    ref_batch_size = (
        num_condition_samples_per_task + num_inference_samples_per_task)

    def _verify_batch_size(tensor):
      """Verifies that every tensor has a valid batch_size.

      Args:
        tensor: The tensor for which the batch_size is checked.

      Returns:
        tensor: The unmodified tensor passed to the function.

      Raises:
        ValueError: In case that the batch_size is not known at graph
          construction time or it does not match the ref_batch_size determined
          in the enclosing function.
      """
      batch_size = tensor.get_shape().as_list()[0]
      if batch_size is None:
        raise ValueError('The batch_size has to be known at graph construction '
                         'time but is None.')
      if batch_size != ref_batch_size:
        raise ValueError(
            'The batch_size has to be '
            'num_condition_samples_per_task + '
            'num_inference_samples_per_task = {} but is {}.'.format(
                ref_batch_size, batch_size))
      return tensor

    def _split_batch_into_condition(tensor):
      return tensor[:num_condition_samples_per_task]

    def _split_batch_into_inference(tensor):
      return tensor[num_condition_samples_per_task:]

    def map_fn(features, labels
              ):
      """Creates a mapping from base_preprocessor specs to meta specs.

      Args:
        features: Features according to the spec structure of the
          base_preprocessor.
        labels: Labels according to the spec structure of the base_preprocessor.

      Returns:
        meta_features: The regrouped features according to the meta feature
          spec.
        meta_labels: The regrouped labels according to the meta label spec.
      """
      # Now that we know all samples have the right batch size we can separate
      # them into data for conditioning and evaluation.
      features = nest.map_structure(_verify_batch_size, features)
      labels = nest.map_structure(_verify_batch_size, labels)

      condition = TSpecStructure()
      inference = TSpecStructure()
      condition.features = utils.flatten_spec_structure(
          nest.map_structure(_split_batch_into_condition, features))
      inference.features = utils.flatten_spec_structure(
          nest.map_structure(_split_batch_into_inference, features))
      condition.labels = utils.flatten_spec_structure(
          nest.map_structure(_split_batch_into_condition, labels))
      meta_labels = utils.flatten_spec_structure(
          nest.map_structure(_split_batch_into_inference, labels))
      meta_features = TSpecStructure()
      meta_features.condition = condition
      meta_features.inference = inference
      return meta_features, meta_labels

    return map_fn

  def _preprocess_fn(self, features,
                     labels,
                     mode
                    ):
    """Flattens inner and sequence dimensions."""
    if mode is None:
      raise ValueError('The mode should never be None.')

    condition_feature = list(features.condition.features.values())[0]
    inference_feature = list(features.inference.features.values())[0]
    # In order to unflatten the flattened examples later, we need to keep
    # track of the original shapes.
    num_condition_samples_per_task = (
        condition_feature.get_shape().as_list()[1])
    num_inference_samples_per_task = (
        inference_feature.get_shape().as_list()[1])

    if num_condition_samples_per_task is None:
      raise ValueError('num_condition_samples_per_task cannot be None.')
    if num_inference_samples_per_task is None:
      raise ValueError('num_inference_samples_per_task cannot be None.')

    flat_features = meta_tfdata.flatten_batch_examples(features)

    flat_labels = None
    # The original preprocessor can only operate on the flattened data.
    if labels is not None:
      flat_labels = meta_tfdata.flatten_batch_examples(labels)

    # We invoke our original preprocessor on the flat batch.
    flat_features.condition.features, flat_features.condition.labels = (
        self._base_preprocessor.preprocess(
            features=flat_features.condition.features,
            labels=flat_features.condition.labels,
            mode=mode))
    (flat_features.inference.features,
     flat_labels) = self._base_preprocessor.preprocess(
         features=flat_features.inference.features,
         labels=flat_labels,
         mode=mode)

    # We need to unflatten with num_*_samples_per_task since the preprocessor
    # might introduce new tensors or reshape existing tensors.
    features.condition = meta_tfdata.unflatten_batch_examples(
        flat_features.condition, num_condition_samples_per_task)
    features.inference = meta_tfdata.unflatten_batch_examples(
        flat_features.inference, num_inference_samples_per_task)
    if flat_labels is not None:
      labels = meta_tfdata.unflatten_batch_examples(
          flat_labels, num_inference_samples_per_task)

    return features, labels


def create_metaexample_spec(
    model_spec,
    num_samples_per_task,
    prefix):
  """Converts a model feature/label spec into a MetaExample spec.

  Args:
    model_spec: The base model tensor spec.
    num_samples_per_task: Number of episodes in the task.
    prefix: The tf.Example feature column name prefix.
  Returns:
    A TSpecStructure. For each spec in model_spec, the output contains
    num_samples_per_task corresponding specs stored as: "<name>/i".
  """
  model_spec = utils.flatten_spec_structure(model_spec)
  meta_example_spec = TSpecStructure()

  for key in model_spec.keys():
    for i in range(num_samples_per_task):
      spec = model_spec[key]
      name_prefix = '{:s}_ep{:d}'.format(prefix, i)
      new_name = name_prefix + '/' + six.ensure_str(spec.name)
      meta_example_spec[key + '/{:}'.format(i)] = (
          utils.ExtendedTensorSpec.from_spec(
              spec, name=new_name))
  return meta_example_spec


def stack_intra_task_episodes(
    in_tensors,
    num_samples_per_task,
):
  """Stacks together tensors from different episodes of the same task.

  Args:
    in_tensors: The input tensors, stored with key names of the form
      "<name>/i", where i is an int in [0, (num_samples_per_task - 1)].
    num_samples_per_task: Number of episodes in the task.

  Returns:
    A structure of tensors that matches out_tensor_spec.
  """
  out_tensors = TSpecStructure()
  # Strip the "/i" postfix from all keys, then get the set of unique keys.
  key_set = set(['/'.join(key.split('/')[:-1]) for key in in_tensors.keys()])
  for key in key_set:
    data = []
    for i in range(num_samples_per_task):
      data.append(in_tensors['{:s}/{:d}'.format(key, i)])
    out_tensors[key] = tf.stack(data, axis=1)
  return out_tensors


@gin.configurable
class FixedLenMetaExamplePreprocessor(MAMLPreprocessorV2):
  """Preprocesses MetaExamples into MetaLearning features/labels.

  This is a simpler version of MetaExamplePreprocessor that only supports
  FixedLenFeatures.
  """

  def __init__(
      self,
      base_preprocessor,
      num_condition_samples_per_task = 1,
      num_inference_samples_per_task = 1):
    """Initialize the MetaExamplePreprocessor.

    Args:
      base_preprocessor: An instance of the base preprocessor class to convert
        into TrainValPairs.
      num_condition_samples_per_task: Number of condition episodes per task.
      num_inference_samples_per_task: Number of inference episodes per task.
    """
    self._num_condition_samples_per_task = num_condition_samples_per_task
    self._num_inference_samples_per_task = num_inference_samples_per_task
    super(FixedLenMetaExamplePreprocessor, self).__init__(base_preprocessor)

  @property
  def num_condition_samples_per_task(self):
    return self._num_condition_samples_per_task

  @property
  def num_inference_samples_per_task(self):
    return self._num_inference_samples_per_task

  def get_in_feature_specification(self, mode):
    # Tempting to use create_maml_feature_spec, but we don't want the
    # prefixes or the meta-batch dimension.
    condition_spec = TSpecStructure()
    condition_spec.features = (
        self._base_preprocessor.get_in_feature_specification(mode))
    condition_spec.labels = (
        self._base_preprocessor.get_in_label_specification(mode))
    inference_spec = TSpecStructure()
    inference_spec.features = (
        self._base_preprocessor.get_in_feature_specification(mode))
    feature_spec = TSpecStructure()
    feature_spec.condition = create_metaexample_spec(
        condition_spec, self._num_condition_samples_per_task, 'condition')
    feature_spec.inference = create_metaexample_spec(
        inference_spec, self._num_inference_samples_per_task, 'inference')
    return utils.flatten_spec_structure(feature_spec)

  def get_in_label_specification(self, mode):
    label_spec = create_metaexample_spec(
        self._base_preprocessor.get_in_label_specification(mode),
        self._num_inference_samples_per_task, 'inference')
    return utils.flatten_spec_structure(label_spec)

  def _preprocess_fn(
      self,
      features,
      labels,
      mode = None):
    out_features = TSpecStructure()
    out_features.condition = stack_intra_task_episodes(
        features.condition, self._num_condition_samples_per_task)
    out_features.inference = stack_intra_task_episodes(
        features.inference, self._num_inference_samples_per_task)

    out_labels = None
    if labels is not None:
      out_labels = stack_intra_task_episodes(
          labels, self._num_inference_samples_per_task)
    return super(FixedLenMetaExamplePreprocessor, self)._preprocess_fn(
        out_features, out_labels, mode)
