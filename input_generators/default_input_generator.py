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

"""Default input generators wrapping tfdata, metadata and replay buffer."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
from absl import logging
import gin
from tensor2robot.input_generators import abstract_input_generator
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata

import tensorflow as tf

from typing import Dict, Optional, Text



@gin.configurable
class DefaultRecordInputGenerator(
    abstract_input_generator.AbstractInputGenerator):
  """A recordio, sstable or tfrecords based input generator."""

  def __init__(self,
               file_patterns = None,
               dataset_map = None,
               label = '',
               **parent_kwargs):
    """Create an instance.

    Args:
      file_patterns: Comma-separated string of file patterns [recordio, sstable,
        tfrecord] we will load the data from. Only one of `file_patterns` or
        `dataset_map` should be set.
      dataset_map: Dictionary of dataset_key and Comma-separated string of file
        patterns we load data from. Used for loading from multiple datasets.
        Only one of `file_patterns` or `dataset_map` should be set.
      label: Name of the input generator.
      **parent_kwargs: All parent arguments.
    """
    super(DefaultRecordInputGenerator, self).__init__(**parent_kwargs)
    if file_patterns and dataset_map:
      raise ValueError(
          'Only one of `file_patterns` or `dataset_map` should be set.')
    self._file_patterns = file_patterns
    self._dataset_map = dataset_map
    self._label = label

  def create_dataset_input_fn(self, mode):
    """Create the dataset input_fn used for train and eval.

    We simply wrap the existing tfdata implementation such that we have a
    consistent input generator interface.

    Args:
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      A valid input_fn for the estimator api.
    """
    self._assert_specs_initialized()
    logging.info('Creating InputGenerator %s with file patterns:\n%s',
                 self._label, self._file_patterns)
    input_fn = tfdata.get_input_fn(
        file_patterns=self._file_patterns or self._dataset_map,
        batch_size=self.batch_size,
        feature_spec=self._feature_spec,
        label_spec=self._label_spec,
        mode=mode,
        preprocess_fn=self._preprocess_fn)


    return input_fn

  def _create_dataset(self, **unused_kwargs):
    """This abstract function is not required for default input generators.

    Since we directly create the input_fn from default implementations we do not
    need to implement this method.
    """


class GeneratorInputGenerator(abstract_input_generator.AbstractInputGenerator):
  """Class to use for constructing input generators from Python generator objects."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, sequence_length=None, **kwargs):
    self._sequence_length = sequence_length
    super(GeneratorInputGenerator, self).__init__(**kwargs)

  @abc.abstractmethod
  def _generator_fn(self, batch_size):
    """Subclasses override this generator.

    Args:
      batch_size: batch size to generate examples

    Yields:
      training examples
    """

  def _create_dataset(self, mode, params=None):
    """Creates a dataset using _generator_fn.

    Args:
      mode: prediction mode
      params: Hyperparameters sent by estimator object.

    Returns:
      A tfdata.Dataset object.
    """
    used_batch_size = tfdata.get_batch_size(params, self._batch_size)

    def shape_transform(x):
      if isinstance(x, tensorspec_utils.ExtendedTensorSpec) and x.is_sequence:
        return [used_batch_size, self._sequence_length] + x.shape.as_list()
      else:
        return [used_batch_size] + x.shape.as_list()

    feature_dtypes = tf.contrib.framework.nest.map_structure(lambda x: x.dtype,
                                           self._feature_spec)
    feature_shapes = tf.contrib.framework.nest.map_structure(shape_transform, self._feature_spec)
    label_dtypes = tf.contrib.framework.nest.map_structure(lambda x: x.dtype, self._label_spec)
    label_shapes = tf.contrib.framework.nest.map_structure(shape_transform, self._label_spec)

    dataset = tf.data.Dataset.from_generator(
        lambda: self._generator_fn(used_batch_size),
        (feature_dtypes, label_dtypes), (feature_shapes, label_shapes))
    if self._preprocess_fn is not None:
      dataset = dataset.map(self._preprocess_fn)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


@gin.configurable
class DefaultRandomInputGenerator(GeneratorInputGenerator):
  """An input generator generating random inputs."""

  def _generator_fn(self, batch_size):
    while True:
      features = tensorspec_utils.make_random_numpy(
          self._feature_spec, batch_size, self._sequence_length)
      labels = tensorspec_utils.make_random_numpy(self._label_spec, batch_size,
                                                  self._sequence_length)
      yield features, labels


@gin.configurable
class DefaultConstantInputGenerator(GeneratorInputGenerator):
  """An input generator generating constant inputs."""

  def __init__(self, constant_value, **kwargs):
    self._constant_value = constant_value
    super(DefaultConstantInputGenerator, self).__init__(**kwargs)

  def _generator_fn(self, batch_size):
    while True:
      features = tensorspec_utils.make_constant_numpy(
          self._feature_spec, self._constant_value, batch_size,
          self._sequence_length)
      labels = tensorspec_utils.make_constant_numpy(
          self._label_spec, self._constant_value, batch_size,
          self._sequence_length)
      yield features, labels
