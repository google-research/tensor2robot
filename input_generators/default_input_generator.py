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
"""Default input generators wrapping tfdata, metadata and replay buffer."""

import abc
import json
import os
from typing import Dict, Optional, Text, Union, Sequence

import gin
import numpy as np
import six
from tensor2robot.input_generators import abstract_input_generator
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

_TF_CONFIG_ENV = 'TF_CONFIG'
_MULTI_EVAL_NAME = 'multi_eval_name'


def _get_tf_config_env():
  """Returns the TF_CONFIG environment variable as dict."""
  return json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))


def get_multi_eval_name(tf_config_env=None):
  """Returns the multi eval index set in the TF_CONFIG."""
  tf_config_env = tf_config_env or _get_tf_config_env()
  return tf_config_env.get(_MULTI_EVAL_NAME)


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

  def create_dataset(self,
                     mode,
                     params=None):
    """Create the actual input_fn.

    This is potentially wrapped in create_dataset_input_fn.

    Args:
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      params: Not used for this implementation but expected by callers. An
        optional dict of hyper parameters that will be passed into input_fn and
        model_fn. Keys are names of parameters, values are basic Python types.
        There are reserved keys for TPUEstimator, including 'batch_size'.

    Returns:
      A valid input_fn for the estimator api.
    """
    input_fn = tfdata.get_input_fn(
        file_patterns=self._file_patterns or self._dataset_map,
        batch_size=self.batch_size,
        feature_spec=self._feature_spec,
        label_spec=self._label_spec,
        mode=mode,
        preprocess_fn=self._preprocess_fn)
    return input_fn(params)


@gin.configurable
class FractionalRecordInputGenerator(DefaultRecordInputGenerator):
  """Fraction of files in dataset (e.g. data ablation experiments)."""

  def __init__(self,
               file_fraction = 1.0,
               **parent_kwargs):
    """Create an instance.

    Args:
      file_fraction: If file_fraction < 1.0, choose first file_fraction percent
        of files, (rounded down to the nearest integer number of files).
      **parent_kwargs: All parent arguments.
    """
    super(FractionalRecordInputGenerator, self).__init__(**parent_kwargs)
    if file_fraction < 1.0:
      data_format, filenames = tfdata.get_data_format_and_filenames(
          self._file_patterns)
      n = int(file_fraction * len(filenames))
      filenames = filenames[:n]
      self._file_patterns = '{}:{}'.format(data_format, ','.join(filenames))


@gin.configurable
class MultiEvalRecordInputGenerator(DefaultRecordInputGenerator):
  """Evaluating on multiple datasets."""

  def __init__(self,
               eval_map = None,
               **parent_kwargs):
    super(MultiEvalRecordInputGenerator, self).__init__(**parent_kwargs)
    # If multi_eval_name is set via TF_CONFIG_ENV variable, override dataset.
    multi_eval_name = get_multi_eval_name()
    if multi_eval_name:
      self._file_patterns = eval_map[multi_eval_name]
    else:
      raise ValueError('multi_eval_name not set in TF_CONFIG env variable')


class GeneratorInputGenerator(
    six.with_metaclass(abc.ABCMeta,
                       abstract_input_generator.AbstractInputGenerator)):
  """Class to use for constructing input generators from Python generator objects."""

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

  def create_dataset(self, mode, params=None):
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

    feature_dtypes = tf.nest.map_structure(lambda x: x.dtype,
                                           self._feature_spec)
    feature_shapes = tf.nest.map_structure(shape_transform, self._feature_spec)
    label_dtypes = tf.nest.map_structure(lambda x: x.dtype, self._label_spec)
    label_shapes = tf.nest.map_structure(shape_transform, self._label_spec)

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


@gin.configurable
class WeightedRecordInputGenerator(DefaultRecordInputGenerator):
  """An input generator sampling multiple datasets with weighting."""

  def __init__(
      self,
      file_patterns,
      num_parallel_calls = 4,
      shuffle_buffer_size = 500,
      prefetch_buffer_size = (tf.data.experimental.AUTOTUNE),
      parallel_shards = 10,
      weights = None,
      seed = None,
      **parent_kwargs):
    """Create an instance.

    Args:
      file_patterns: Comma-separated string of file patterns, where each
        individual file contains data for one task.
      num_parallel_calls: The number elements to process in parallel.
      shuffle_buffer_size: How many examples to shuffle within each file.
      prefetch_buffer_size: How many examples to prefetch.
      parallel_shards: Shards for applying preprocess_fn.
      weights: Weight to sampling each file pattern with. Should be equal in
        length to number of file_patterns.
      seed: Seed for weighted dataset sampling.
      **parent_kwargs: All parent arguments.
    """
    super(WeightedRecordInputGenerator, self).__init__(**parent_kwargs)
    self._file_patterns = file_patterns
    self._num_parallel_calls = num_parallel_calls
    self._shuffle_buffer_size = shuffle_buffer_size
    self._prefetch_buffer_size = prefetch_buffer_size
    self._parallel_shards = parallel_shards
    self._weights = weights
    self._seed = seed

  def create_dataset(self, mode, params, **unused_kwargs):
    """This abstract function is not required for default input generators."""
    batch_size = tfdata.get_batch_size(params, self.batch_size)
    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    data_format, filenames_list = tfdata.get_data_format_and_filenames_list(
        self._file_patterns)
    datasets = []
    if self._weights is not None:
      if len(filenames_list) != len(self._weights):
        raise ValueError(
            'Weights need to be same length as number of filenames.')
    for filenames in filenames_list:
      filenames_dataset = tf.data.Dataset.list_files(
          filenames, shuffle=is_training)
      if is_training:
        cycle_length = min(self._parallel_shards, len(filenames))
      else:
        cycle_length = 1
      dataset = filenames_dataset.apply(
          tf.data.experimental.parallel_interleave(
              tfdata.DATA_FORMAT[data_format],
              cycle_length=cycle_length,
              sloppy=is_training))
      if is_training:
        dataset = dataset.shuffle(
            buffer_size=self._shuffle_buffer_size).repeat()
      else:
        dataset = dataset.repeat()
      datasets.append(dataset)
    if self._weights is None:
      weights = [float(1) for _ in range(len(datasets))]
    else:
      weights = self._weights
    sum_weight = np.sum(weights)
    weights = [float(w) / sum_weight for w in weights]
    dataset = tf.data.experimental.sample_from_datasets(
        datasets=datasets, weights=weights, seed=self._seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Parse all datasets together.
    dataset = tfdata.serialized_to_parsed(
        dataset,
        self._feature_spec,
        self._label_spec,
        num_parallel_calls=self._num_parallel_calls)
    if self._preprocess_fn is not None:
      dataset = dataset.map(
          self._preprocess_fn, num_parallel_calls=self._parallel_shards)
    if self._prefetch_buffer_size is not None:
      dataset = dataset.prefetch(self._prefetch_buffer_size)
    return dataset
