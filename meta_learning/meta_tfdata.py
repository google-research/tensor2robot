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
"""Utility functions for providing input data to meta-learning algorithms."""

import collections
import gin
from tensor2robot.utils import tensorspec_utils as utils
from tensor2robot.utils import tfdata
import tensorflow.compat.v1 as tf  # tf
from tensorflow.contrib import framework as contrib_framework

nest = contrib_framework.nest
TrainValPair = collections.namedtuple(
    'TrainValPair', ['train', 'val', 'val_mode'])


@gin.configurable
def parallel_read(file_patterns,
                  parse_fn,
                  shuffle_filenames=True,
                  num_train_samples_per_task=4,
                  num_val_samples_per_task=4,
                  shuffle_buffer_size=50,
                  filter_fn=None,
                  interleave_cycle_length=None,
                  mode=tf.estimator.ModeKeys.TRAIN):
  """Read and parse multiple examples per task, per train/val split.

  This pipeline does the following:
    1. Shuffle & repeats filenames.
    2. Open up and shuffle each file.
    3. Outputs num_train + num_val examples from each shard
    4. De-serialize the protos.

  Args:
    file_patterns: Comma-separated string of file patterns, where each
      individual file contains data for one task.
    parse_fn: Python function that takes as an argument a Dataset
      whose output is a string tensor and returns a Dataset
      whose output is a collection of parsed TFEXamples.
    shuffle_filenames: If True, filenames are shuffled so tasks are sampled at
      random.
    num_train_samples_per_task: How many examples to dequeue for training.
    num_val_samples_per_task: How many examples to dequeue for validation.
    shuffle_buffer_size: How many examples to shuffle within each file.
    filter_fn: A callable function which will get a valid tensorspec filled with
      tensors as input and returns a tf.bool as output. If true the dataset
      will keep the example or drop it otherwise.
    interleave_cycle_length: Integer, cycle length when interleaving examples
      from different task files. If 0 or None, cycle length will default
      to num_tasks.
    mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

  Returns:
    Dataset whose output is that of `parse_fn` (e.g. features, labels).

  Raises:
    ValueError: File patterns do not exist.
  """
  data_format, filenames = tfdata.get_data_format_and_filenames(file_patterns)

  # We shuffle filenames. Dequeues filenames.
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  num_tasks = len(filenames)
  # Shuffle returns a new permutation *per epoch*. Upon epoch completion,
  # shuffling is repeated.
  if shuffle_filenames:
    dataset = dataset.shuffle(buffer_size=num_tasks).repeat()
  else:
    dataset = dataset.repeat()
  # From `task_batch_size` tasks at a time, dequeue 2*N elements, where N is
  # num_samples_per_task. These form N training and N validation instances for
  # meta-learning.
  def filename_map_fn(x):
    """Builds a dataset for a file path.

    Args:
      x: path name of data
    Returns:
      A dataset of parsed tensors, filtered by filter_fn.
    """
    dataset_ = tfdata.DATA_FORMAT[data_format](x)
    # It is important to have at least num_train_samples_per_task +
    # num_val_samples_per_task in the dataset in order to apply the batching
    # thereafter.
    effective_shuffle_buffer_size = max(
        shuffle_buffer_size,
        num_train_samples_per_task + num_val_samples_per_task)
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset_ = dataset_.shuffle(
          buffer_size=effective_shuffle_buffer_size).repeat()
    else:
      dataset_ = dataset_.repeat()
    dataset_ = dataset_.batch(
        batch_size=num_train_samples_per_task + num_val_samples_per_task,
        drop_remainder=True)
    dataset = parse_fn(dataset_)
    if filter_fn is not None:
      dataset = dataset.unbatch().filter(filter_fn).batch(
          batch_size=num_train_samples_per_task + num_val_samples_per_task,
          drop_remainder=True)
    return dataset

  # Sample 1 example-set from each task (file). Dequeues one task's worth
  # (a batch) of SARSTransitions. [num_samples_per_task * 2, ...]
  if not interleave_cycle_length:
    interleave_cycle_length = num_tasks
  dataset = dataset.interleave(
      filename_map_fn,
      cycle_length=interleave_cycle_length,
      block_length=1
  )
  return dataset


def split_train_val(dataset, num_train_samples_per_task):
  """Split a Dataset's output into train and val slices.

  Args:
    dataset: tf.data.Dataset whose output is features, labels.
    num_train_samples_per_task: Integer specifying .
  Returns:
    Dataset whose outputs are TrainValPair where each of train and val have
      the same structure as the original dataset.
  """
  # Split SARSTransition in half, into train/validation tuples.
  def train_val_split_fn(features, labels):
    def _split(tensors):
      train_tensors = nest.map_structure(
          lambda x: x[:num_train_samples_per_task], tensors)
      val_tensors = nest.map_structure(
          lambda x: x[num_train_samples_per_task:], tensors)
      return TrainValPair(train_tensors, val_tensors,
                          tf.convert_to_tensor([True]))

    return _split(features), _split(labels)
  return dataset.map(train_val_split_fn)


def tile_val_mode(pair):
  """Tile val_mode to num_tasks * num_train_samples_per_task batch elements.

  Args:
    pair: TensorSpecStruct whose tensors have shape [num_tasks,
      num_train_samples_per_task, ...].

  Raises:
    ValueError: If num_train_samples does not match num_val_samples.
  """
  train_tensor = list(utils.flatten_spec_structure(pair.train).values())[0]
  num_train_samples_per_task = train_tensor.shape.as_list()[1]
  val_tensor = list(utils.flatten_spec_structure(pair.val).values())[0]
  num_val_samples_per_task = val_tensor.shape.as_list()[1]
  if num_train_samples_per_task != num_val_samples_per_task:
    raise ValueError('Flattening example and batch dimensions requires '
                     'num_train_samples and num_val_samples to be the same.')
  pair.val_mode = tf.tile(pair.val_mode, [num_train_samples_per_task, 1])


def flatten_batch_examples(
    tensor_collection
):
  """Flatten task and samples dimension for M-L algorithms like RL^2.

  Args:
    tensor_collection: Tensor collection whose tensors have shape [num_tasks,
      num_train_samples_per_task, ...].

  Returns:
    Tensor collection with the same structure as `tensor_collection`, but with
      first two dimensions flattened.

  """
  def reshape_batch(x):
    if x.shape.ndims == 1:
      return x
    if x.shape.as_list()[1] is None:
      new_shape = tf.concat([[-1], tf.shape(x)[2:]], axis=0)
      return tf.reshape(x, new_shape)
    else:
      s = tf.shape(x)
      new_shape = tf.concat([[s[0] * s[1]], s[2:]], axis=0)
      return tf.reshape(x, new_shape)
  return nest.map_structure(reshape_batch, tensor_collection)


def unflatten_batch_examples(
    tensor_collection,
    num_samples_per_task):
  """Unflatten task and samples dimension for M-L algorithms like RL^2.

  Args:
    tensor_collection: Tensor collection whose tensors have shape [num_tasks *
      num_samples_per_task, ...].
    num_samples_per_task: The number of samples we have per task.

  Returns:
    result_unflattened: A tensor_collection with the same elements but the
      shape has been changed to [num_tasks, num_train_samples_per_task, ...].
  """
  result_unflattened = utils.flatten_spec_structure(tensor_collection)
  for key, value in result_unflattened.items():
    result_unflattened[key] = tf.reshape(
        value, [-1, num_samples_per_task] + value.shape.as_list()[1:])
  return result_unflattened


def merge_first_n_dims(structure, n):
  """Merges together the first n dims of each tensor in structure.

  Args:
    structure: A structure (tuple, namedtuple, list, dictionary, ...).
    n: Integer, the number of dimensions to merge together.
  Returns:
    A structure matching the input structure, where the first n dimensions
    of each tensor have been merged together.
  """
  def _helper(elem):
    if isinstance(elem, tf.Tensor):
      shape = tf.shape(elem)
      return tf.reshape(elem, tf.concat([[-1], shape[n:]], axis=0))
    else:
      return elem
  return nest.map_structure(_helper, structure)


def expand_batch_dims(structure, batch_sizes):
  """Expands the first dimension of each tensor in structure to be batch_sizes.

  Args:
    structure: A structure (tuple, namedtuple, list, dictionary, ...).
    batch_sizes: A 1-D tensor of shapes describing the batch dims.
  Returns:
    A structure matching the input structure, where each tensor's first
    dimension has been expanded to match batch_sizes.
  """
  def _helper(tensor):
    if isinstance(tensor, tf.Tensor):
      shape = tf.shape(tensor)
      return tf.reshape(tensor, tf.concat([batch_sizes, shape[1:]], axis=0))
    else:
      return tensor
  return nest.map_structure(_helper, structure)


# TODO(T2R_CONTRIBUTORS): Refactor this code to a separate file.
def multi_batch_apply(f, num_batch_dims, *args, **kwargs):
  """Vectorized application of f on tensors with multiple batch dims.

  Batch dims must be the same for every tensor in args/kwargs.

  Args:
    f: Callable, needs only expect one batch dim in input tensors.
    num_batch_dims: Integer, the number of batch dims.
    *args: Args passed into f (tensors will be reshaped to 1 batch dim).
    **kwargs: Kwargs passed into f (tensors will be reshaped to 1 batch dim).
  Returns:
    The result of calling f on args, kwargs.
  """
  flattened_inputs = nest.flatten(args) + nest.flatten(kwargs)
  tensor_inputs = [inp for inp in flattened_inputs
                   if isinstance(inp, tf.Tensor)]
  batch_sizes = tf.shape(tensor_inputs[0])[:num_batch_dims]
  merged_args = merge_first_n_dims(args, num_batch_dims)
  merged_kwargs = merge_first_n_dims(kwargs, num_batch_dims)
  outputs = f(*merged_args, **merged_kwargs)
  return expand_batch_dims(outputs, batch_sizes)
