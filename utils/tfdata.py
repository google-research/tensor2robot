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

"""Automatic tf.data.Dataset input pipeline from TensorSpec collections."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import itertools
from absl import logging
import gin
import six

from tensor2robot.utils import tensorspec_utils
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text, Tuple



nest = tf.contrib.framework.nest

DATA_FORMAT = {
    'tfrecord': tf.data.TFRecordDataset
}
RECORD_READER = {
    'tfrecord': tf.TFRecordReader
}


def get_batch_size(params, batch_size):
  """Get the correct batch size to use in an input_fn.

  TPUEstimator may provide different batch sizes if using it in
  tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2 setting.
  See comments about `per_host_input_for_training` in
  //third_party/tensorflow/python/tpu/tpu_config.py.

  Args:
    params: Hyperparameters passed to the input_fn. For TPUEstimator this
      includes 'batch_size'
    batch_size: Batch size provided to the input_fn.

  Returns:
    The batch size to invoke for the input_fn.
  """
  params_batch_size = params.get('batch_size') if params else None
  if params_batch_size and params_batch_size != batch_size:
    logging.info(
        'The input_fn has a batch_size set through `params`, as '
        'well as in the input generator. These batch sizes do '
        'not match. Using the batch size %d from params',
        params_batch_size)
    return params_batch_size
  return batch_size


def infer_data_format(file_patterns):
  """Infer the data format from a file pattern.

  Args:
    file_patterns: A comma-separated string of file patterns [recordio, sstable,
      tfrecord] we will load the data from.

  Raises:
    ValueError: In case more than one data formats are found within the file
      patterns.
  Returns:
    data_format: A valid key for DATA_FORMAT inferred from the file patterns.
  """
  data_format = None
  for key in DATA_FORMAT:
    if key in file_patterns:
      if data_format is not None:
        raise ValueError('More than one data_format {} and {} have '
                         'been found in {}.'.format(key, data_format,
                                                    file_patterns))
      data_format = key

  if data_format is None:
    raise ValueError(
        'Could not infer file record type from extension '
        'of pattern "%s"' % file_patterns)
  return data_format


def get_data_format_and_filenames(
    file_patterns):
  """Obtain the data format and filenames from comma-separated file patterns.

  Args:
    file_patterns: A comma-separated string of file patterns [recordio, sstable,
      tfrecord] we will load the data from.

  Raises:
    ValueError: In case no files can be found that matches the file patterns.

  Returns:
    data_format: A valid key for DATA_FORMAT inferred from the file patterns.
    filenames: All files that match the comma-separated file patterns.
  """
  data_format = infer_data_format(file_patterns)
  file_patterns = file_patterns.replace('{}:'.format(data_format), '')
  filenames = list(
      itertools.chain.from_iterable(
          tf.io.gfile.glob(pattern) for pattern in file_patterns.split(',')))
  if not filenames:
    raise ValueError('File list for pattern {} is empty'.format(file_patterns))
  return data_format, filenames


# (TODO) - replace this call with utils from
# tasks/grasping/inhand_classification/dataset_utils.py
def get_dataset_metadata(file_patterns):
  """Get approximate dataset size for optimal shuffling parameters.

  Args:
    file_patterns: A comma-separated string of file patterns.

  Returns:
    data_format: String describing what type of records we are reading from. One
      of ('sstable', 'recordio', 'tfrecord').
    num_shards: How many shards there are.
    num_examples_per_shard: Approximately how many examples are in each shard.

  Raises:
    ValueError: if `file_pattern` does not match any files.
  """
  data_format, files = get_data_format_and_filenames(
      file_patterns=file_patterns)
  num_shards = len(files)
  # estimate
  logging.info('Estimating dataset size from %s...', files[0])
  if data_format == 'sstable':
    num_examples_per_shard = len(sstable.SSTable(files[0]))
  else:
    # Assume at least one example per shard.
    num_examples_per_shard = 1
    with RECORD_READER[data_format](files[0]) as reader:
      for _ in reader:
        num_examples_per_shard += 1
  return data_format, num_shards, num_examples_per_shard


def parallel_read(file_patterns,
                  num_shards = None,
                  num_readers = None,
                  num_epochs = None):
  """Parallel reading of serialized TFExamples stored in files.

  See http://www.moderndescartes.com/essays/shuffle_viz/ for more information
  on how to ensure high-discrepancy random sampling.

  Args:
    file_patterns: A comma-separated string of file patterns [recordio, sstable,
      tfrecord] we will load the data from.
    num_shards: How many shards there are.
    num_readers: How many tasks to file shards to read in parallel. Optimal
      num_readers is the same as number of shards.
    num_epochs: How many epochs to repeat the input data. By default, repeats
      forever.

  Returns:
    Dataset whose outputs are serialized protos.

  """
  data_format, filenames = get_data_format_and_filenames(
      file_patterns=file_patterns)
  if num_shards is None:
    num_shards = len(filenames)
  # Dataset of shard names.
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  # Shuffle shards (avoid inter-shard correlations).
  dataset = dataset.apply(
      tf.data.experimental.shuffle_and_repeat(num_shards, num_epochs))
  # Interleave shards (avoid intra-shard correlations).
  if num_readers is None:
    num_readers = num_shards
  dataset = dataset.apply(
      tf.data.experimental.parallel_interleave(
          DATA_FORMAT[data_format], cycle_length=num_readers))
  return dataset


def serialized_to_parsed(dataset,
                         feature_tspec,
                         label_tspec,
                         num_parallel_calls = 2):
  """Auto-generating TFExample parsing code from feature and label tensor specs.

  Supports both single-TFExample parsing (default) and batched parsing (e.g.
  when we are pulling batches from Replay Buffer).

  Args:
    dataset: tf.data.Dataset whose outputs are serialized tf.Examples.
    feature_tspec: Collection of TensorSpec designating how to extract features.
    label_tspec: Collection of TensorSpec designating how to extract labels.
    num_parallel_calls: (Optional.) A tf.int32 scalar tf.Tensor, representing
      the number elements to process in parallel. If not specified, elements
      will be processed sequentially.

  Returns:
    tf.data.Dataset whose output is single (features, labels) tuple.
  """
  tensor_dict = {}
  tensor_spec_dict = {}
  feature_dict, feature_tspec_dict = (
      tensorspec_utils.tensorspec_to_feature_dict(feature_tspec))
  tensor_dict.update(feature_dict)
  tensor_spec_dict.update(feature_tspec_dict)
  label_dict, label_tspec_dict = (
      tensorspec_utils.tensorspec_to_feature_dict(label_tspec))
  tensor_dict.update(label_dict)
  tensor_spec_dict.update(label_tspec_dict)

  parse_tf_example_fn = create_parse_tf_example_fn(
      tensor_dict=tensor_dict,
      tensor_spec_dict=tensor_spec_dict,
      feature_tspec=feature_tspec,
      label_tspec=label_tspec)
  dataset = dataset.map(
      map_func=parse_tf_example_fn, num_parallel_calls=num_parallel_calls)
  return dataset


def create_parse_tf_example_fn(tensor_dict,
                               tensor_spec_dict,
                               feature_tspec,
                               label_tspec=None):
  """Create a parse function for serialized tf.Example protos.

  Args:
    tensor_dict: The {tensor_spec.name: tf.FixedLenFeature} mapping resulting
      from feature_tspec and the optional label_tspec.
    tensor_spec_dict: The {tensor_spec.name: tensor_spec} mapping resulting from
      feature_tspec and the optional label_tspec.
    feature_tspec: A valid tensor_spec structure for features.
    label_tspec: Optional a valid tensor_spec structure for labels.

  Returns:
    parse_tf_example_fn: A callable function which can take serialized
      tf.Example protos and returns decoded features, according to the
      feature_tspec (label_tspec). If more than one argument is passed in
      (e.g., a (key, value)-tuple from an SSTableDataset), the serialized
      sample is expected to be the last one.
  """

  def parse_tf_example_fn(*input_values):
    """Maps string tensor to a parsed TFExample.

    Args:
      *input_values: A string tensor if mapping from a RecordIODataset or
        TFRecordDataset, or a (key, string tensor) tuple if mapping from a
        SSTableDataset.

    Returns:
      features: Collection of tensors conforming to feature_tspec.
      labels: Collection of tensors conforming to label_tspec.
    Raises:
        ValueError: If dtype other than uint8 is supplied for image specs.
    """
    if len(input_values) == 2:
      # Assume an SSTable key, value pair.
      _, example_proto = input_values
    else:
      example_proto, = input_values

    def parse_wrapper(example, spec_dict):
      """Wrap tf.parse_example to support bfloat16 dtypes.

      This allows models which declare bfloat16 as inputs to not require an
      additional preprocessing step to cast all inputs from float32 to bfloat16.
      Consider this to be analogous to JPEG decoding in the data step.

      Args:
        example: TFExample
        spec_dict: Dictionary of feature name -> tf.FixedLenFeature
      Returns:
        Parsed feature map
      """
      def is_bfloat_feature(value):
        return value.dtype == tf.bfloat16

      def maybe_map_bfloat(value):
        if is_bfloat_feature(value):
          if isinstance(value, tf.FixedLenFeature):
            return tf.FixedLenFeature(value.shape, tf.float32,
                                      default_value=value.default_value)
          else:
            return tf.FixedLenSequenceFeature(
                value.shape, tf.float32, default_value=value.default_value)
        return value

      # Change bfloat features to float32 for parsing.
      new_spec_dict = {
          k: maybe_map_bfloat(v) for k, v in six.iteritems(spec_dict)}
      for k, v in six.iteritems(new_spec_dict):
        if v.dtype not in [tf.float32, tf.string, tf.int64]:
          raise ValueError('Feature specification with invalid data type for '
                           'tf.Example parsing: "%s": %s' % (k, v.dtype))

      # Separate new_spec_dict into Context and Sequence features. In the event
      # that there are no SequenceFeatures, the context_features dictionary
      # (containing FixedLenFeatures) is passed to tf.parse_examples.
      context_features, sequence_features = {}, {}
      for k, v in six.iteritems(new_spec_dict):
        v = maybe_map_bfloat(v)
        if isinstance(v, tf.FixedLenSequenceFeature):
          sequence_features[k] = v
        elif isinstance(v, tf.FixedLenFeature):
          context_features[k] = v
        else:
          raise ValueError(
              'Only FixedLenFeature and FixedLenSequenceFeature are currently '
              'supported.')

      # If there are any sequence features, we use parse_sequence_example.
      if sequence_features:
        result, sequence_result, feature_lengths = tf.io.parse_sequence_example(
            example, context_features=context_features,
            sequence_features=sequence_features)
        del feature_lengths
        result.update(sequence_result)
      else:
        result = tf.parse_example(example, context_features)
      to_convert = [
          k for k, v in six.iteritems(spec_dict) if is_bfloat_feature(v)]

      for c in to_convert:
        result[c] = tf.cast(result[c], tf.bfloat16)

      return result

    parsed_tensors = parse_wrapper(example_proto, tensor_dict)

    # Interpret encoded images.
    def decode_image(key, raw_bytes):
      """Decodes single or batches of JPEG- or PNG-encoded string tensors.

      Args:
        key: String key specified in feature map.
        raw_bytes: String tensor to decode as JPEG or PNG.

      Returns:
        Decoded image tensor with shape specified by tensor spec.
      Raises:
        ValueError: If dtype other than uint8 is supplied for image specs.
      """
      img_batch_dims = tf.shape(raw_bytes)
      # The spatial + channel dimensions of a single image, assumed to be the
      # last 3 entries of the image feature's tensor spec.
      single_img_dims = tensor_spec_dict[key].shape[-3:]

      # Collapse (possibly multiple) batch dims to a single batch dim for
      # decoding purposes.
      raw_bytes = tf.reshape(raw_bytes, [-1])
      img = tf.map_fn(
          tf.image.decode_image, raw_bytes, dtype=tf.uint8, back_prop=False)
      img.set_shape(raw_bytes.shape.concatenate(single_img_dims))

      # Expand the collapsed batch dim back to the original img_batch_dims.
      img = tf.reshape(img, tf.concat([img_batch_dims, single_img_dims], 0))

      return img

    for key, val in parsed_tensors.items():
      tensor_spec = tensor_spec_dict[key]
      if tensorspec_utils.is_encoded_image_spec(tensor_spec):
        parsed_tensors[key] = decode_image(key, val)
        if tensor_spec.dtype != tf.uint8:
          raise ValueError('Encoded images with key {} must be '
                           'specified with uint8 dtype.'.format(key))

    # Ensure that we have a consistent ordered mapping despite the underlying
    # spec structure.
    flat_feature_tspec = tensorspec_utils.TensorSpecStruct(
        sorted(tensorspec_utils.flatten_spec_structure(feature_tspec).items()))
    # Using the flat spec structure we allow to map the same parsed_tensor
    # to multiple features or labels. Note, the spec structure ensures that
    # the corresponding tensorspecs are iddentical in such cases.
    features = tensorspec_utils.TensorSpecStruct(
        [(key, parsed_tensors[value.name])
         for key, value in flat_feature_tspec.items()])

    features = tensorspec_utils.validate_and_pack(
        flat_feature_tspec, features, ignore_batch=True)
    if label_tspec is not None:
      # Ensure that we have a consistent ordered mapping despite the underlying
      # spec structure.
      flat_label_tspec = tensorspec_utils.TensorSpecStruct(
          sorted(tensorspec_utils.flatten_spec_structure(label_tspec).items()))
      labels = tensorspec_utils.TensorSpecStruct(
          [(key, parsed_tensors[value.name])
           for key, value in flat_label_tspec.items()])
      labels = tensorspec_utils.validate_and_pack(
          flat_label_tspec, labels, ignore_batch=True)
      return features, labels
    return features

  return parse_tf_example_fn


@gin.configurable
def grasping_input_fn_tmpl(
    file_patterns,
    batch_size,
    feature_spec,
    label_spec,
    num_parallel_calls = 4,
    is_training = False,
    preprocess_fn=None,
    shuffle_buffer_size = 1000,
    prefetch_buffer_size = (tf.data.experimental.AUTOTUNE),
    parallel_shards = 10):
  """Near-compatibility with grasping_data.py input pipeline."""
  data_format, filenames = get_data_format_and_filenames(file_patterns)

  filenames_dataset = tf.data.Dataset.list_files(filenames, shuffle=is_training)
  if is_training:
    cycle_length = min(parallel_shards, len(filenames))
  else:
    cycle_length = 1
  dataset = filenames_dataset.apply(
      tf.contrib.data.parallel_interleave(
          DATA_FORMAT[data_format],
          cycle_length=cycle_length,
          sloppy=is_training))

  if is_training:
    dataset = dataset.apply(
        tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size))
  else:
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)

  dataset = serialized_to_parsed(
      dataset,
      feature_spec,
      label_spec,
      num_parallel_calls=num_parallel_calls)

  if preprocess_fn is not None:
    # TODO(psanketi): Consider adding num_parallel calls here.
    dataset = dataset.map(preprocess_fn, num_parallel_calls=parallel_shards)
  dataset = dataset.prefetch(prefetch_buffer_size)
  return dataset


def get_input_fn(feature_spec, label_spec, file_patterns, mode, batch_size,
                 preprocess_fn):
  """Input function for record-backed data."""
  def input_fn(params=None):
    """Input function passed to the Estimator API.

    Args:
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      features, labels tensor expected by `model_fn`.
    """
    used_batch_size = get_batch_size(params, batch_size)
    dataset = grasping_input_fn_tmpl(
        file_patterns=file_patterns,
        batch_size=used_batch_size,
        feature_spec=feature_spec,
        label_spec=label_spec,
        is_training=(mode == tf.estimator.ModeKeys.TRAIN),
        preprocess_fn=preprocess_fn)
    return dataset

  return input_fn
