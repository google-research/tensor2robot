# coding=utf-8
# Copyright 2024 The Tensor2Robot Authors.
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

"""Generates random subsampling indices."""

import numpy as np
import tensorflow.compat.v1 as tf


def get_uniform_subsample_indices(sequence_lengths,
                                  min_length):
  """Generate consistent frame rate indices.

  Given a sequence, this will always select the same frames, and is
  guaranteed to include the last frame. It isn't guaranteed to include the
  first.

  Note that if min_length = 1 then we will always return the last frame.

  Args:
    sequence_lengths: An int tensor of shape [B], for the length of each
      sequence. Since the provided tensors are padded to the length of the
      longest sequence in the batch, sequence_lengths is needed to avoid
      sampling from the padding.
    min_length: The min_length to subsample.

  Returns:
    An int tensor of shape [B, min_length], for how to index into the sequence.
  """
  def get_indices(sequence_length):
    indices = tf.cast(tf.range(min_length), float)
    indices = tf.round(indices * tf.cast(sequence_length-1, float) / min_length)
    indices = tf.cast(sequence_length - 1, float) - indices
    return tf.sort(tf.cast(indices, tf.int64))
  indices = tf.map_fn(get_indices, sequence_lengths)
  batch_size = sequence_lengths.shape[0]
  indices.set_shape((batch_size, min_length))
  return indices


def get_subsample_indices_nofirstlast(sequence_lengths,
                                      min_length):
  """Generate random indices for sequence subsampling, no first/last needed.

  Given a sequence, this will randomly sample min_length frames from the
  sequence with replacement. There is no requirement to include the first
  or last frame of the sequence.

  Args:
    sequence_lengths: An int tensor of shape [B], for the length of each
      sequence. Since the provided tensors are padded to the length of the
      longest sequence in the batch, sequence_lengths is needed to avoid
      sampling from the padding.
    min_length: The min_length to subsample.

  Returns:
    An int tensor of shape [B, min_length], for how to index into the sequence.
  """
  def get_indices(sequence_length):
    indices = (tf.random.uniform(shape=[min_length]) *
               tf.cast(sequence_length, float))
    indices = tf.cast(tf.math.floor(indices), tf.int64)
    return tf.sort(indices)
  indices = tf.map_fn(get_indices, sequence_lengths)
  batch_size = sequence_lengths.shape[0]
  indices.set_shape((batch_size, min_length))
  return indices


def get_subsample_indices(sequence_lengths,
                          min_length):
  """Generates random indices for sequence subsampling.

  Args:
    sequence_lengths: An int tensor of shape [B], for the length of each
      sequence. Since the provided tensors are padded to the length of the
      longest sequence in the batch, sequence_lengths is needed to avoid
      sampling from the padding.
    min_length: The min_length to subsample. The sampling always includes the
      first and last frame. If the sequence is long enough, timesteps are
      sampled without replacement. Otherwise, they are sampled with
      replacement. If min_length = 1, the example is picked randomly from
      the entire sequence.

  Returns:
    An int tensor of shape [B, min_length], for how to index into the sequence.
  """
  def get_indices(sequence_length):
    """Generates indices for single sequence."""
    def without_replacement():
      # sample integers from [1, seq_len-1)
      indices = tf.random.shuffle(tf.range(1, sequence_length-1))
      middle = indices[:min_length - 2]
      return tf.sort(tf.concat([[0], middle, [sequence_length-1]], axis=0))

    def with_replacement():
      # sample integers from [0, seq_len)
      indices = (
          tf.random.uniform(shape=[min_length - 2]) *
          tf.cast(sequence_length, float))
      middle = tf.cast(tf.math.floor(indices), tf.int64)
      return tf.sort(tf.concat([[0], middle, [sequence_length-1]], axis=0))

    def random_frame():
      # Used when min_length == 1.
      indices = (
          tf.random.uniform(shape=[min_length]) *
          tf.cast(sequence_length, float))
      middle = tf.cast(tf.math.floor(indices), tf.int64)
      return tf.sort(middle)

    # pylint: disable=g-long-lambda
    samples = tf.cond(
        tf.equal(min_length, 1),
        random_frame,
        lambda: tf.cond(
            sequence_length >= min_length,
            without_replacement,
            with_replacement))
    # pylint: enable=g-long-lambda
    return samples

  indices = tf.map_fn(get_indices, sequence_lengths)
  batch_size = sequence_lengths.shape[0]
  indices.set_shape((batch_size, min_length))
  return indices


def get_subsample_indices_randomized_boundary(sequence_lengths,
                                              min_length,
                                              min_delta_t,
                                              max_delta_t):
  """Generates random indices for sequence subsampling with random boundaries.

  Args:
    sequence_lengths: An int tensor of shape [B], for the length of each
      sequence. Since the provided tensors are padded to the length of the
      longest sequence in the batch, sequence_lengths is needed to avoid
      sampling from the padding.
    min_length: The min_length to subsample. The sampling always includes the
      first and last frame. If the sequence is long enough, timesteps are
      sampled without replacement. Otherwise, they are sampled with
      replacement. If min_length = 1, the example is picked randomly from
      the entire sequence.
    min_delta_t: smallest range of time-steps that can be sampled,
     if sampled length smaller sequence_length,
     then we use sequence_length instead
    max_delta_t: largest range of time-steps that can be sampled

  Returns:
    An int tensor of shape [B, min_length], for how to index into the seuqence.
  """

  def get_indices(sequence_length):
    """Generates indices for single sequence."""
    episode_delta_t = tf.random.uniform(shape=[], minval=min_delta_t,
                                        maxval=max_delta_t + 1,
                                        dtype=tf.dtypes.int64)
    episode_delta_t = tf.math.reduce_min(tf.concat([sequence_length[None],
                                                    episode_delta_t[None]],
                                                   axis=0))
    episode_start = tf.random.uniform(shape=[], minval=0,
                                      maxval=(sequence_length -
                                              episode_delta_t + 1),
                                      dtype=tf.dtypes.int64)
    episode_end = episode_start + episode_delta_t - 1

    def without_replacement():
      # sample integers from [episode_start, episode_end)
      indices = tf.random.shuffle(tf.range(episode_start + 1,
                                           episode_end))
      middle = indices[:min_length - 2]
      middle = tf.reshape(middle, [min_length - 2])
      return tf.sort(tf.concat([[episode_start],
                                middle, [episode_end]], axis=0))

    def with_replacement():
      # sample integers from [episode_start, episode_end)
      indices = tf.random.uniform(shape=[min_length - 2]) * \
        tf.cast(episode_delta_t, float)
      middle = episode_start + tf.cast(tf.math.floor(indices), tf.int64)
      return tf.sort(tf.concat([[episode_start], middle,
                                [episode_end]], axis=0))

    def random_frame():
      # Used when min_length == 1.
      return tf.random.uniform([1], minval=episode_start,
                               maxval=episode_end, dtype=tf.dtypes.int64)

    # pylint: disable=g-long-lambda
    samples = tf.cond(
        tf.equal(min_length, 1),
        random_frame,
        lambda: tf.cond(
            episode_delta_t >= min_length,
            without_replacement,
            with_replacement))
    # pylint: enable=g-long-lambda
    return samples

  indices = tf.map_fn(get_indices, sequence_lengths)
  batch_size = sequence_lengths.shape[0]
  indices.set_shape((batch_size, min_length))
  return indices


def get_np_subsample_indices(sequence_lengths,
                             min_length):
  """Same behavior as get_subsample_indices, but in numpy format."""
  def get_indices(sequence_length):
    """Generates indices for single sequence."""
    if min_length == 1:
      return np.random.randint(0, sequence_length, size=(1,))
    elif sequence_length >= min_length:
      # without replacement.
      arr = np.arange(1, sequence_length - 1)
      np.random.shuffle(arr)
      middle = arr[:min_length - 2]
      return np.sort(
          np.concatenate([[0], middle, [sequence_length-1]], axis=0))
    else:
      # with replacement.
      middle = np.random.randint(0, sequence_length, size=[min_length - 2])
      return np.sort(
          np.concatenate([[0], middle, [sequence_length-1]], axis=0))

  # Should do better than for loop at a later time....
  batch_size = sequence_lengths.shape[0]
  indices = np.zeros((batch_size, min_length), dtype=np.int64)
  for i in range(batch_size):
    indices[i] = get_indices(sequence_lengths[i])
  return indices
