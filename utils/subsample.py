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

# Lint as: python3
"""Generates random subsampling indices."""



import tensorflow.compat.v1 as tf


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
    An int tensor of shape [B, min_length], for how to index into the seuqence.
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
