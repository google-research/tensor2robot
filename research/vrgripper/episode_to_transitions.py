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

# Lint as: python2, python3
"""Functions for converting env episode data to tfrecords of transitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gin
import numpy as np
from PIL import Image
import six
from six.moves import range
import tensorflow as tf



_bytes_feature = (
    lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v)))
_int64_feature = (
    lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v)))
_float_feature = (
    lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v)))

_IMAGE_KEY_PREFIX = 'image'


@gin.configurable
def make_fixed_length(
    input_list,
    fixed_length,
    always_include_endpoints=True,
    randomized=True):
  """Create a fixed length list by sampling entries from input_list.

  Args:
    input_list: The original list we sample entries from.
    fixed_length: An integer: the desired length of the output list.
    always_include_endpoints: If True, always include the first and last entries
      of input_list in the output.
    randomized: If True, select entries from input_list by random sampling with
      replacement. If False, select entries from input_list deterministically.
  Returns:
    A list of length fixed_length containing sampled entries of input_list.
  """
  original_length = len(input_list)
  if original_length <= 2:
    return None
  if not randomized:
    indices = np.sort(np.mod(np.arange(fixed_length), original_length))
    return [input_list[i] for i in indices]
  if always_include_endpoints:
    # Always include entries 0 and N-1.
    endpoint_indices = np.array([0, original_length - 1])
    # The remaining (fixed_length-2) frames are sampled with replacement
    # from entries [1, N-1) of input_list.
    other_indices = 1 + np.random.choice(
        original_length - 2, fixed_length-2, replace=True)
    indices = np.concatenate(
        (endpoint_indices, other_indices),
        axis=0)
  else:
    indices = np.random.choice(
        original_length, fixed_length, replace=True)
  indices = np.sort(indices)
  return [input_list[i] for i in indices]




@gin.configurable
def episode_to_transitions_reacher(episode_data, is_demo=False):
  """Converts reacher env data to transition examples."""
  transitions = []
  for i, transition in enumerate(episode_data):
    del i
    feature_dict = {}
    (obs_t, action, reward, obs_tp1, done, debug) = transition
    del debug
    feature_dict['pose_t'] = _float_feature(obs_t)
    feature_dict['pose_tp1'] = _float_feature(obs_tp1)
    feature_dict['action'] = _float_feature(action)
    feature_dict['reward'] = _float_feature([reward])
    feature_dict['done'] = _int64_feature([int(done)])
    feature_dict['is_demo'] = _int64_feature([int(is_demo)])
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    transitions.append(example)
  return transitions


@gin.configurable
def episode_to_transitions_metareacher(episode_data):
  """Converts metareacher env data to transition examples."""
  context_features = {}
  feature_lists = collections.defaultdict(list)

  context_features['is_demo'] = _int64_feature(
      [int(episode_data[0][-1]['is_demo'])])
  context_features['target_idx'] = _int64_feature(
      [episode_data[0][-1]['target_idx']])

  for i, transition in enumerate(episode_data):
    del i
    (obs_t, action, reward, obs_tp1, done, debug) = transition
    del debug
    feature_lists['pose_t'].append(_float_feature(obs_t))
    feature_lists['pose_tp1'].append(_float_feature(obs_tp1))
    feature_lists['action'].append(_float_feature(action))
    feature_lists['reward'].append(_float_feature([reward]))
    feature_lists['done'].append(_int64_feature([int(done)]))

  tf_feature_lists = {}
  for key in feature_lists:
    tf_feature_lists[key] = tf.train.FeatureList(feature=feature_lists[key])

  return [tf.train.SequenceExample(
      context=tf.train.Features(feature=context_features),
      feature_lists=tf.train.FeatureLists(feature_list=tf_feature_lists))]


