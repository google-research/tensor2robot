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

# Lint as python3
"""Utility function for dealing with meta-examples.
"""

from typing import List, Union

import six
import tensorflow.compat.v1 as tf  # tf

Example = Union[tf.train.Example, tf.train.SequenceExample]


def make_meta_example(
    condition_examples,
    inference_examples,
):
  """Creates a single MetaExample from train_examples and val_examples."""
  if isinstance(condition_examples[0], tf.train.Example):
    meta_example = tf.train.Example()
    append_fn = append_example
  else:
    meta_example = tf.train.SequenceExample()
    append_fn = append_sequence_example
  for i, train_example in enumerate(condition_examples):
    append_fn(meta_example, train_example, 'condition_ep{:d}'.format(i))

  for i, val_example in enumerate(inference_examples):
    append_fn(meta_example, val_example, 'inference_ep{:d}'.format(i))
  return meta_example


def append_example(example, ep_example, prefix):
  """Add episode Example to Meta TFExample with a prefix."""
  context_feature_map = example.features.feature
  for key, feature in six.iteritems(ep_example.features.feature):
    context_feature_map[six.ensure_str(prefix) + '/' +
                        six.ensure_str(key)].CopyFrom(feature)


def append_sequence_example(meta_example, ep_example, prefix):
  """Add episode SequenceExample to the Meta SequenceExample with a prefix."""
  context_feature_map = meta_example.context.feature
  # Append context features.
  for key, feature in six.iteritems(ep_example.context.feature):
    context_feature_map[six.ensure_str(prefix) + '/' +
                        six.ensure_str(key)].CopyFrom(feature)
  # Append Sequential features.
  sequential_feature_map = meta_example.feature_lists.feature_list
  for key, feature_list in six.iteritems(ep_example.feature_lists.feature_list):
    sequential_feature_map[six.ensure_str(prefix) + '/' +
                           six.ensure_str(key)].CopyFrom(feature_list)
