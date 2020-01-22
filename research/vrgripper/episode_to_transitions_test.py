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
"""Tests for episodes_to_transitions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
from tensor2robot.research.vrgripper import episode_to_transitions
import tensorflow.compat.v1 as tf


class EpisodeToTransitionsTest(tf.test.TestCase):

  def test_make_fixed_length(self):
    fixed_length = 10
    dummy_feature_dict_lists = [
        [{'dummy_feature': i} for i in range(5)],
        [{'dummy_feature': i} for i in range(20)],
    ]

    for feature_dict_list in dummy_feature_dict_lists:
      filtered_feature_dict_list = episode_to_transitions.make_fixed_length(
          feature_dict_list,
          fixed_length=fixed_length,
          always_include_endpoints=True)
      self.assertLen(filtered_feature_dict_list, fixed_length)

      # The first and last entries of the original list should be present in
      # the filtered list.
      self.assertEqual(feature_dict_list[0], filtered_feature_dict_list[0])
      self.assertEqual(feature_dict_list[-1], filtered_feature_dict_list[-1])
