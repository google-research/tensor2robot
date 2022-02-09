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

"""Functions for converting env episode data to tfrecords of transitions."""

import gin
from PIL import Image
from tensor2robot.utils import image
import tensorflow.compat.v1 as tf

_bytes_feature = (
    lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v)))
_int64_feature = (
    lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v)))
_float_feature = (
    lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v)))


@gin.configurable
def episode_to_transitions_pose_toy(episode_data):
  """Converts pose toy env episode data to transition Examples."""
  # This is just saving data for a supervised regression problem, so obs_tp1
  # can be discarded.
  transitions = []
  for transition in episode_data:
    (obs_t, action, reward, obs_tp1, done, debug) = transition
    del obs_tp1
    del done
    features = {}
    obs_t = Image.fromarray(obs_t)
    features['state/image'] = _bytes_feature([image.jpeg_string(obs_t)])
    features['pose'] = _float_feature(action.flatten().tolist())
    features['reward'] = _float_feature([reward])
    features['target_pose'] = _float_feature(debug['target_pose'].tolist())
    transitions.append(
        tf.train.Example(features=tf.train.Features(feature=features)))
  return transitions


