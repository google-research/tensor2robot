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
"""MAML-based meta-learning models for the duck task."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
import gin
import numpy as np
from tensor2robot.meta_learning import maml_model
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework
nest = contrib_framework.nest


@gin.configurable
class PoseEnvRegressionModelMAML(maml_model.MAMLModel):
  """MAML Regression environment for duck task."""

  def _make_dummy_labels(self):
    """Helper function to make dummy labels for pack_labels."""
    label_spec = self._base_model.get_label_specification(
        tf.estimator.ModeKeys.TRAIN)
    reward_shape = tuple(label_spec.reward.shape)
    pose_shape = tuple(label_spec.target_pose.shape)
    dummy_reward = np.zeros(reward_shape).astype(np.float32)
    dummy_pose = np.zeros(pose_shape).astype(np.float32)
    return tensorspec_utils.TensorSpecStruct(
        reward=dummy_reward, target_pose=dummy_pose)

  def _select_inference_output(self, predictions):
    """Inference output selection for regression models."""
    # We select our output for inference.
    predictions.condition_output = predictions.full_condition_output.action
    predictions.inference_output = predictions.full_inference_output.action
    return predictions

  def pack_features(self, state, prev_episode_data, timestep):
    """Combines current state and conditioning data into MetaExample spec.

    See create_metaexample_spec for an example of the spec layout.

    If prev_episode_data does not contain enough episodes to fill
      num_condition_samples_per_task, we stuff dummy episodes with reward=0.5
      so that no inner gradients are applied.

    Args:
      state: VRGripperObservation containing image and pose.
      prev_episode_data: A list of episode data, each of which is a list of
        tuples containing transition data. Each transition tuple takes the form
        (obs, action, rew, new_obs, done, debug).
      timestep: Current episode timestep.
    Returns:
      TensorSpecStruct containing conditioning (features, labels)
        and inference (features) keys.
    Raises:
      ValueError: If no demonstration is provided.
    """
    meta_features = tensorspec_utils.TensorSpecStruct()
    meta_features['inference/features/state/0'] = state
    def pack_condition_features(episode_data, idx, dummy_values=False):
      """Pack previous episode data into condition_ep* features/labels.

      Args:
        episode_data: List of (obs, action, rew, new_obs, done, debug) tuples.
        idx: Index of the conditioning episode. 0 for demo, 1 for first trial,
          etc.
        dummy_values: If an episode is not available yet, set the loss_mask
          to 0.

      """
      transition = episode_data[0]
      meta_features['condition/features/state/%d' % idx] = transition[0]
      reward = np.array([transition[2]])
      reward = 2 * reward - 1
      if dummy_values:
        # success_weight of 0. = no gradients in inner loop for this batch.
        reward = np.array([0.])
      meta_features['condition/labels/target_pose/%d' % idx] = transition[1]
      meta_features['condition/labels/reward/%d' % idx] = reward.astype(
          np.float32)

    if prev_episode_data:
      pack_condition_features(prev_episode_data[0], 0)
    else:
      dummy_labels = self._make_dummy_labels()
      dummy_episode = [(state, dummy_labels.target_pose, dummy_labels.reward)]
      pack_condition_features(dummy_episode, 0, dummy_values=True)
    return nest.map_structure(lambda x: np.expand_dims(x, 0), meta_features)
