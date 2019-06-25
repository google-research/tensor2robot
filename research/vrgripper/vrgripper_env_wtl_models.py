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

"""T2R models for the Watch, Try, Learn experiments: https://arxiv.org/abs/1906.03352."""  # pylint: disable=line-too-long
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np
from tensor2robot.layers import mdn
from tensor2robot.layers import tec
from tensor2robot.layers import vision_layers
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.meta_learning import preprocessors
from tensor2robot.models import abstract_model
from tensor2robot.research.vrgripper import episode_to_transitions
from tensor2robot.research.vrgripper import vrgripper_env_models
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf  # tf
from typing import Any, Dict, Optional, Text, Tuple

TRAIN = tf.estimator.ModeKeys.TRAIN
PREDICT = tf.estimator.ModeKeys.PREDICT
TensorSpec = tensorspec_utils.ExtendedTensorSpec


@gin.configurable
def pack_wtl_meta_features(
    state,
    prev_episode_data,
    timestep,
    fixed_length,
    num_condition_samples_per_task,
    vision=False,
    deterministic_condition=True):
  """Combines current state and conditioning data into MetaExample spec.

  Args:
    state: VRGripperObservation containing image and pose.
    prev_episode_data: A list of episode data, each of which is a list of
      tuples containing transition data. Each transition tuple takes the form
      (obs, action, rew, new_obs, done, debug).
    timestep: Current episode timestep.
    fixed_length: An int, the fixed length data the model expects.
    num_condition_samples_per_task: An int, the number of conditioning episodes
      given per task.
    vision: If True, assume a vision model. Otherwise, assume it's a low-dim
      state model.
    deterministic_condition: If True, conditioning episodes are cut to fixed
      length in a deterministic manner.
  Returns:
    TensorSpecStruct containing conditioning (features, labels)
      and inference (features) keys.
  Raises:
    ValueError: If no demonstration is provided.
  """
  del timestep
  if len(prev_episode_data) < 1:
    raise ValueError(
        'prev_episode_data should at least contain one (demo) episode.')
  meta_features = tensorspec_utils.TensorSpecStruct()

  # Inference features (tiled sequence dim).
  if vision:
    # state is shape (100, 100, 3). We tile it to match sequences batch dim
    # (fixed_length, 100, 100, 3) and stack across inner batch dim
    # (1, len(prev_episode_data), fixed_length, 100, 100, 3). Then we append an
    # outer batch (always 1 at test time).
    batch_obs = np.tile(
        state.image, [fixed_length] + len(state.image.shape) * [1])
    batch_gripper = np.tile(
        state.pose, [fixed_length] + len(state.pose.shape) * [1])
    meta_features['inference/features/image/0'] = batch_obs.astype(np.uint8)
    meta_features['inference/features/gripper_pose/0'] = batch_gripper.astype(
        np.float32)
  else:
    batch_full_state = np.tile(
        state.full_state_pose,
        [fixed_length] + len(state.full_state_pose.shape) * [1])
    meta_features['inference/features/full_state_pose/0'] = (
        batch_full_state.astype(np.float32))
  def pack_condition_features(episode_data, idx):
    """Pack previous episode data into condition_ep* features/labels.

    Args:
      episode_data: List of (obs, action, rew, new_obs, done, debug) tuples.
      idx: Index of the conditioning episode. 0 for demo, 1 for first trial,
        etc.
    """
    # Conditioning Context (The assumption is that policy is always adapting
    # from some conditioning data, whether it is demo and/or trials.
    episode_data = episode_to_transitions.make_fixed_length(
        episode_data, fixed_length, randomized=not deterministic_condition)
    # Condition features.
    if vision:
      batch_obs = np.stack([t[0].image for t in episode_data])
      batch_gripper = np.stack([t[0].pose for t in episode_data])
      meta_features['condition/features/image/%d' % idx] = batch_obs.astype(
          np.uint8)
      meta_features['condition/features/gripper_pose/%d' %
                    idx] = batch_gripper.astype(np.float32)
    else:
      batch_full_state = np.stack([t[0].full_state_pose for t in episode_data])
      meta_features['condition/features/full_state_pose/%d' %
                    idx] = batch_full_state.astype(np.float32)
    # Condition label.
    batch_action = np.stack([t[1] for t in episode_data])
    meta_features['condition/labels/action/%d' % idx] = batch_action.astype(
        np.float32)

    cumulative_return = np.sum([t[2] for t in episode_data])
    success = float(cumulative_return > 0) * np.ones((40, 1), dtype=np.float32)
    meta_features['condition/labels/success/%d' % idx] = success

  for i in range(num_condition_samples_per_task):
    pack_condition_features(prev_episode_data[i % len(prev_episode_data)], i)
  # Add meta-dim and type to everything.
  return tf.contrib.framework.nest.map_structure(lambda x: np.expand_dims(x, 0), meta_features)


@gin.configurable
class VRGripperEnvSimpleTrialModel(abstract_model.AbstractT2RModel):
  """Simple model for trial that uses last frame of condition demo."""

  def __init__(
      self,
      action_size = 7,
      episode_length = 40,
      fc_embed_size = 32,
      ignore_embedding = False,
      num_mixture_components = 1,
      num_condition_samples_per_task = 1,
      retrial = False,
      embed_type = 'temporal',
      save_checkpoint_steps = 30000,
      **kwargs):
    """Initialize the model."""
    super(VRGripperEnvSimpleTrialModel, self).__init__(**kwargs)
    self._action_size = action_size
    self._episode_length = episode_length
    self._fc_embed_size = fc_embed_size
    self._ignore_embedding = ignore_embedding
    self._num_mixture_components = num_mixture_components
    self._obs_size = 32
    self._retrial = retrial
    self._num_condition_samples_per_task = num_condition_samples_per_task
    self._embed_type = embed_type
    self._save_checkpoint_steps = save_checkpoint_steps

  def _episode_feature_specification(
      self, mode):
    """Returns the feature spec for a single episode."""
    del mode
    full_state_pose_spec = TensorSpec(
        shape=(self._obs_size,), dtype=tf.float32, name='full_state_pose')
    spec = tensorspec_utils.TensorSpecStruct(
        full_state_pose=full_state_pose_spec)
    spec = tensorspec_utils.copy_tensorspec(
        spec, batch_size=self._episode_length)
    return spec

  def _episode_label_specification(
      self, mode):
    """Returns the label spec for a single episode."""
    del mode
    action_spec = TensorSpec(
        shape=(self._action_size,), dtype=tf.float32, name='action_world')
    success_spec = TensorSpec(shape=(1,), dtype=tf.float32, name='success')
    tspec = tensorspec_utils.TensorSpecStruct(
        action=action_spec, success=success_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  @property
  def preprocessor(self):
    """See base class."""
    base_preprocessor = vrgripper_env_models.DefaultVRGripperPreprocessor(
        model_feature_specification_fn=self._episode_feature_specification,
        model_label_specification_fn=self._episode_label_specification)
    self._preprocessor = preprocessors.FixedLenMetaExamplePreprocessor(
        base_preprocessor=base_preprocessor,
        num_condition_samples_per_task=self._num_condition_samples_per_task)
    return self._preprocessor

  def get_feature_specification(
      self, mode):
    """See base class."""
    return preprocessors.create_maml_feature_spec(
        self._episode_feature_specification(mode),
        self._episode_label_specification(mode))

  def get_label_specification(
      self, mode):
    """See base class."""
    return preprocessors.create_maml_label_spec(
        self._episode_label_specification(mode))

  def inference_network_fn(
      self,
      features,
      labels,
      mode,
      config = None,
      params = None):
    """See base class."""
    inf_full_state_pose = features.inference.features.full_state_pose
    con_full_state_pose = features.condition.features.full_state_pose
    # Map success labels [0, 1] -> [-1, 1]
    con_success = 2 * features.condition.labels.success - 1
    if self._retrial and con_full_state_pose.shape[1] != 2:
      raise ValueError('Unexpected shape {}.'.format(con_full_state_pose.shape))
    if self._embed_type == 'temporal':
      fc_embedding = meta_tfdata.multi_batch_apply(
          tec.reduce_temporal_embeddings, 2,
          con_full_state_pose[:, 0:1, :, :],
          self._fc_embed_size, 'demo_embedding')[:, :, None, :]
    elif self._embed_type == 'mean':
      fc_embedding = con_full_state_pose[:, 0:1, -1:, :]
    else:
      raise ValueError('Invalid embed_type: {}.'.format(self._embed_type))
    fc_embedding = tf.tile(fc_embedding, [1, 1, 40, 1])
    if self._retrial:
      con_input = tf.concat([
          con_full_state_pose[:, 1:2, :, :],
          con_success[:, 1:2, :, :],
          fc_embedding], -1)
      if self._embed_type == 'mean':
        trial_embedding = meta_tfdata.multi_batch_apply(
            tec.embed_fullstate, 3, con_input,
            self._fc_embed_size, 'trial_embedding')
        trial_embedding = tf.reduce_mean(trial_embedding, -2)
      else:
        trial_embedding = meta_tfdata.multi_batch_apply(
            tec.reduce_temporal_embeddings, 2,
            con_input, self._fc_embed_size, 'trial_embedding')
      trial_embedding = tf.tile(trial_embedding[:, :, None, :], [1, 1, 40, 1])
      fc_embedding = tf.concat([fc_embedding, trial_embedding], -1)
    if self._ignore_embedding:
      fc_inputs = inf_full_state_pose
    else:
      fc_inputs = [inf_full_state_pose, fc_embedding]
      if self._retrial:
        fc_inputs.append(con_success[:, 1:2, :, :])
      fc_inputs = tf.concat(fc_inputs, -1)
    outputs = {}
    with tf.variable_scope('a_func', reuse=tf.AUTO_REUSE, use_resource=True):
      if self._num_mixture_components > 1:
        fc_inputs, _ = meta_tfdata.multi_batch_apply(
            vision_layers.BuildImageFeaturesToPoseModel, 3, fc_inputs,
            num_outputs=None)
        dist_params = meta_tfdata.multi_batch_apply(
            mdn.predict_mdn_params, 3,
            fc_inputs, self._num_mixture_components,
            self._action_size, False)
        outputs['dist_params'] = dist_params
        gm = mdn.get_mixture_distribution(
            dist_params,
            self._num_mixture_components,
            self._action_size)
        action = mdn.gaussian_mixture_approximate_mode(gm)
      else:
        action, _ = meta_tfdata.multi_batch_apply(
            vision_layers.BuildImageFeaturesToPoseModel,
            3, fc_inputs, self._action_size)

    outputs.update({
        'inference_output': action,
    })

    return outputs

  def model_train_fn(
      self,
      features,
      labels,
      inference_outputs,
      mode,
      config = None,
      params = None
  ):
    """Returns weighted sum of losses and individual losses. See base class."""
    if self._num_mixture_components > 1:
      gm = mdn.get_mixture_distribution(
          inference_outputs['dist_params'],
          self._num_mixture_components,
          self._action_size)
      bc_loss = -tf.reduce_mean(gm.log_prob(labels.action))
    else:
      bc_loss = tf.losses.mean_squared_error(
          labels=labels.action,
          predictions=inference_outputs['inference_output'])
    if mode == TRAIN and self.use_summaries(params):
      tf.summary.scalar('bc_loss', bc_loss)
    return bc_loss

  def model_eval_fn(
      self,
      features,
      labels,
      inference_outputs,
      train_loss,
      train_outputs,
      mode,
      config = None,
      params = None):
    """Log the streaming mean of any train outputs. See also base class."""
    if train_outputs is not None:
      eval_outputs = {}
      for key, value in train_outputs.items():
        eval_outputs['mean_' + key] = tf.metrics.mean(value)
      return eval_outputs

  def get_run_config(self):
    return tf.estimator.RunConfig(
        save_checkpoints_steps=self._save_checkpoint_steps)

  def pack_features(
      self, state, prev_episode_data, timestep
  ):
    """Combine current state and previous episode data into a MetaExample spec.

    Args:
      state: VRGripperObservation containing image and pose.
      prev_episode_data: A list of episode data, each of which is a list of
        tuples containing transition data. Each transition tuple takes the form
        (obs, action, rew, new_obs, done, debug).
      timestep: Current episode timestep.
    Returns:
      TensorSpecStruct containing conditioning (features, labels)
        and inference (features) keys.
    """
    return pack_wtl_meta_features(
        state,
        prev_episode_data,
        timestep,
        self._episode_length,
        self.preprocessor.num_condition_samples_per_task)


@gin.configurable
class VRGripperEnvVisionTrialModel(abstract_model.AbstractT2RModel):
  """Task Embedded Control Network: https://arxiv.org/pdf/1810.03237.pdf."""

  def __init__(
      self,
      action_size = 7,
      episode_length = 40,
      embed_loss_weight = 0.,
      fc_embed_size = 32,
      ignore_embedding = False,
      num_mixture_components = 1,
      num_condition_samples_per_task = 1,
      save_checkpoint_steps = 2000,
      **kwargs):
    """Initialize the model."""
    super(VRGripperEnvVisionTrialModel, self).__init__(**kwargs)
    self._action_size = action_size
    self._episode_length = episode_length
    self._embed_loss_weight = embed_loss_weight
    self._fc_embed_size = fc_embed_size
    self._ignore_embedding = ignore_embedding
    self._num_mixture_components = num_mixture_components
    self._num_condition_samples_per_task = num_condition_samples_per_task
    self._save_checkpoint_steps = save_checkpoint_steps

  def _episode_feature_specification(
      self, mode):
    """Returns the feature spec for a single episode."""
    del mode
    image_spec = TensorSpec(
        shape=(100, 100, 3), dtype=tf.float32, name='image0',
        data_format='jpeg')
    gripper_pose_spec = TensorSpec(
        shape=(14,), dtype=tf.float32, name='world_pose_gripper')
    tspec = tensorspec_utils.TensorSpecStruct(
        image=image_spec,
        gripper_pose=gripper_pose_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  def _episode_label_specification(
      self, mode):
    """Returns the label spec for a single episode."""
    del mode
    action_spec = TensorSpec(
        shape=(self._action_size,), dtype=tf.float32, name='action_world')
    success_spec = TensorSpec(shape=(1,), dtype=tf.float32, name='success')
    tspec = tensorspec_utils.TensorSpecStruct(
        action=action_spec, success=success_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  @property
  def preprocessor(self):
    """See base class."""
    base_preprocessor = vrgripper_env_models.DefaultVRGripperPreprocessor(
        model_feature_specification_fn=self._episode_feature_specification,
        model_label_specification_fn=self._episode_label_specification)
    self._preprocessor = preprocessors.FixedLenMetaExamplePreprocessor(
        base_preprocessor=base_preprocessor,
        num_condition_samples_per_task=self._num_condition_samples_per_task)
    return self._preprocessor

  def get_feature_specification(
      self, mode):
    """See base class."""
    return preprocessors.create_maml_feature_spec(
        self._episode_feature_specification(mode),
        self._episode_label_specification(mode))

  def get_label_specification(
      self, mode):
    """See base class."""
    return preprocessors.create_maml_label_spec(
        self._episode_label_specification(mode))

  def _embed_episode(
      self, episode_data):
    """Produces embeddings from episode data."""
    demo_fp = meta_tfdata.multi_batch_apply(
        tec.embed_condition_images, 3,
        episode_data.features.image[:, 0:1, :, :],
        'image_embedding')
    demo_inputs = tf.concat(
        [demo_fp, episode_data.features.gripper_pose[:, 0:1, :, :]], -1)
    embedding = meta_tfdata.multi_batch_apply(
        tec.reduce_temporal_embeddings, 2,
        demo_inputs, self._fc_embed_size, 'fc_demo_reduce')
    if self._num_condition_samples_per_task > 1:
      con_success = 2 * episode_data.labels.success - 1
      trial_embedding = meta_tfdata.multi_batch_apply(
          tec.embed_condition_images, 3,
          episode_data.features.image[:, 1:2, :, :],
          'image_embedding')
      trial_embedding = tf.concat(
          [trial_embedding,
           episode_data.features.gripper_pose[:, 1:2, :, :],
           con_success[:, 1:2, :, :],
           tf.tile(embedding[:, :, None, :], [1, 1, 40, 1])], -1)
      trial_embedding = meta_tfdata.multi_batch_apply(
          tec.reduce_temporal_embeddings, 2,
          trial_embedding, self._fc_embed_size, 'fc_trial_reduce')
      embedding = tf.concat([embedding, trial_embedding], axis=-1)
    return embedding

  def inference_network_fn(
      self,
      features,
      labels,
      mode,
      config = None,
      params = None):
    """See base class."""
    condition_embedding = self._embed_episode(features.condition)
    gripper_pose = features.inference.features.gripper_pose
    fc_embedding = tf.tile(
        condition_embedding[:, :, None, :],
        [1, 1, self._episode_length, 1])
    with tf.variable_scope(
        'state_features', reuse=tf.AUTO_REUSE, use_resource=True):
      state_features, _ = meta_tfdata.multi_batch_apply(
          vision_layers.BuildImagesToFeaturesModel, 3,
          features.inference.features.image)
    if self._ignore_embedding:
      fc_inputs = tf.concat([state_features, gripper_pose], -1)
    else:
      fc_inputs = tf.concat([state_features, gripper_pose, fc_embedding], -1)

    outputs = {}
    with tf.variable_scope('a_func', reuse=tf.AUTO_REUSE, use_resource=True):
      if self._num_mixture_components > 1:
        dist_params = meta_tfdata.multi_batch_apply(
            mdn.predict_mdn_params, 3,
            fc_inputs, self._num_mixture_components,
            self._action_size, False)
        outputs['dist_params'] = dist_params
        gm = mdn.get_mixture_distribution(
            dist_params,
            self._num_mixture_components,
            self._action_size)
        action = mdn.gaussian_mixture_approximate_mode(gm)
      else:
        action, _ = meta_tfdata.multi_batch_apply(
            vision_layers.BuildImageFeaturesToPoseModel,
            3, fc_inputs, self._action_size)
    outputs['inference_output'] = action
    return outputs

  def model_train_fn(
      self,
      features,
      labels,
      inference_outputs,
      mode,
      config = None,
      params = None
  ):
    """Returns weighted sum of losses and individual losses. See base class."""
    if self._num_mixture_components > 1:
      gm = mdn.get_mixture_distribution(
          inference_outputs['dist_params'],
          self._num_mixture_components,
          self._action_size)
      bc_loss = -tf.reduce_mean(gm.log_prob(labels.action))
    else:
      bc_loss = tf.losses.mean_squared_error(
          labels=labels.action,
          predictions=inference_outputs['inference_output'])
    train_outputs = {'bc_loss': bc_loss}
    if mode == TRAIN and self.use_summaries(params):
      tf.summary.scalar('bc_loss', bc_loss)
    return bc_loss, train_outputs

  def model_eval_fn(
      self,
      features,
      labels,
      inference_outputs,
      train_loss,
      train_outputs,
      mode,
      config = None,
      params = None):
    """Log the streaming mean of any train outputs. See also base class."""
    if train_outputs is not None:
      eval_outputs = {}
      for key, value in train_outputs.items():
        eval_outputs['mean_' + key] = tf.metrics.mean(value)
      return eval_outputs

  def get_run_config(self):
    return tf.estimator.RunConfig(
        save_checkpoints_steps=self._save_checkpoint_steps)

  def pack_features(
      self, state, prev_episode_data, timestep
  ):
    """Combine current state and previous episode data into a MetaExample spec.

    Args:
      state: VRGripperObservation containing image and pose.
      prev_episode_data: A list of episode data, each of which is a list of
        tuples containing transition data. Each transition tuple takes the form
        (obs, action, rew, new_obs, done, debug).
      timestep: Current episode timestep.
    Returns:
      TensorSpecStruct containing conditioning (features, labels)
        and inference (features) keys.
    """
    return pack_wtl_meta_features(
        state,
        prev_episode_data,
        timestep,
        self._episode_length,
        self.preprocessor.num_condition_samples_per_task,
        vision=True)
