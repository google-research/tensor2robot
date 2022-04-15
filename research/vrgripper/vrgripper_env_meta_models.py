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

"""T2R Meta-learning models for VRGripper env tasks."""

from typing import Any, Dict, Optional, Tuple

import gin
import numpy as np
from six.moves import range
from tensor2robot.layers import mdn
from tensor2robot.layers import tec
from tensor2robot.layers import vision_layers
from tensor2robot.meta_learning import maml_model
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.meta_learning import preprocessors
from tensor2robot.models import abstract_model
from tensor2robot.research.vrgripper import episode_to_transitions
from tensor2robot.research.vrgripper import vrgripper_env_models
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf

TRAIN = tf_estimator.ModeKeys.TRAIN
PREDICT = tf_estimator.ModeKeys.PREDICT
TensorSpec = tensorspec_utils.ExtendedTensorSpec


def pack_vrgripper_meta_features(
    state,
    prev_episode_data,
    timestep,
    fixed_length,
    num_condition_samples_per_task):
  """Combines current state and conditioning data into MetaExample spec.

  See create_metaexample_spec for an example of the spec layout.

  Args:
    state: VRGripperObservation containing image and pose.
    prev_episode_data: A list of episode data, each of which is a list of
      tuples containing transition data. Each transition tuple takes the form
      (obs, action, rew, new_obs, done, debug).
    timestep: Current episode timestep.
    fixed_length: An int, the fixed length data the model expects.
    num_condition_samples_per_task: An int, the number of conditioning episodes
      given per task.
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
  batch_obs = np.tile(
      state.image, [fixed_length] + len(state.image.shape) * [1])
  batch_gripper = np.tile(
      state.pose, [fixed_length] + len(state.pose.shape) * [1])
  meta_features['inference/features/image/0'] = batch_obs.astype(np.uint8)
  meta_features['inference/features/gripper_pose/0'] = batch_gripper.astype(
      np.float32)

  # state is shape (100, 100, 3). We tile it to match sequences batch dim
  # (fixed_length, 100, 100, 3) and stack across inner batch dim
  # (1, len(prev_episode_data), fixed_length, 100, 100, 3). Then we append an
  # outer batch (always 1 at test time).
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
        episode_data, fixed_length)
    # Condition features.
    batch_obs = np.stack([t[0].image for t in episode_data])
    batch_gripper = np.stack([t[0].pose for t in episode_data])
    meta_features['condition/features/image/%d' % idx] = batch_obs.astype(
        np.uint8)
    meta_features['condition/features/gripper_pose/%d' %
                  idx] = batch_gripper.astype(np.float32)

    # Condition label.
    batch_action = np.stack([t[1] for t in episode_data])

    meta_features['condition/labels/action/%d' %
                  idx] = batch_action.astype(np.float32)

  for i in range(num_condition_samples_per_task):
    pack_condition_features(prev_episode_data[i % len(prev_episode_data)], i)
  # Add meta-dim and type to everything.
  return tf.nest.map_structure(lambda x: np.expand_dims(x, 0), meta_features)


@gin.configurable
class VRGripperEnvRegressionModelMAML(maml_model.MAMLModel):
  """MAML Regression environment for VRGripper."""

  def _select_inference_output(self, predictions):
    """Inference output selection for regression models."""
    # We select our output for inference.
    predictions.condition_output = predictions.full_condition_output.inference_output
    predictions.inference_output = predictions.full_inference_output.inference_output
    return predictions

  def pack_features(self, state, prev_episode_data, timestep):
    return pack_vrgripper_meta_features(
        state,
        prev_episode_data,
        timestep,
        self._base_model._episode_length,  # pylint: disable=protected-access
        self.preprocessor.num_condition_samples_per_task)


@gin.configurable
class VRGripperEnvTecModel(abstract_model.AbstractT2RModel):
  """Task Embedded Control Network: https://arxiv.org/pdf/1810.03237.pdf."""

  def __init__(
      self,
      action_size = 7,
      gripper_pose_size = 14,
      num_waypoints = 1,
      episode_length = 40,
      embed_loss_weight = 0.,
      fc_embed_size = 32,
      ignore_embedding = False,
      action_decoder_cls=mdn.MDNDecoder,
      predict_end_weight = 0.,
      use_film = False,
      **kwargs):
    """Initialize the model.

    Args:
      action_size: The number of action dimensions.
      gripper_pose_size: Size of vector containing gripper state.
      num_waypoints: How many future waypoints to predict.
      episode_length: The fixed length of episode data.
      embed_loss_weight: Weight on embedding loss.
      fc_embed_size: Size of embedding vector that is provided to policy's
        fully connected layers.
      ignore_embedding: If True, the policy does not use the embedding vector.
        Used for debugging.
      action_decoder_cls: Decoder class used to transform actions into a
        distribution.
      predict_end_weight: Weight of end-token prediction loss.
      use_film: If True, applies FILM (https://arxiv.org/abs/1709.07871). FILM
        params come from a linear layer on the TEC embedding.
      **kwargs: Passed to parent.
    """
    super(VRGripperEnvTecModel, self).__init__(**kwargs)
    self._action_size = action_size
    self._gripper_pose_size = gripper_pose_size
    self._num_waypoints = num_waypoints
    self._episode_length = episode_length
    self._embed_loss_weight = embed_loss_weight
    self._fc_embed_size = fc_embed_size
    self._ignore_embedding = ignore_embedding
    self._action_decoder = action_decoder_cls()
    self._predict_end_weight = predict_end_weight
    self._use_film = use_film

  def _episode_feature_specification(
      self, mode):
    """Returns the feature spec for a single episode."""
    del mode
    image_spec = TensorSpec(
        shape=(100, 100, 3), dtype=tf.float32, name='image0',
        data_format='jpeg')
    gripper_pose_spec = TensorSpec(shape=(self._gripper_pose_size,),
                                   dtype=tf.float32, name='world_pose_gripper')
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
        shape=(self._num_waypoints*self._action_size,),
        dtype=tf.float32, name='action_world')
    tspec = tensorspec_utils.TensorSpecStruct(action=action_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  @property
  def preprocessor(self):
    """See base class."""
    base_preprocessor = vrgripper_env_models.DefaultVRGripperPreprocessor(
        model_feature_specification_fn=self._episode_feature_specification,
        model_label_specification_fn=self._episode_label_specification)
    self._preprocessor = preprocessors.FixedLenMetaExamplePreprocessor(
        base_preprocessor=base_preprocessor)
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
    image_embedding = meta_tfdata.multi_batch_apply(
        tec.embed_condition_images, 3,
        episode_data.features.image,
        'image_embedding')
    embedding = meta_tfdata.multi_batch_apply(
        tec.reduce_temporal_embeddings, 2,
        image_embedding, self._fc_embed_size, 'fc_reduce')
    return tf.math.l2_normalize(embedding, axis=-1)

  def inference_network_fn(
      self,
      features,
      labels,
      mode,
      config = None,
      params = None):
    """See base class."""
    condition_embedding = self._embed_episode(features.condition)
    film_output_params = None
    if self._use_film:
      # condition_embedding = [batch, task, embedding]
      film_output_params = meta_tfdata.multi_batch_apply(
          vision_layers.BuildFILMParams, 2,
          condition_embedding)
      # Need to stretch to [batch, task, T, film_size] since later call expects
      # 3 batch dimensions. FILM params are identical across time but change for
      # different conditioning episodes.
      film_output_params = tf.expand_dims(film_output_params, axis=-2)
      film_output_params = tf.tile(
          film_output_params, [1, 1, self._episode_length, 1])

    gripper_pose = features.inference.features.gripper_pose
    fc_embedding = tf.tile(
        condition_embedding[Ellipsis, :self._fc_embed_size][:, :, None, :],
        [1, 1, self._episode_length, 1])
    with tf.variable_scope(
        'state_features', reuse=tf.AUTO_REUSE, use_resource=True):
      state_features, _ = meta_tfdata.multi_batch_apply(
          vision_layers.BuildImagesToFeaturesModel, 3,
          features.inference.features.image,
          film_output_params=film_output_params)
    if self._ignore_embedding:
      fc_inputs = tf.concat([state_features, gripper_pose], -1)
    else:
      fc_inputs = tf.concat([state_features, gripper_pose, fc_embedding], -1)

    outputs = {}
    aux_output_dim = 0
    # We only predict end token for next step.
    if self._predict_end_weight > 0:
      aux_output_dim = 1
    with tf.variable_scope('a_func', reuse=tf.AUTO_REUSE, use_resource=True):
      action_params, end_token = meta_tfdata.multi_batch_apply(
          vision_layers.BuildImageFeaturesToPoseModel,
          3, fc_inputs, num_outputs=None, aux_output_dim=aux_output_dim)
      action = self._action_decoder(
          params=action_params,
          output_size=self._num_waypoints*self._action_size)

    outputs.update({
        'inference_output': action,  # used for policy.
        'condition_embedding': condition_embedding,
    })

    if self._predict_end_weight > 0:
      outputs['end_token_logits'] = end_token
      outputs['end_token'] = tf.nn.sigmoid(end_token)
      outputs['inference_output'] = tf.concat(
          [outputs['inference_output'], outputs['end_token']], -1)

    if mode != PREDICT:
      # During training we embed the inference episodes to compute the triplet
      # loss between condition/inference embeddings.
      inference_embedding = self._embed_episode(features.inference)
      outputs['inference_embedding'] = inference_embedding
    return outputs

  def _compute_end_loss(self, inference_outputs, labels):
    # MetaTidyTec and VRGripper differ in how end loss is computed, so this
    # function refactors it out.
    end_loss = 0.0
    if self._predict_end_weight > 0:
      # Label last two states as end states to add more signal.
      zero_labels = tf.zeros_like(
          inference_outputs['end_token_logits'])[:, :, :-2, :]
      one_labels = tf.ones_like(
          inference_outputs['end_token_logits'])[:, :, -2:, :]
      end_labels = tf.concat([zero_labels, one_labels], 2)
      end_loss = tf.losses.sigmoid_cross_entropy(
          multi_class_labels=end_labels,
          logits=inference_outputs['end_token_logits'])
    return end_loss

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
    bc_loss = self._action_decoder.loss(labels)
    bc_loss = tf.identity(bc_loss, name='bc_loss')
    embed_loss = tec.compute_embedding_contrastive_loss(
        inference_outputs['inference_embedding'],
        inference_outputs['condition_embedding'])
    end_loss = self._compute_end_loss(inference_outputs, labels)
    train_outputs = {'bc_loss': bc_loss, 'embed_loss': embed_loss,
                     'end_loss': end_loss}
    return (bc_loss + self._embed_loss_weight * embed_loss +
            self._predict_end_weight * end_loss, train_outputs)  # pytype: disable=bad-return-type

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
    if self.use_summaries(params) and train_outputs is not None:
      eval_outputs = {}
      for key, value in train_outputs.items():
        eval_outputs[key] = tf.metrics.mean(value)
      return eval_outputs

  def add_summaries(self,
                    features,
                    labels,
                    inference_outputs,
                    train_loss,
                    train_outputs,
                    mode,
                    config=None,
                    params=None):
    if not self.use_summaries(params):
      return
    if mode != PREDICT:
      for key in ['bc_loss', 'embed_loss']:
        tf.summary.scalar(key, train_outputs[key])
      if self._predict_end_weight > 0:
        tf.summary.scalar('end_loss', train_outputs['end_loss'])
    # Marginal distribution (over batch, timesteps) over each action dim.
    pose = inference_outputs['inference_output']
    for i, key in enumerate(
        ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper_close']):
      tf.summary.histogram('estimated_pose/%s' % key, pose[Ellipsis, i])
    if self._predict_end_weight > 0:
      tf.summary.histogram('estimated_pose/end_weight', pose[Ellipsis, -1])

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
    return pack_vrgripper_meta_features(
        state,
        prev_episode_data,
        timestep,
        self._episode_length,
        self.preprocessor.num_condition_samples_per_task)



