# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""T2RModels for duck pose prediction task."""

from typing import Optional, Tuple, Text
from absl import logging
import gin
import numpy as np
from six.moves import range
from tensor2robot.layers import vision_layers
from tensor2robot.models import abstract_model
from tensor2robot.models import critic_model
from tensor2robot.models import regression_model
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.research.dql_grasping_lib import tf_modules
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf
import tf_slim as slim
from tensorflow.keras import layers

TensorSpec = tensorspec_utils.ExtendedTensorSpec  # pylint: disable=invalid-name

TRAIN = tf_estimator.ModeKeys.TRAIN
PREDICT = tf_estimator.ModeKeys.PREDICT


class DefaultPoseEnvContinuousPreprocessor(
    abstract_preprocessor.AbstractPreprocessor):
  """The default pose env preprocessor.

  This preprocessor simply converts all images from uint8 to float32.
  """

  def get_in_feature_specification(self, mode):
    """See base class."""
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec['state/image'] = TensorSpec(
        shape=self._model_feature_specification_fn(mode).state.image.shape,
        dtype=tf.uint8,
        name=self._model_feature_specification_fn(mode).state.image.name,
        data_format=self._model_feature_specification_fn(
            mode).state.image.data_format)
    feature_spec['action/pose'] = self._model_feature_specification_fn(
        mode).action.pose
    return feature_spec

  def get_in_label_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def get_out_feature_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))

  def get_out_label_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def _preprocess_fn(
      self,
      features,
      labels,
      mode,
  ):
    """See base class."""
    features.state.image = tf.image.convert_image_dtype(features.state.image,
                                                        tf.float32)
    return features, labels


@gin.configurable
class PoseEnvContinuousMCModel(critic_model.CriticModel):
  """Continuous MC model for Pose Env."""

  def get_action_specification(self):
    pose_spec = TensorSpec(shape=(2,), dtype=tf.float32, name='pose')
    return tensorspec_utils.TensorSpecStruct(
        pose=pose_spec)

  def get_state_specification(self):
    image_spec = TensorSpec(
        shape=(64, 64, 3), dtype=tf.float32, name='state/image',
        data_format='jpeg')
    return tensorspec_utils.TensorSpecStruct(
        image=image_spec)

  @property
  def default_preprocessor_cls(self):
    return DefaultPoseEnvContinuousPreprocessor

  def get_label_specification(self, mode):
    del mode
    reward_spec = TensorSpec(shape=(), dtype=tf.float32, name='reward')
    return tensorspec_utils.TensorSpecStruct(reward=reward_spec)

  def _q_features(self, state, action, is_training=True, reuse=tf.AUTO_REUSE):
    """Compute feature representation of state and action.

    This method is re-used for computing partial context features for RL^2
    algorithm.

    Args:
      state: Image observation tensor.
      action: Continuous-valued 1-D action tensor.
      is_training: Bool specifying whether this graph is being constructed for
        training or not.
      reuse: Bool or TF re-use ENUM.

    Returns:
      Q values of shape (num_batch, feature_size) where feature_size is
        h * w * 32.
    """
    # State-action embedding module.
    net = state
    channels = 32
    with tf.variable_scope('q_features', reuse=reuse):
      with slim.arg_scope(tf_modules.argscope(is_training=is_training)):
        for layer_index in range(3):
          net = layers.conv2d(net, channels, kernel_size=3)
          logging.info('conv%d %s', layer_index, net.get_shape())
      action_context = layers.fully_connected(action, channels)
      _, h, w, _ = net.shape.as_list()
      num_batch_net = tf.shape(net)[0]
      num_batch_context = tf.shape(action_context)[0]
      # assume num_batch_context >= num_batch_net
      net = tf.tile(net, [num_batch_context // num_batch_net, 1, 1, 1])
      action_context = tf.reshape(action_context,
                                  [num_batch_context, 1, 1, channels])
      action_context = tf.tile(action_context, [1, h, w, 1])
      net += action_context
      net = tf.layers.flatten(net)
    return net

  def q_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None,
             reuse=tf.AUTO_REUSE):
    del params
    is_training = mode == TRAIN
    with tf.variable_scope(scope, reuse=reuse, use_resource=True):
      image = tf.image.convert_image_dtype(features.state.image, tf.float32)
      net = self._q_features(
          image, features.action.pose, is_training=is_training, reuse=reuse)
      net = layers.stack(net, layers.fully_connected, [100, 100])
      net = layers.fully_connected(
          net,
          num_outputs=1,
          normalizer_fn=None,
          weights_regularizer=None,
          activation_fn=None)
      return {'q_predicted': tf.squeeze(net, 1)}

  def pack_features(self, state, context, timestep, actions):
    del context, timestep
    batch_obs = np.expand_dims(state, 0)
    return tensorspec_utils.TensorSpecStruct(
        state=batch_obs, action=actions)


class DefaultPoseEnvRegressionPreprocessor(
    abstract_preprocessor.AbstractPreprocessor):
  """The default pose env preprocessor.

  This preprocessor simply converts all images from uint8 to float32.
  """

  def get_in_feature_specification(
      self, mode):
    """See base class."""
    feature_spec = tensorspec_utils.TensorSpecStruct()
    feature_spec['state'] = TensorSpec(
        shape=self._model_feature_specification_fn(mode).state.shape,
        dtype=tf.uint8,
        name=self._model_feature_specification_fn(mode).state.name,
        data_format=self._model_feature_specification_fn(
            mode).state.data_format)
    return feature_spec

  def get_in_label_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def get_out_feature_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))

  def get_out_label_specification(
      self, mode):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def _preprocess_fn(
      self, features,
      labels,
      mode
  ):
    """See base class."""
    features.state = tf.image.convert_image_dtype(features.state, tf.float32)
    return features, labels


@gin.configurable
class PoseEnvRegressionModel(regression_model.RegressionModel):
  """Continuous regression output model for Pose Env."""

  @property
  def default_preprocessor_cls(self):
    return DefaultPoseEnvRegressionPreprocessor

  def get_feature_specification(self, mode):
    del mode
    state_spec = TensorSpec(
        shape=(64, 64, 3), dtype=tf.float32, name='state/image',
        data_format='jpeg')
    return tensorspec_utils.TensorSpecStruct(state=state_spec)

  def get_label_specification(self, mode):
    del mode
    target_spec = TensorSpec(
        shape=(self._action_size), dtype=tf.float32, name='target_pose')
    reward_spec = TensorSpec(shape=(1), dtype=tf.float32, name='reward')
    return tensorspec_utils.TensorSpecStruct(
        target_pose=target_spec, reward=reward_spec)

  def pack_features(self, state, context, timestep):
    del context, timestep
    batch_obs = np.expand_dims(state, 0)
    return tensorspec_utils.TensorSpecStruct(state=batch_obs)

  @property
  def action_size(self):
    return self._action_size

  def get_config(self):
    """This model trains fairly quickly so evaluate frequently."""
    return tf_estimator.RunConfig(
        save_checkpoints_steps=2000,
        keep_checkpoint_max=5)

  def a_func(
      self,
      features,
      scope,
      mode,
      config = None,
      params = None,
      reuse=tf.AUTO_REUSE,
      context_fn=None,
  ):
    """A (state) regression function.

    This function can return a stochastic or a deterministic tensor.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_spefication.
      scope: String specifying variable scope.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: Optional configuration object. Will receive what is passed to
        Estimator in config parameter, or the default config. Allows updating
        things in your model_fn based on configuration such as num_ps_replicas,
        or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.
      reuse: Whether or not to reuse variables under variable scope 'scope'.
      context_fn: Optional python function that takes in features and returns
        new features of same shape. For merging information like in RL^2.

    Returns:
      outputs: A {key: Tensor} mapping. The key 'action' is required.
    """
    del config
    is_training = mode == TRAIN
    image = tf.image.convert_image_dtype(features.state, tf.float32)
    with tf.variable_scope(scope, reuse=reuse, use_resource=True):
      with tf.variable_scope('state_features', reuse=reuse, use_resource=True):
        feature_points, end_points = vision_layers.BuildImagesToFeaturesModel(
            image,
            is_training=is_training,
            normalizer_fn=slim.layer_norm)
      del end_points
      if context_fn:
        feature_points = context_fn(feature_points)
      estimated_pose, _ = vision_layers.BuildImageFeaturesToPoseModel(
          feature_points, num_outputs=self._action_size)
    return {'inference_output': estimated_pose,
            'state_features': feature_points}

  def loss_fn(self, labels, inference_outputs, mode, params=None):
    del mode
    return tf.losses.mean_squared_error(
        labels=labels.target_pose,
        predictions=inference_outputs['inference_output'],
        weights=labels.reward)
