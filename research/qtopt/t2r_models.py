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

"""T2R Model for QT-Opt. Reference OSS implementation ONLY."""

import abc
from typing import Optional, Tuple, Text

from absl import logging
import gin
import six
from tensor2robot.models import abstract_model
from tensor2robot.models import critic_model
# image_transformations code has some small discrepancies from the distortion
# parameters used in the QT-Opt paper.
from tensor2robot.preprocessors import image_transformations
from tensor2robot.preprocessors import spec_transformation_preprocessor
from tensor2robot.research.qtopt import networks
from tensor2robot.research.qtopt import optimizer_builder
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import training as contrib_training
from tensorflow_estimator.contrib.estimator.python.estimator import replicate_model_fn

TRAIN = tf_estimator.ModeKeys.TRAIN
EVAL = tf_estimator.ModeKeys.EVAL
PREDICT = tf_estimator.ModeKeys.PREDICT
INPUT_SHAPE = (512, 640, 3)
TARGET_SHAPE = (472, 472)


def pack_features_kuka_e2e(tf_model, *policy_inputs):
  """Crop, Convert, Maybe Distort images.

  Args:
    tf_model: Model that is converting policy inputs to features..
    *policy_inputs: Inputs to the policy

  Returns:
    features: preprocessed features.
  """
  del tf_model, policy_inputs
  raise NotImplementedError


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class LegacyGraspingModelWrapper(critic_model.CriticModel):
  """T2R wrapper class around grasping network definitions."""

  def __init__(self,
               loss_function=tf.losses.log_loss,
               learning_rate=1e-4,
               model_weights_averaging=.9999,
               momentum=.9,
               export_batch_size=1,
               use_avg_model_params=True,
               learning_rate_decay_factor=.999,
               **kwargs):
    """Constructor."""

    # Set the default value for hparams if the user doesn't specificy
    # them.
    default_hparams = dict(
        batch_size=32,
        examples_per_epoch=3000000,
        learning_rate_decay_factor=learning_rate_decay_factor,
        learning_rate=learning_rate,
        model_weights_averaging=model_weights_averaging,
        momentum=momentum,
        num_epochs_per_decay=2.0,
        optimizer='momentum',
        rmsprop_decay=.9,
        rmsprop_epsilon=1.0,
        use_avg_model_params=use_avg_model_params)

    self.hparams = contrib_training.HParams(**default_hparams)
    self._export_batch_size = export_batch_size
    self.kwargs = kwargs
    super(LegacyGraspingModelWrapper, self).__init__(
        loss_function=loss_function,
        create_optimizer_fn=lambda _: optimizer_builder.BuildOpt(self.hparams),
        action_batch_size=kwargs.get('action_batch_size'),
        use_avg_model_params=use_avg_model_params)

  @abc.abstractproperty
  def legacy_model_class(self):
    pass

  def create_legacy_model(self):
    class_ = self.legacy_model_class
    return class_(**self.kwargs)

  @abc.abstractmethod
  def pack_features(self, *policy_inputs):
    pass

  def get_trainable_variables(self):
    """Returns list of trainable model variables."""
    return contrib_framework.get_trainable_variables(
        self.legacy_model_class.__name__)

  def get_variables(self):
    """Returns list of model variables."""
    return contrib_framework.get_variables(self.legacy_model_class.__name__)

  def get_label_specification(self, mode):
    del mode
    grasp_success_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='grasp_success')
    return tensorspec_utils.TensorSpecStruct(
        reward=grasp_success_spec)

  def create_legacy_input_specification(self):
    """Compatibility method needed by cem policies."""
    return self.legacy_model_class.create_input_specifications()

  def get_global_step(self):
    # tf.train.get_global_step() does not work well under model_fn for TPU.
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      return tf.broadcast_to(
          tf.get_variable('global_step', shape=[], dtype=tf.int64),
          shape=(self._export_batch_size,))

  def q_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None,
             reuse=tf.AUTO_REUSE):
    """See base class."""
    images = [features.state.image, features.state.image_1]
    grasp_params = tf.concat(
        [features.action.world_vector, features.action.vertical_rotation],
        axis=1)
    model = self.create_legacy_model()
    is_training = mode == TRAIN
    _, end_points = model.model(images, grasp_params, is_training=is_training)
    # Return sigmoid(logits).
    return {
        'q_predicted': end_points['predictions'],
        'global_step': self.get_global_step()
    }

  def create_optimizer(self, params):
    """Create the optimizer and scaffold used for training."""
    config = self.get_run_config()
    original_optimizer = self._create_optimizer_fn(self.use_summaries(params))

    # Override self.scaffold_fn with a custom scaffold_fn that uses the
    # swapping saver required for MovingAverageOptimizer.
    use_avg_model_params = self.hparams.use_avg_model_params

    def scaffold_fn():
      """Create a scaffold object."""
      # MovingAverageOptimizer requires Swapping Saver.
      scaffold = tf.train.Scaffold()
      if use_avg_model_params:
        saver = original_optimizer.swapping_saver(
            keep_checkpoint_every_n_hours=1)
      else:
        saver = None
      scaffold = tf.train.Scaffold(saver=saver, copy_from_scaffold=scaffold)
      # The saver needs to be added to the graph for td3 hooks.
      tf.add_to_collection(tf.GraphKeys.SAVERS, scaffold.saver)
      return scaffold

    self._scaffold_fn = scaffold_fn
    optimizer = original_optimizer
    if (self._use_sync_replicas_optimizer and
        config is not None and config.num_worker_replicas > 1):
      optimizer = tf.train.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=config.num_worker_replicas - 1,
          total_num_replicas=config.num_worker_replicas)
    if self.is_device_gpu:
      optimizer = replicate_model_fn.TowerOptimizer.TowerOptimizer(optimizer)
    return optimizer

  def create_train_op(self,
                      loss,
                      optimizer,
                      update_ops=None,
                      train_outputs=None):
    """Create the train of from the loss obtained from model_train_fn.

    Args:
      loss: The loss we compute within model_train_fn.
      optimizer: An instance of `tf.train.Optimizer`.
      update_ops: List of update ops to execute alongside the training op.
      train_outputs: (Optional) A dict with additional tensors the training
        model generates.

    Returns:
      train_op: Op for the training step.
    """
    # We overwrite the default train op creation since we only want to train
    # with a subset of the variables.
    variables_to_train = self.get_trainable_variables()
    summarize_gradients = self._summarize_gradients
    if self.is_device_tpu:
      # TPUs don't support summaries up until now. Hence, we overwrite the user
      # provided summarize_gradients option to False.
      if self._summarize_gradients:
        logging.info('We cannot use summarize_gradients on TPUs.')
      summarize_gradients = False
    return contrib_training.create_train_op(
        loss,
        optimizer,
        summarize_gradients=summarize_gradients,
        variables_to_train=variables_to_train,
        update_ops=update_ops)

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config=None,
                     params=None):
    """See base class."""
    del mode, config, params
    self.loss_fn(features, labels, inference_outputs)
    return tf.losses.get_total_loss()


class DefaultGrasping44ImagePreprocessor(
    spec_transformation_preprocessor.SpecTransformationPreprocessor):
  """The default preprocessor for the Grasping44.

  This preprocessor takes the feature and label specs from the model and
  alters some of the specs, e.g. image conversions. The default processor
  does not list it's in_*_specification and out_*_specification explicitly since
  it is very close to the model and only performs the minimal required model
  specific changes. New more general preprocessors should list their
  in_*_specification as well as out_*_specification.
  """

  def _transform_in_feature_specification(
      self, tensor_spec_struct
  ):
    """The specification for the input features for the preprocess_fn.

    Here we will transform the feature spec to represent the requirements
    for preprocessing.

    Args:
      tensor_spec_struct: A flat spec structure {str: TensorSpec}.

    Returns:
      tensor_spec_struct: The transformed flat spec structure {str:
      TensorSpec}.
    """
    self.update_spec(
        tensor_spec_struct,
        'state/image',
        shape=(512, 640, 3),
        dtype=tf.uint8,
        data_format='jpeg')
    return tensor_spec_struct

  def _preprocess_fn(
      self, features,
      labels,
      mode
  ):
    """The preprocessing function which will be executed prior to the model_fn.

    Args:
      features: The input features extracted from a single example in our
        in_feature_specification format.
      labels: (Optional) The input labels extracted from a single example in our
        in_label_specification format.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.

    Returns:
      features: The preprocessed features, potentially adding
        additional tensors derived from the input features.
      labels: (Optional) The preprocessed labels, potentially
        adding additional tensors derived from the input features and labels.
    """
    if mode == TRAIN:
      image = image_transformations.RandomCropImages(
          [features.state.image], INPUT_SHAPE, TARGET_SHAPE)[0]
    else:
      image = image_transformations.CenterCropImages(
          [features.state.image], INPUT_SHAPE, TARGET_SHAPE)[0]
    image = tf.image.convert_image_dtype(image, tf.float32)
    if mode == TRAIN:
      image = (
          image_transformations.ApplyPhotometricImageDistortions([image])[0])
    features.state.image = image
    return features, labels


@gin.configurable
class Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom(
    LegacyGraspingModelWrapper):
  """QT-Opt T2R model."""

  def __init__(self, action_batch_size=None, **hparams):
    """Port of Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom model.

    The Grasping44 model which controls gripper open/close actions. This model
    maintains current gripper status as part of the state. Good performance with
    Q-Learning on kuka_e2e_grasping task.

    Args:
      action_batch_size: If specified, the size of action minibatches used in
        PREDICT mode.
      **hparams: Args to be passed to the parent class.
    """
    super(Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom,
          self).__init__(
              action_batch_size=action_batch_size, **hparams)

  def get_state_specification(self):
    image_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(472, 472, 3), dtype=tf.float32, name='image_1')
    return tensorspec_utils.TensorSpecStruct(image=image_spec)

  def get_action_specification(self):
    close_gripper_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='close_gripper')
    open_gripper_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='open_gripper')
    terminate_episode_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='terminate_episode')
    gripper_closed_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='gripper_closed')
    world_vector_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(3), dtype=tf.float32, name='world_vector')
    vertical_rotation_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(2), dtype=tf.float32, name='vertical_rotation')
    height_to_bottom_spec = tensorspec_utils.ExtendedTensorSpec(
        shape=(1,), dtype=tf.float32, name='height_to_bottom')

    return tensorspec_utils.TensorSpecStruct(
        world_vector=world_vector_spec,
        vertical_rotation=vertical_rotation_spec,
        close_gripper=close_gripper_spec,
        open_gripper=open_gripper_spec,
        terminate_episode=terminate_episode_spec,
        gripper_closed=gripper_closed_spec,
        height_to_bottom=height_to_bottom_spec)

  @property
  def default_preprocessor_cls(self):
    return DefaultGrasping44ImagePreprocessor

  def q_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None,
             reuse=tf.AUTO_REUSE,
             goal_vector_fn=None,
             goal_spatial_fn=None):
    base_model = self.create_legacy_model()
    concat_axis = 1
    if mode == PREDICT and self._tile_actions_for_predict:
      concat_axis = 2
    images = [None, features.state.image]
    grasp_params = base_model.create_grasp_params_input(
        features.action.to_dict(), concat_axis)

    is_training = mode == TRAIN
    _, end_points = base_model.model(
        images,
        grasp_params,
        goal_spatial_fn=goal_spatial_fn,
        goal_vector_fn=goal_vector_fn,
        is_training=is_training)
    return {
        'q_predicted': end_points['predictions'],
        'global_step': self.get_global_step()
    }

  def pack_features(self, *policy_inputs):
    return pack_features_kuka_e2e(self, *policy_inputs)

  @property
  def legacy_model_class(self):
    return networks.Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom
