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

"""T2R Model for BC-Z."""
import enum
from typing import Any, Dict, List, Optional, Text, Tuple

import gin
from tensor2robot.hooks import golden_values_hook_builder
from tensor2robot.layers import bcz_networks
from tensor2robot.layers import resnet
from tensor2robot.layers import vision_layers
from tensor2robot.models import abstract_model
from tensor2robot.preprocessors import distortion
from tensor2robot.preprocessors import spec_transformation_preprocessor
from tensor2robot.research.bcz import pose_components_lib
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf
from tensorflow_graphics.geometry.transformation import quaternion as quaternion_lib
import tensorflow_probability as tfp
import tf_slim as contrib_slim

slim = contrib_slim


tfd = tfp.distributions
nest = tf.nest
TensorSpec = tensorspec_utils.ExtendedTensorSpec  # pylint: disable=invalid-name

TRAIN = tf_estimator.ModeKeys.TRAIN
EVAL = tf_estimator.ModeKeys.EVAL
PREDICT = tf_estimator.ModeKeys.PREDICT

RunConfigType = abstract_model.RunConfigType
ParamsType = abstract_model.ParamsType
DictOrSpec = abstract_model.DictOrSpec
ActionComponent = pose_components_lib.ActionComponent
StateComponent = pose_components_lib.StateComponent
ModelTrainOutputType = abstract_model.ModelTrainOutputType
ExportOutputType = abstract_model.ExportOutputType

NUM_DEBUG_TASKS = 21  # Set to 17 for compatibility with old ckpt.
OLD_DATA = False

GRIPPER_CLOSE_FRACTION_TO_OPEN_GRIPPER = 0.4
MIN_GRIPPER_CLOSE = 0.2


@gin.constants_from_enum
class ConditionMode(enum.Enum):
  ONEHOT_TASKID = 1
  LANGUAGE_EMBEDDING = 2


@gin.configurable
class BCZPreprocessor(
    spec_transformation_preprocessor.SpecTransformationPreprocessor):
  """Image conversions and cropping for sequence or single-frames.
  """

  def __init__(
      self,
      image_size=(100, 100),
      crop_size=(512, 640),
      input_size=(512, 640),
      is_sequence=False,
      mixup_alpha=0.0,
      cutout_size=0,
      mock_subtask=False,
      binarize_gripper=True,
      rescale_gripper=False,
      **kwargs):
    self._image_size = image_size
    self._crop_size = crop_size
    self._input_size = input_size
    self._is_sequence = is_sequence
    self._mixup_alpha = mixup_alpha
    self._cutout_size = cutout_size
    self._mock_subtask = mock_subtask
    self._binarize_gripper = binarize_gripper
    self._rescale_gripper = rescale_gripper
    super(BCZPreprocessor, self).__init__(**kwargs)

  @property
  def rescale_gripper(self):
    return self._rescale_gripper

  def get_in_feature_specification(
      self, mode):
    # Don't want to parse the original_image, since we don't want to parse it
    # and we are adding this feature in preprocess_fn to satisfy the model's
    # inputs.
    tensor_spec_struct = tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))
    if mode != PREDICT and 'original_image' in tensor_spec_struct.keys():
      del tensor_spec_struct['original_image']
    if mode != PREDICT and 'original_depth_image' in tensor_spec_struct.keys():
      del tensor_spec_struct['original_depth_image']
    return self._transform_in_feature_specification(tensor_spec_struct)

  def _transform_in_feature_specification(
      self, flat_spec_structure
  ):
    """The specification for the input features for the preprocess_fn.

    Here we will transform the feature spec to represent the requirements
    for preprocessing.

    Args:
      flat_spec_structure: A flat spec structure {str: TensorSpec}.

    Returns:
      flat_spec_structure: The transformed flat spec structure {str:
      TensorSpec}.
    """
    # We replace the specification for the 'state/image'.
    # The model expects preprocessed and cropped 100x100 images,
    # while the original input to preprocessing are 640x512 kcam images.
    self.update_spec(
        flat_spec_structure,
        'image',
        shape=self._input_size + (3,),
        dtype=tf.uint8,
        data_format='jpeg')
    return flat_spec_structure

  def _preprocess_fn(
      self,
      features,
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
    features.original_image = features.image
    features.image = distortion.preprocess_image(
        features.image, mode, self._is_sequence, input_size=self._input_size,
        target_size=self._image_size, crop_size=self._crop_size)
    if self._mixup_alpha > 0. and labels and mode == TRAIN:
      lmbda = tfd.Beta(self._mixup_alpha, self._mixup_alpha).sample()
      # Mixup regularization.
      for key in ['image']:
        x2 = tf.reverse(features[key], axis=[0])
        features[key] = lmbda * features[key] + (1-lmbda)*x2
      for key, x in labels.future.items():
        x2 = tf.reverse(x, axis=[0])
        labels.future[key] = lmbda * x + (1-lmbda)*x2
    if self._cutout_size > 0 and mode == TRAIN:
      raise NotImplementedError(
          'Open-source BC-Z Model does not support cutout augmentation.')
    # Binarize target gripper close value to reduce overfitting.
    key = 'target_close'
    if labels and self._binarize_gripper and key in labels.future.keys():
      labels.future[key] = tf.cast(
          labels.future[key] > GRIPPER_CLOSE_FRACTION_TO_OPEN_GRIPPER,
          labels.future[key].dtype)
    # Rescale gripper targets such that the close threshold is 0.5,
    # only makes sense when binarize_gripper=False.
    if labels and self._rescale_gripper and key in labels.future.keys():
      # old: (~0.2, 1). Rescale labels to (0, 1) so log loss is balanced.
      labels.future[key] = tf.maximum(
          0.,
          (labels.future[key] - MIN_GRIPPER_CLOSE) / (1 - MIN_GRIPPER_CLOSE))
    # For testing with randomized inputs, gather_nd needs subtask_id to be
    # < len(dataset_keys).
    if self._mock_subtask:
      features.subtask_id = tf.zeros_like(features.subtask_id)
    return features, labels


@gin.configurable
def spatial_softmax_network(features,
                            is_training,
                            pose_components,
                            num_waypoints,
                            condition_input=None):
  """Spatial-Softmax based image-to-action network.

  Args:
    features: Input features to model.
    is_training: If is training mode or not.
    pose_components: List of ActionComponent tuples.
    num_waypoints: How many actions to predict (a trajectory).
    condition_input: Optional 2D conditioning input tensor.

  Returns:
    estimated_pose: Tensor of shape [batch_size, action_size] corresponding
      to predicted pose vector.
    feature_points: Tensor of shape [batch_size, feature_points_size]
      corresponding to state representation.
  """
  with tf.variable_scope('vision_model', reuse=tf.AUTO_REUSE):
    feature_points, _ = vision_layers.BuildImagesToFeaturesModel(
        features.image,
        is_training=is_training,
        normalizer_fn=slim.layer_norm)
    # Concatenate task embedding. Don't concat current gripper pose yet.
    if condition_input is not None:
      feature_points = tf.concat([feature_points, condition_input], -1)
    action_sizes = [t[1] for t in pose_components]
    estimated_pose, _ = vision_layers.BuildImageFeaturesToPoseModel(
        feature_points,
        aux_input=None,
        aux_output_dim=0,
        num_outputs=sum(action_sizes) * num_waypoints)
  network_output_dict = {}
  i = 0
  for name, size, is_residual, _ in pose_components:
    if is_residual:
      name += '_residual'
    n = size * num_waypoints
    network_output_dict[name] = tf.reshape(
        estimated_pose[Ellipsis, i:i+n], [-1, num_waypoints, size])
    i += n
  return network_output_dict, feature_points


@gin.configurable
def resnet_film_network(features,
                        mode,
                        pose_components,
                        num_waypoints,
                        film_generator_fn=None,
                        condition_input=None,
                        concat_cond_image=None,
                        fc_layers=(100, 100)):
  """ResNet-50-based image-to-action network."""
  is_training = mode == TRAIN
  # User needs to gin-configure resnet_model.film_generator_fn
  # Add preprocessed image to golden values.
  golden_values_hook_builder.add_golden_tensor(
      features.image, name='preprocessed_image')
  with tf.variable_scope('vision_model', reuse=tf.AUTO_REUSE):
    image = features.image
    if concat_cond_image is not None:
      image = tf.concat([image, concat_cond_image], axis=-1)
    outputs = resnet.resnet_model(
        image,
        is_training,
        num_classes=1,  # The classification head is unused.
        return_intermediate_values=True,
        film_generator_fn=film_generator_fn,
        film_generator_input=condition_input)
    net = tf.squeeze(outputs['final_reduce_mean'], axis=[1, 2])
    # Separate prediction heads for each action component.
    action_sizes, names = [], []
    for name, size, is_residual, _ in pose_components:
      if is_residual:
        name += '_residual'
      names.append(name)
      action_sizes.append(size)
    estimated_components = bcz_networks.MultiHeadMLP(net, action_sizes,
                                                     num_waypoints, fc_layers,
                                                     is_training)
    # block_layer3 is used to optionally infer the task.
    state_features = tf.reduce_mean(outputs['block_layer3'], axis=[1, 2])
    network_output_dict = dict(zip(names, estimated_components))
    network_output_dict['policy_image_features'] = net
    return network_output_dict, state_features


@gin.configurable
def predict_stop_network(state_embedding,
                         fc_layers=(100, 100),
                         num_waypoints=1,
                         scope_name='predict_stop'):
  """Small MLP for predicting (continue, fail/help, success).

  The order of the labels is defined by abstract_policy.StopState.

  Args:
    state_embedding: Input tensor from which to infer the stop state.
    fc_layers: Sequence of integers defining number and size of hidden fully
      connected layers.
    num_waypoints: Number of waypoints to predict stop states for.
    scope_name: Network scope name.

  Returns:
    Logits tensor of shae [batch_size, (num_waypoints-1)*3]
  """
  with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
    net = slim.stack(
        state_embedding, slim.fully_connected, fc_layers,
        activation_fn=tf.nn.relu, normalizer_fn=slim.layer_norm)
    logits = slim.fully_connected(net, 3, activation_fn=None)
    if num_waypoints > 1:
      # Only backprop 1st action.
      net = tf.stop_gradient(net)
      rest_logits = slim.fully_connected(
          net, (num_waypoints - 1) * 3, activation_fn=None)
      logits = tf.concat([logits, rest_logits], axis=-1)
  return logits


def infer_outputs(features,
                  network_output_dict,
                  action_components,
                  rescale_target_close,
                  repeat_feature_batch_dim = None):
  """Convert network_fn outputs to (absolute pose) expected by environment.

  network_fn is shared between different models, and as such can be re-purposed
  to generate different numbers of parameters for each pose component (e.g.
  means and variances for Gaussian mixtures).

  To avoid bugs in discrepancies between train vs. inference, note that
  infer_outputs is designed to be used as a *inference-only* function, i.e. for
  recovering actions. You may use its outputs to compute training graph, but you
  should do so explicitly.

  Args:
    features: TensorSpecStruct containing current_pose and image keys.
    network_output_dict: Dictionary of pose components predicted by pose
      regression network.
    action_components: List of ActionComponent tuples.
    rescale_target_close: If True, rescales target_close prediction from [0, 1]
      to [MIN_GRIPPER, 1].
    repeat_feature_batch_dim: If not None, number of times to duplicate the
      batch dimension of features to match the batch dimension of
      the network outputs.

  Returns:
    Dictionary of pose prediction inference outputs. Description of keys:
      action: flattened [batch, num_waypoints, action_size] tensor to be used
        by policy inference.
      action_trajectory: Is [..., num_waypoints, action_size] to make losses
        and visualization easier to compute.
      xyz: XYZ components of camera-frame future poses.
      quaternion: Normalized quaternion component of action.
      quaternion_norm: Norm of the quaternion (before normalization).
      gripper_close_fraction: Last component of the tensor, representing how far
        the gripper is closed.
      image: Image input to the pose regressor.
  """
  batch_dims = list(network_output_dict.values())[0].shape.as_list()[:-2]
  # When mode == PREDICT, batch_dim is set to None.
  if batch_dims[0] is None:
    batch_dims[0] = -1
  # Sigmoid the gripper output
  inference_outputs = {}
  action_outputs = []
  # Build lookup for residual.
  for name, _, is_residual, _ in action_components:
    predict_name = name
    if is_residual:
      predict_name += '_residual'

      if repeat_feature_batch_dim is not None and repeat_feature_batch_dim > 1:
        network_batch_dim = network_output_dict[predict_name].shape.as_list()[0]
        feature_batch_dim = features.present[name].shape.as_list()[0]
        # Only repeat the batch dimension when needed.
        if network_batch_dim is not None and network_batch_dim // feature_batch_dim == repeat_feature_batch_dim:
          features.present[name] = tf.repeat(
              features.present[name], repeat_feature_batch_dim, axis=0)

    if name == 'xyz':
      xyz = network_output_dict[predict_name]
      if is_residual:
        xyz += tf.reshape(features.present[name], batch_dims + [1, 3])
      action_outputs.append(xyz)
    elif name == 'quaternion':
      quaternion = network_output_dict[predict_name]
      # Normalize quaternion.
      quaternion_norm = tf.linalg.norm(quaternion, axis=-1, keepdims=True)
      quaternion = quaternion / quaternion_norm
      if is_residual:
        curr_quat = tf.reshape(features.present[name], batch_dims + [1, 4])
        quaternion = quaternion_lib.multiply(curr_quat, quaternion)
      action_outputs.append(quaternion)
      # old models minimize norm(unnorm_quaternion) - quaternion_label, which
      # works better than unnorm_quaternion - quaternion_label. Therefore,
      # we re-write network_output_dict with the normalized quaternion.
      network_output_dict['quaternion'] = quaternion
      inference_outputs['quaternion_norm'] = quaternion_norm
    elif name == 'axis_angle':
      axis_angle = network_output_dict[predict_name]
      if is_residual:
        curr_axis_angle = tf.reshape(
            features.present[name], batch_dims + [1, 3])
        axis_angle += curr_axis_angle
      action_outputs.append(axis_angle)
    elif name == 'arm_joints':
      joints = network_output_dict[predict_name]
      if is_residual:
        joints += tf.reshape(features.present[name], batch_dims + [1, 7])
      action_outputs.append(joints)
    elif name == 'arm_joints_velocity':
      joints_velocity = network_output_dict[predict_name]
      action_outputs.append(joints_velocity)
    elif name == 'pantilt':
      pantilt = network_output_dict[predict_name]
      if is_residual:
        pantilt += tf.reshape(features.present[name], batch_dims + [1, 2])
      action_outputs.append(pantilt)
    elif name == 'robot_linear_velocity':
      robot_linear_velocity = network_output_dict[predict_name]
      if is_residual:
        robot_linear_velocity += tf.reshape(
            features.present[name], batch_dims + [1, 1])
      action_outputs.append(robot_linear_velocity)
    elif name == 'robot_angular_velocity':
      robot_angular_velocity = network_output_dict[predict_name]
      if is_residual:
        robot_angular_velocity += tf.reshape(
            features.present[name], batch_dims + [1, 1])
      action_outputs.append(robot_angular_velocity)
    elif name in ['target_close', 'stop_token']:
      # We have to do this again in train_outputs for computing loss.
      value = network_output_dict[predict_name]
      if is_residual:
        raise ValueError(
            'target_close/stop_token do not support residual gripper')
      value = tf.nn.sigmoid(value)
      if rescale_target_close:
        value = MIN_GRIPPER_CLOSE + value * (1 - MIN_GRIPPER_CLOSE)
      action_outputs.append(value)
    elif name == 'base_joystick_xy':
      action_outputs.append(tf.nn.tanh(network_output_dict[predict_name]))
  # Training losses will be computed from *UNMODIFIED* network_output_dict
  # values, except quaternion, which seems to work better when regressing
  # normalized predictions (and we have overwritten in this function).
  inference_outputs.update(network_output_dict)
  # Inference outputs for each pose component.
  assert len(action_outputs) == len(action_components)
  for i, output in enumerate(action_outputs):
    inference_outputs['action/' + action_components[i][0]] = output
  estimated_pose = tf.concat(action_outputs, axis=-1)
  inference_outputs['action_trajectory'] = estimated_pose
  if 'image' in features.keys():
    inference_outputs['image'] = features.image
  if 'depth_image' in features.keys():
    inference_outputs['depth_image'] = features.depth_image
  return inference_outputs


@gin.configurable
def compute_stop_state_loss(stop_state_labels,
                            stop_state_predictions,
                            class_weights=gin.REQUIRED):
  """Constructs loss for the stop_state_prediction."""
  class_weights = tf.constant(class_weights)
  weights = tf.reduce_sum(stop_state_labels * class_weights, -1)
  return tf.losses.softmax_cross_entropy(
      stop_state_labels,
      stop_state_predictions,
      weights=weights)


@gin.configurable
def training_outputs(features,
                     labels,
                     network_output_dict,
                     action_components,
                     quaternion_penalty=0.01,
                     loss_name='huber',
                     repeat_label_batch_dim=None):
  """Compute training output dictionary and pose regression losses.

  Compute losses from output of network_fn, not from inference_outputs actions.

  Args:
    features: TensorSpecStruct for BCZModel.
    labels: TensorSpecStruct for BCZModel.
    network_output_dict: Outputs returned by infer_outputs.
    action_components: List of ActionComponent tuples.
    quaternion_penalty: Penalty coeff. for normalization from QuarterNet
      (Pavllo et al. 2018).
    loss_name: 'huber' or 'mse' for XYZ and Quaternion losses.
    repeat_label_batch_dim: If not None, number of times to duplicate the
      batch dimension of labels to match the batch dimension of
      the network outputs.
  Returns:
    Dictionary containing training outputs (loss subcomponents).
  """
  del features
  if loss_name == 'mse':
    reg_loss_fn = tf.losses.mean_squared_error
  elif loss_name == 'huber':
    reg_loss_fn = tf.losses.huber_loss
  elif loss_name == 'clipped_huber':
    # Some loss can be initially very large and mess with summaries and grads.
    # Clip the loss.
    def reg_loss_fn(**kwargs):
      return tf.nn.relu6(tf.losses.huber_loss(**kwargs))
  elif loss_name == 'piecewise_scaled_huber':
    # Some loss can be initially very large and mess with summaries and grads.
    # Clipping the loss could cause some gradient issue, Here we scaled the loss
    # if it is larger than a threshold.
    reg_loss_fn = piecewise_scaled_huber(loss_fn=tf.losses.huber_loss)
  else:
    raise ValueError('invalid loss')
  # Predict stop did worse in bcz, make sure this is set to False.
  if 'stop_token' in labels.future.keys():
    stop_mask_value = 1.0 - labels.future.stop_token
  else:
    stop_mask_value = 1.0
  # Dictionary to hold all the losses.
  train_outputs = {}
  # Dictionary to hold outputs that aren't losses, gets merged in at the end.
  nonloss_outputs = {}
  for name, _, is_residual, weight in action_components:
    predict_name = name
    if is_residual:
      predict_name += '_residual'
    predicted = network_output_dict[predict_name]
    label_name = name
    if is_residual:
      label_name += '_residual'
    label = labels.future[label_name]

    if repeat_label_batch_dim is not None and repeat_label_batch_dim > 1:
      label = tf.repeat(label, repeat_label_batch_dim, axis=0)

    if name in ['target_close', 'stop_token']:
      # Binary predictions trained with log_loss.
      predicted = tf.nn.sigmoid(predicted)
      nonloss_outputs[name + '_predicted'] = predicted
      loss_fn = tf.losses.log_loss
    else:
      loss_fn = reg_loss_fn
    # Broadcast stop_token.
    stop_mask = stop_mask_value * tf.ones_like(predicted)
    train_outputs[name + '_loss'] = loss_fn(
        labels=label, predictions=predicted,
        weights=weight * stop_mask)
    # We only use first waypoint to evaluate quality of model. Note that
    # first waypoint is already included in the trajectory loss.
    nonloss_outputs['first_' + name + '_error'] = loss_fn(
        labels=label[Ellipsis, 0, :], predictions=predicted[Ellipsis, 0, :],
        weights=weight)

  # Append quaternion label, if applicable.
  name = 'quaternion_norm'
  if name in network_output_dict:
    predicted = network_output_dict[name]
    train_outputs[name + '_loss'] = reg_loss_fn(
        labels=tf.ones_like(predicted), predictions=predicted,
        weights=quaternion_penalty * stop_mask_value)

  # Stop state prediction loss.
  if 'stop_state' in network_output_dict:
    stop_labels = tf.one_hot(
        tf.cast(labels.future.stop_state, tf.int64), depth=3)
    stop_predictions = network_output_dict['stop_state']
    train_outputs['stop_state_loss'] = compute_stop_state_loss(
        stop_labels,
        stop_predictions)

  # Add regularization losses.
  regularization_losses = tf.compat.v1.losses.get_regularization_losses()
  if regularization_losses:
    train_outputs['total_regularization_loss'] = tf.add_n(
        regularization_losses)
  loss = tf.add_n(list(train_outputs.values()))
  train_outputs.update(nonloss_outputs)
  # Add each of the losses to a collection.
  for name, tensor in train_outputs.items():
    golden_values_hook_builder.add_golden_tensor(tensor, name)
  return loss, train_outputs


def get_gripper_accuracy_metrics(inference_outputs, features, labels):
  """Return metrics for gripper close prediction accuracy."""
  # Binarize the gripper first.
  key = 'target_close'
  current = features.present[key]  # note that this is parsing sensed_close.
  dtype = labels.future[key].dtype
  thresh = 0
  predicted_is_closing = tf.cast(
      inference_outputs[key][:, 0] - current > thresh, dtype)
  label_is_closing = tf.cast(
      labels.future[key][:, 0] - current > thresh, dtype)
  predicted_is_opening = tf.cast(
      inference_outputs[key][:, 0] - current < -thresh, dtype)
  label_is_opening = tf.cast(
      labels.future[key][:, 0] - current < -thresh, dtype)

  metrics = {}
  for s, label, predicted in zip(['closing', 'opening'],
                                 [label_is_closing, label_is_opening],
                                 [predicted_is_closing, predicted_is_opening]):
    metrics[s + '_accuracy'] = tf.metrics.accuracy(
        label, predicted)
    metrics[s + '_auc'] = tf.metrics.auc(
        label, predicted)
    metrics[s + '_precision'] = tf.metrics.precision(
        label, predicted)
    metrics[s + '_recall'] = tf.metrics.recall(
        label, predicted)
    metrics[s + '_pos_freq'] = tf.metrics.accuracy(
        tf.ones_like(label), label)
  return metrics


def xyz_action_trajectory(outputs):
  if 'action/quaternion' in outputs:
    rotation = outputs['action/quaternion']
  elif 'action/axis_angle' in outputs:
    rotation = outputs['action/axis_angle']
  return tf.concat(
      [outputs['action/xyz'], rotation], axis=-1)


@gin.configurable
def piecewise_scaled_huber(loss_fn, threshold=0.2, slope=0.001):
  def clipped_loss_fn(**kwargs):
    loss = loss_fn(**kwargs)
    return tf.cond(loss > 1, lambda: threshold + (loss - threshold) * slope,
                   lambda: loss)

  return clipped_loss_fn


@gin.configurable
class BCZModel(abstract_model.AbstractT2RModel):
  """Single-image configurable regression model for single-task BC-Z env."""

  def __init__(
      self,
      state_components = None,
      action_components = None,
      predict_stop = False,
      image_size = (100, 100),
      input_size = None,
      dataset_keys = None,
      num_waypoints = 1,
      num_past = 0,
      num_total_users = 0,
      network_fn=resnet_film_network,
      ignore_task_embedding=False,
      task_embedding_noise_std=0.1,
      init_checkpoint=None,
      mask_stop_token=False,
      cond_modality = ConditionMode.ONEHOT_TASKID,
      **kwargs):
    """Constructor.

    Args:
      state_components: What auxiliary state inputs to condition the policy on.
      action_components: What outputs to predict.
      predict_stop: If True, predicts 3-way classification for whether
        to continue, halt (unsafe / failure / ask for help) or stop in a
        successful state.
      image_size: What size image inputs to use.
      input_size: If defined, it indicates this model uses a preprocessor that
        changes the initial input to image_size. These preprocessors are
        expected to yield the initial input as well, this should be configured
        to match that size. If None, we assume no original_image feature.
      dataset_keys: Names of different tasks, in indexed order.
      num_waypoints: How many future waypoints to predict.
      num_past: How many past waypoints to condition the model on.
      num_total_users: If > 0, conditions network on one-hot encoded user id.
      network_fn: Which model to build.
      ignore_task_embedding: If True, does not condition on the subtask_idx.
      task_embedding_noise_std: Add noise to task embedding before conditioning
        action decoder on it.
      init_checkpoint: Full init checkpoint for fine-tuning an already-trained
        model.
      mask_stop_token: If True, parses stop_token and uses it to mask prediction
        losses.
      cond_modality: Whether to condition task on one-hot ID or language
        embedding.
      **kwargs: Keyword arguments for AbstractPreprocessor.
    """
    super(BCZModel, self).__init__(**kwargs)
    self._image_size = image_size
    self._input_size = input_size
    self._predict_stop = predict_stop
    self._dataset_keys = dataset_keys
    self._num_waypoints = num_waypoints
    self._num_past = num_past
    self._network_fn = network_fn
    self._ignore_task_embedding = ignore_task_embedding
    self._task_embedding_noise_std = task_embedding_noise_std
    if action_components is None:
      action_components = pose_components_lib.DEFAULT_ACTION_COMPONENTS
    self._action_components = action_components
    if state_components is None:
      state_components = []
    self._state_components = state_components
    self._init_checkpoint = init_checkpoint
    self._mask_stop_token = mask_stop_token
    self._num_total_users = num_total_users
    self._cond_mode = cond_modality

  @property
  def default_preprocessor_cls(self):
    return BCZPreprocessor

  @property
  def action_component_names(self):
    return [p[0] for p in self._action_components]

  @property
  def is_joint_space(self):
    return 'arm_joints' in self.action_component_names

  @property
  def is_xyz_space(self):
    return 'xyz' in self.action_component_names

  def pack_features(
      self,
      state,
      prev_episode_data,
      timestep):
    """Pass-through function, as environment should do the feature packing."""
    del prev_episode_data, timestep
    return state

  def get_feature_specification(self, mode):
    del mode
    features = tensorspec_utils.TensorSpecStruct()
    # Present
    suffix = '' if OLD_DATA else '/encoded'
    features.image = TensorSpec(
        shape=self._image_size + (3,),
        dtype=tf.float32, name='present/image' + suffix,
        data_format='jpeg',
        is_sequence=False)
    present = tensorspec_utils.TensorSpecStruct()
    # Present poses cannot be residual.
    for name, size, _ in self._state_components:
      present[name] = TensorSpec(
          shape=(size), dtype=tf.float32,
          name='present/' + name, is_sequence=False)
    for name, size, _, _ in self._action_components:
      # We parse sensed_close for present['target_close'] because
      # target_close contains future information not in the present.
      data_name = 'sensed_close' if name == 'target_close' else name
      present[name] = TensorSpec(
          shape=(size), dtype=tf.float32,
          name='present/' + data_name, is_sequence=False)
    features.present = present
    if self._cond_mode == ConditionMode.ONEHOT_TASKID:
      features.subtask_id = tensorspec_utils.ExtendedTensorSpec(
          shape=(1,), dtype=tf.int64, name='subtask_id')
    elif self._cond_mode == ConditionMode.LANGUAGE_EMBEDDING:
      features.sentence_embedding = tensorspec_utils.ExtendedTensorSpec(
          shape=(512,), dtype=tf.float32, name='sentence_embedding')
    if self._num_total_users:
      features.user_id = tensorspec_utils.ExtendedTensorSpec(
          shape=(1,), dtype=tf.int64, name='user_int')
    # Optional Feature Specs (present).
    prefix = '' if OLD_DATA else 'present/'
    features.camera_intrinsics = tensorspec_utils.ExtendedTensorSpec(
        shape=(3, 3), dtype=tf.float32, name=prefix + 'camera_rgb/intrinsics',
        is_optional=True)
    features.camera_pose_base = tensorspec_utils.ExtendedTensorSpec(
        shape=(12,), dtype=tf.float32, name=prefix + 'camera_pose_base',
        is_optional=True)
    # Doesn't have a name, since preprocessor yields this.
    input_size = self._input_size if self._input_size else (512, 640)
    features.original_image = tensorspec_utils.ExtendedTensorSpec(
        shape=input_size + (3,), dtype=tf.uint8, data_format='jpeg',
        is_optional=True)
    # Past.
    if self._num_past:
      past = tensorspec_utils.TensorSpecStruct()
      for name, size, residual in self._state_components:
        if residual:
          name += '_residual'
        past[name] = TensorSpec(
            shape=(self._num_past, size), dtype=tf.float32,
            name='past/' + name, is_sequence=False)
      features.past = past
    return features

  def get_label_specification(self, mode):
    del mode
    future = tensorspec_utils.TensorSpecStruct()
    if self._predict_stop:
      future['stop_state'] = TensorSpec(
          shape=(), dtype=tf.int64, name='present/stop_state')
    for name, size, residual, _ in self._action_components:
      if residual:
        name += '_residual'
      future[name] = TensorSpec(
          shape=(self._num_waypoints, size), dtype=tf.float32,
          name='future/' + name, is_sequence=False)
    # Used for loss masking trajectories. (B, N, 1)
    if self._mask_stop_token:
      future.stop_token = TensorSpec(
          shape=(self._num_waypoints, 1),
          dtype=tf.float32, name='future/stop_token', is_sequence=False)
    return tensorspec_utils.TensorSpecStruct(future=future)

  def augment_condition_input(self, condition_input, features, is_training):
    if self._task_embedding_noise_std is not None and is_training:
      condition_input += tf.random.normal(
          tf.shape(condition_input), stddev=self._task_embedding_noise_std)
    if self._ignore_task_embedding:
      # Zero out task embeddings as a baseline.
      condition_input = None
    if self._state_components:
      curr_pose = tf.concat(
          [features.present[t[0]] for t in self._state_components], axis=-1)
      if condition_input is None:
        condition_input = curr_pose
      else:
        condition_input = tf.concat([condition_input, curr_pose], axis=-1)
    if self._num_total_users:
      user_id = tf.one_hot(features.user_id[:, 0], self._num_total_users)
      condition_input = tf.concat([condition_input, user_id], axis=-1)
    if self._num_past:
      # Append past action history to conditioning input.
      pose_size = sum([t[1] for t in self._state_components])
      prev_poses = []
      for name, _, residual in self._state_components:
        if residual:
          name += '_residual'
        prev_poses.append(features.past[name])
      prev_poses = tf.concat(prev_poses, axis=-1)
      prev_poses = tf.reshape(prev_poses, [-1, self._num_past * pose_size])
      if condition_input is None:
        condition_input = prev_poses
      else:
        condition_input = tf.concat([condition_input, prev_poses], axis=-1)
    return condition_input

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config = None,
                           params = None):
    """A (state) regression function."""
    del config
    is_training = mode == TRAIN
    if self._cond_mode == ConditionMode.ONEHOT_TASKID:
      condition_input = tf.one_hot(features.subtask_id[:, 0], NUM_DEBUG_TASKS)
    elif self._cond_mode == ConditionMode.LANGUAGE_EMBEDDING:
      condition_input = features.sentence_embedding
    condition_input = self.augment_condition_input(condition_input, features,
                                                   is_training)
    rescale_target_close = self.preprocessor.rescale_gripper
    def _network_actions(c):
      network_outputs_dict, state_embedding = self._network_fn(
          features, mode, self._action_components, self._num_waypoints,
          condition_input=c)
      outputs = infer_outputs(
          features, network_outputs_dict, self._action_components,
          rescale_target_close)
      return outputs, state_embedding
    outputs, state_embedding = _network_actions(condition_input)

    # Infer stop state.
    if self._predict_stop:
      outputs['stop_state'] = predict_stop_network(state_embedding)
    if not self._ignore_task_embedding:
      outputs['condition_input'] = condition_input
    if self._init_checkpoint is not None:
      # Load checkpoint root scope (key) into this model's root scope (value).
      assignment_map = {'/': '/'}
      tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)
    return outputs

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None):
    return training_outputs(features, labels, inference_outputs,
                            self._action_components)

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
    metrics = {}
    if train_outputs is not None:
      for key, value in train_outputs.items():
        metrics['mean_' + key] = tf.metrics.mean(value)
    name = None
    if self.is_joint_space:
      name = 'mean_first_arm_joints_error'
    elif self.is_xyz_space:
      name = 'mean_first_xyz_error'
    if name:
      _ = tf.identity(metrics[name][0], name=name)
    # Stop state prediction accuracy.
    if self._predict_stop:
      predictions = tf.argmax(
          inference_outputs['stop_state'], axis=-1, output_type=tf.int64)
      metrics['accuracy_stop_state'] = tf.metrics.accuracy(
          labels.future.stop_state, predictions)

    # Explicit not-None check required here, or else TensorFlow type checking
    # complains about implicit bool values used for tf.Tensor.
    if (train_outputs and labels is not None
        and 'target_close' in self.action_component_names):
      metrics.update(
          get_gripper_accuracy_metrics(inference_outputs, features, labels))
    return metrics

  def add_summaries(self,
                    features,
                    labels,
                    inference_outputs,
                    train_loss,
                    train_outputs,
                    mode,
                    config=None,
                    params=None):
    """Summary function to support visualization in meta learning inner loop."""
    if not self.use_summaries(params):
      return
    if 'image' in features.keys():
      tf.summary.image('image', inference_outputs['image'])
    # Losses
    if train_outputs:
      for key, value in train_outputs.items():
        # Exclude Tensors with ndims >= 2
        if not (isinstance(value, tf.Tensor) and len(value.shape) >= 2):
          tf.summary.scalar(key, value)
