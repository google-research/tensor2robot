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

"""T2RModels for VRGripper env tasks."""

from typing import Callable, Dict, List, Optional, Text, Tuple
import gin
import numpy as np
from tensor2robot.layers import mdn
from tensor2robot.layers import vision_layers
from tensor2robot.meta_learning import meta_tfdata
from tensor2robot.models import abstract_model
from tensor2robot.models import regression_model
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.preprocessors import distortion
from tensor2robot.utils import tensorspec_utils
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v1 as tf  # tf
import tensorflow_probability as tfp
from tensorflow.contrib import layers as contrib_layers

TensorSpec = tensorspec_utils.ExtendedTensorSpec
TRAIN = tf_estimator.ModeKeys.TRAIN
PREDICT = tf_estimator.ModeKeys.PREDICT
FLOAT_DTYPES = [tf.bfloat16, tf.float32, tf.float64]


@gin.configurable
class DefaultVRGripperPreprocessor(abstract_preprocessor.AbstractPreprocessor):
  """The default VRGripperEnv preprocessor."""

  def __init__(self,
               src_img_res = (220, 300),
               crop_size = (200, 280),
               mixup_alpha = 0.0,
               **kwargs):
    """Construct the preprocessor.

    Args:
      src_img_res: The true height and width of the image data. If the model
        expects images of a different size, we automatically resize the images.
      crop_size: Before resizing the image, take a crop of the image to this
        height and width. Is a no-op if equal to src_img_res. Crop is done
        randomly at train time, and is take from the center otherwise.
      mixup_alpha: If > 0., turns on Mixup data augmentation for features and
        labels.
      **kwargs: Keyword args passed to parent class.
    """
    super(DefaultVRGripperPreprocessor, self).__init__(**kwargs)
    self._src_img_res = src_img_res
    self._crop_size = crop_size
    self._mixup_alpha = mixup_alpha

  def get_in_feature_specification(self, mode
                                  ):
    """See base class."""
    feature_spec = tensorspec_utils.copy_tensorspec(
        self._model_feature_specification_fn(mode))
    # Don't want to parse the original_image, since we don't want to parse it
    # and we are adding this feature in preprocess_fn to satisfy the model's
    # inputs.
    if mode != PREDICT and 'original_image' in feature_spec:
      del feature_spec['original_image']

    if 'image' in feature_spec:
      true_img_shape = feature_spec.image.shape.as_list()
      # Overwrite the H, W dimensions.
      true_img_shape[-3:-1] = self._src_img_res
      feature_spec.image = TensorSpec.from_spec(
          feature_spec.image, shape=true_img_shape, dtype=tf.uint8)
    return tensorspec_utils.flatten_spec_structure(feature_spec)

  def get_in_label_specification(self, mode
                                ):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def get_out_feature_specification(self, mode
                                   ):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_feature_specification_fn(mode))

  def get_out_label_specification(self, mode
                                 ):
    """See base class."""
    return tensorspec_utils.flatten_spec_structure(
        self._model_label_specification_fn(mode))

  def _preprocess_fn(
      self, features,
      labels,
      mode
  ):
    """Resize images and convert them from uint8 -> float32."""
    if 'image' in features:
      ndim = len(features.image.shape)
      is_sequence = (ndim > 4)
      input_size = self._src_img_res
      target_size = self._crop_size
      features.original_image = features.image
      features.image = distortion.preprocess_image(features.image, mode,
                                                   is_sequence, input_size,
                                                   target_size)

      features.image = tf.image.convert_image_dtype(features.image, tf.float32)
      out_feature_spec = self.get_out_feature_specification(mode)
      if out_feature_spec.image.shape != features.image.shape:
        features.image = meta_tfdata.multi_batch_apply(
            tf.image.resize_images, 2, features.image,
            out_feature_spec.image.shape.as_list()[-3:-1])

    if self._mixup_alpha > 0. and labels and mode == TRAIN:
      lmbda = tfp.distributions.Beta(
          self._mixup_alpha, self._mixup_alpha).sample()
      for key, x in features.items():
        if x.dtype in FLOAT_DTYPES:
          features[key] = lmbda * x + (1-lmbda)*tf.reverse(x, axis=[0])
      if labels is not None:
        for key, x in labels.items():
          if x.dtype in FLOAT_DTYPES:
            labels[key] = lmbda * x + (1 - lmbda) * tf.reverse(x, axis=[0])
    return features, labels


@gin.configurable
class VRGripperRegressionModel(regression_model.RegressionModel):
  """Continuous regression output model for VRGripper Env."""

  def __init__(self,
               use_gripper_input = True,
               normalize_outputs = False,
               output_mean = None,
               output_stddev = None,
               outer_loss_multiplier = 1.,
               num_mixture_components = 1,
               output_mixture_sample = False,
               condition_mixture_stddev = False,
               episode_length = 40,
               **kwargs):
    """Initialize the VRGripperRegressionModel.

    Args:
      use_gripper_input: If True, concatenate gripper pose with input to the
        fully connected layers when predicting actions.
      normalize_outputs:  If True, scale actions by `output_stddev` and
        translate by `output_mean`.
      output_mean:  The empirical mean of demonstration actions.
      output_stddev:  The empirical standard deviation of demonstration actions.
      outer_loss_multiplier:  A scaling factor for the outer loss.
      num_mixture_components:  The number of gaussian mixture components. Use 1
        for standard mean squared error regression.
      output_mixture_sample: If True (and num_mixture_components > 1), output
        actions by sampling from a gaussian mixture. Otherwise, we use the mean
        of the most likely component.
      condition_mixture_stddev: If True, the mixture standard deviations will be
        output from a neural net and thus conditioned on image/state. Otherwise,
        they will simply be learned variables (unconditioned on image/state).
      episode_length: The fixed length of an episode in the data.
      **kwargs: Passed to parent.

    Raises:
      ValueError: If `output_mean` or `output_stddev` have incorrect length.
    """
    super(VRGripperRegressionModel, self).__init__(**kwargs)
    self._use_gripper_input = use_gripper_input
    self._normalize_outputs = normalize_outputs
    self._output_mean = None
    self._output_stddev = None
    self._outer_loss_multiplier = outer_loss_multiplier
    self._num_mixture_components = num_mixture_components
    self._output_mixture_sample = output_mixture_sample
    self._condition_mixture_stddev = condition_mixture_stddev
    self._episode_length = episode_length
    if output_mean and output_stddev:
      if not len(output_mean) == len(output_stddev) == self.action_size:
        raise ValueError(
            'Output mean and stddev have lengths {:d} and {:d}.'.format(
                len(output_mean), len(output_stddev)))
      self._output_mean = np.array([output_mean])
      self._output_stddev = np.array([output_stddev])

  @property
  def default_preprocessor_cls(self):
    return DefaultVRGripperPreprocessor

  def get_feature_specification(self, mode):
    del mode
    image_spec = TensorSpec(
        shape=(100, 100, 3),
        dtype=tf.float32,
        name='image0',
        data_format='jpeg')
    gripper_pose_spec = TensorSpec(
        shape=(14,), dtype=tf.float32, name='world_pose_gripper')
    tspec = tensorspec_utils.TensorSpecStruct(
        image=image_spec, gripper_pose=gripper_pose_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  def get_label_specification(self, mode):
    del mode
    action_spec = TensorSpec(
        shape=(self._action_size,), dtype=tf.float32, name='action_world')
    tspec = tensorspec_utils.TensorSpecStruct(action=action_spec)
    return tensorspec_utils.copy_tensorspec(
        tspec, batch_size=self._episode_length)

  @property
  def action_size(self):
    return self._action_size

  def _single_batch_a_func(self,
                           features,
                           scope,
                           mode,
                           context_fn=None,
                           reuse=tf.AUTO_REUSE):
    """A state -> action regression function that expects a single batch dim."""
    gripper_pose = features.gripper_pose if self._use_gripper_input else None
    with tf.variable_scope(scope, reuse=reuse, use_resource=True):
      with tf.variable_scope('state_features', reuse=reuse, use_resource=True):
        feature_points, end_points = vision_layers.BuildImagesToFeaturesModel(
            features.image,
            is_training=(mode == TRAIN),
            normalizer_fn=contrib_layers.layer_norm)

      if context_fn:
        feature_points = context_fn(feature_points)

      fc_input = tf.concat([feature_points, gripper_pose], -1)
      outputs = {}
      if self._num_mixture_components > 1:
        dist_params = mdn.predict_mdn_params(
            fc_input,
            self._num_mixture_components,
            self._action_size,
            condition_sigmas=self._condition_mixture_stddev)
        gm = mdn.get_mixture_distribution(
            dist_params, self._num_mixture_components, self._action_size,
            self._output_mean if self._normalize_outputs else None)
        if self._output_mixture_sample:
          # Output a mixture sample as action.
          action = gm.sample()
        else:
          action = mdn.gaussian_mixture_approximate_mode(gm)
        outputs['dist_params'] = dist_params
      else:
        action, _ = vision_layers.BuildImageFeaturesToPoseModel(
            fc_input, num_outputs=self._action_size)
        action = self._output_mean + self._output_stddev * action
    outputs.update({
        'inference_output': action,
        'image': features.image,
        'feature_points': feature_points,
        'softmax': end_points['softmax']
    })
    return outputs

  def a_func(self,
             features,
             scope,
             mode,
             context_fn=None,
             reuse=tf.AUTO_REUSE,
             config=None,
             params=None):
    """A (state) regression function.

    This function can return a stochastic or a deterministic tensor.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_spefication.
      scope: String specifying variable scope.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      context_fn: Optional python function that takes in features and returns
        new features of same shape. For merging information like in RL^2.
      reuse: Whether or not to reuse variables under variable scope 'scope'.
      config: Optional configuration object. Will receive what is passed to
        Estimator in config parameter, or the default config. Allows updating
        things in your model_fn based on configuration such as num_ps_replicas,
        or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      outputs: A {key: Tensor} mapping. The key 'action' is required.
    """
    del config, params
    return meta_tfdata.multi_batch_apply(self._single_batch_a_func, 2, features,
                                         scope, mode, context_fn, reuse)

  def loss_fn(self, labels, inference_outputs, mode, params=None):
    """This implements outer loss and configurable inner losses."""
    if params and params.get('is_outer_loss', False):
      pass
    if self._num_mixture_components > 1:
      gm = mdn.get_mixture_distribution(
          inference_outputs['dist_params'], self._num_mixture_components,
          self._action_size,
          self._output_mean if self._normalize_outputs else None)
      return -tf.reduce_mean(gm.log_prob(labels.action))
    else:
      return self._outer_loss_multiplier * tf.losses.mean_squared_error(
          labels=labels.action,
          predictions=inference_outputs['inference_output'])


@gin.configurable
class VRGripperDomainAdaptiveModel(VRGripperRegressionModel):
  """Base model which uses a learned loss to do domain adaptive imitation.

  The model conditions on video only (no actions or gripper pose).
  """

  def __init__(self,
               predict_con_gripper_pose = False,
               learned_loss_conv1d_layers = (10, 10,
                                                                        6),
               **kwargs):
    """Initialize the model.

    Args:
      predict_con_gripper_pose: If True, predict the condition gripper pose
        input from the image features. Otherwise, set to zeros.
      learned_loss_conv1d_layers: A tuple describing the conv1d layers of the
        learned loss. If None, the learned loss won't use conv1d layers.
      **kwargs: Passed to parent.
    """
    super(VRGripperDomainAdaptiveModel, self).__init__(**kwargs)
    self._predict_con_gripper_pose = predict_con_gripper_pose
    self._learned_loss_conv1d_layers = learned_loss_conv1d_layers

  def _predict_gripper_pose(self, feature_points):
    """Predict the condition gripper pose from feature points."""
    out = feature_points
    out = tf.layers.dense(out, 40, activation=tf.nn.relu, use_bias=False)
    out = contrib_layers.layer_norm(out)
    out = tf.layers.dense(out, 14, activation=None)
    return out

  def single_batch_a_func(
      self, features, scope,
      mode,
      context_fn, reuse,
      config,
      params):
    """Single step action predictor when there is a single batch dim."""
    del config
    with tf.variable_scope(scope, reuse=reuse, use_resource=True):
      with tf.variable_scope('state_features', reuse=reuse, use_resource=True):
        feature_points, end_points = vision_layers.BuildImagesToFeaturesModel(
            features.image,
            is_training=(mode == TRAIN),
            normalizer_fn=contrib_layers.layer_norm)

      if context_fn:
        feature_points = context_fn(feature_points)

      if params and params.get('is_inner_loop', False):
        if self._predict_con_gripper_pose:
          gripper_pose = self._predict_gripper_pose(feature_points)
        else:
          gripper_pose = tf.zeros_like(features.gripper_pose)
      else:
        gripper_pose = features.gripper_pose

      action, _ = vision_layers.BuildImageFeaturesToPoseModel(
          feature_points, aux_input=gripper_pose, num_outputs=self._action_size)
      action = self._output_mean + self._output_stddev * action
    return {
        'inference_output': action,
        'image': features.image,
        'feature_points': feature_points,
        'softmax': end_points['softmax'],
    }

  def a_func(self,
             features,
             scope,
             mode,
             context_fn = None,
             reuse=tf.AUTO_REUSE,
             config = None,
             params = None
            ):
    """Single step action predictor. See parent class."""
    return meta_tfdata.multi_batch_apply(self.single_batch_a_func, 2, features,
                                         scope, mode, context_fn, reuse, config,
                                         params)

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None
                    ):
    """Output learned loss if inner loop, or behavior clone if outer loop."""
    if params and params.get('is_outer_loss', False):
      # Outer loss case: use standard RegressionModel loss.
      return self.loss_fn(labels, inference_outputs, mode, params)
    # Inner loss case: compute learned loss function.
    with tf.variable_scope(
        'learned_loss', reuse=tf.AUTO_REUSE, use_resource=True):
      predicted_action, _ = meta_tfdata.multi_batch_apply(
          vision_layers.BuildImageFeaturesToPoseModel,
          2,
          inference_outputs['feature_points'],
          num_outputs=self._action_size)
      if self._learned_loss_conv1d_layers is None:
        return tf.losses.mean_squared_error(predicted_action,
                                            inference_outputs['action'])
      ll_input = tf.concat([
          predicted_action, inference_outputs['feature_points'],
          inference_outputs['inference_output']
      ], -1)
      net = ll_input
      for num_filters in self._learned_loss_conv1d_layers[:-1]:
        net = tf.layers.conv1d(
            net, num_filters, 10, activation=tf.nn.relu, use_bias=False)
        net = contrib_layers.layer_norm(net)
      net = tf.layers.conv1d(net, self._learned_loss_conv1d_layers[-1],
                             1)  # 1x1 convolution.
      return tf.reduce_mean(tf.reduce_sum(tf.square(net), axis=(1, 2)))
