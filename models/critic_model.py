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

"""TFModel abstract subclasses."""

import abc
from typing import Callable, Optional, Text

from absl import flags
import gin
import six
from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

FLAGS = flags.FLAGS
TRAIN = tf_estimator.ModeKeys.TRAIN
EVAL = tf_estimator.ModeKeys.EVAL
PREDICT = tf_estimator.ModeKeys.PREDICT

RunConfigType = abstract_model.RunConfigType
ParamsType = abstract_model.ParamsType
DictOrSpec = abstract_model.DictOrSpec
ModelTrainOutputType = abstract_model.ModelTrainOutputType
ExportOutputType = abstract_model.ExportOutputType


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class CriticModel(abstract_model.AbstractT2RModel):
  """Critic model with continuous actions trained using MC returns."""

  def __init__(
      self,
      loss_function = tf.losses.mean_squared_error,
      action_batch_size = None,
      **kwargs):
    """Constructor for ContinuousMCModel.

    Args:
      loss_function: Python function taking in (labels, predictions) that builds
        loss tensor.
      action_batch_size: If specified, a tiling of actions for prediction along
        a sub-dimension.
      **kwargs: Additional arguments for the TFModel parent class.
    """
    super(CriticModel, self).__init__(**kwargs)
    self._loss_function = loss_function
    self._action_batch_size = action_batch_size
    self._tile_actions_for_predict = action_batch_size is not None

    # Rigid separation of state and action features, as they are treated
    # differently. State features are often duplicated across the examples in a
    # batch via a broadcast rule, while action features are unique to each
    # example in a batch. The state features and action features themselves
    # should not be nested.

  @abc.abstractmethod
  def get_action_specification(self):
    """Gets model inputs (including context) for the action for inference.

    Returns:
      action_params_spec: A named tuple with fields for the action.
        The action features holds all tensors that are unique to each action.
    """

  @abc.abstractmethod
  def get_state_specification(self):
    """Gets model inputs (including context) for the state for inference.

    Returns:
      state_params_spec: A named tuple with fields for the state.
      The state features are shared by all potential actions.
    """

  def pack_state_action_to_feature_spec(
      self, state_params,
      action_params
  ):
    """Gets a feature spec namedtuple from the state and action.

    Args:
      state_params: Instance of state_spec_class.
      action_params: Instance of action_spec_class.

    Returns:
      feature_spec: An instance of self.feature_spec_class. This contains
        features for both the action and state.
    """
    return tensorspec_utils.TensorSpecStruct(
        state=state_params, action=action_params)

  def get_feature_specification(
      self, mode):
    """Gets model inputs (incl.

    context) for inference.

    Returns:
      feature_spec: A named tuple with fields for both the state and action.
      The state features are shared by all potential actions. The action
      component holds all tensors that are unique to each potential action.
    Arguments:
      mode: The mode for feature specifications
    """
    feature_spec = tensorspec_utils.TensorSpecStruct(
        state=self.get_state_specification(),
        action=self.get_action_specification())

    if mode == tf_estimator.ModeKeys.PREDICT and self._tile_actions_for_predict:

      def _expand_spec(spec):
        new_shape = (
            tf.TensorShape([self._action_batch_size]).concatenate(spec.shape))
        return tensorspec_utils.ExtendedTensorSpec.from_spec(
            spec, shape=new_shape)

      tiled_action_spec = tf.nest.map_structure(_expand_spec,
                                                self.get_action_specification())

      return tensorspec_utils.TensorSpecStruct(
          state=self.get_state_specification(), action=tiled_action_spec)
    return feature_spec

  @abc.abstractmethod
  def q_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None,
             reuse=tf.AUTO_REUSE):
    """Q(state, action) value function.

    We only need to define the q_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      scope: String specifying variable scope.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or contrib_tpu.RunConfig) Will
        receive what is passed to Estimator in config parameter, or the default
        config (tf.estimator.RunConfig). Allows updating things in your model_fn
        based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.
      reuse: Whether or not to reuse variables under variable scope 'scope'.

    Returns:
      outputs: A {key: Tensor} mapping. The key 'q_predicted' is required.
    """

  def loss_fn(self, features,
              labels,
              inference_outputs):
    """Convenience function for critic models.

    We only need to define the q_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      features: TensorSpecStruct encapsulating input features.
      labels: This is the second item returned from the input_fn and parsed by
        self._extract_and_validate_inputs. A dictionary which fulfills the
        requirements of the self.get_labels_spefication.
      inference_outputs: A dict containing the output tensors of
        model_inference_fn.

    Returns:
      A scalar loss tensor.
    """
    del features
    return self._loss_function(
        labels=labels.reward, predictions=inference_outputs['q_predicted'])

  def inference_network_fn(
      self,
      features,
      labels,
      mode,
      config = None,
      params = None):
    """See base class."""
    del labels

    outputs = self.q_func(
        features=features,
        mode=mode,
        scope='q_func',
        config=config,
        params=params,
        reuse=tf.AUTO_REUSE)
    if isinstance(outputs, tuple):
      update_ops = outputs[1]
      outputs = outputs[0]
    else:
      update_ops = None

    if not isinstance(outputs, dict):
      raise ValueError('The output of q_func is expected to be a dict.')

    if 'q_predicted' not in outputs:
      raise ValueError('For critic models q_predicted is a required key in '
                       'outputs but is not in {}.'.format(list(outputs.keys())))

    if self.use_summaries(params):
      tf.summary.histogram('q_t_predicted', outputs['q_predicted'])
    return outputs, update_ops

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None):
    """See base class."""
    del mode, config, params
    loss = self.loss_fn(features, labels, inference_outputs)
    return loss
