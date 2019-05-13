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

"""TFModel abstract subclasses."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
from absl import flags
import gin
import six

from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf

from typing import Any, Dict, Optional, Text

FLAGS = flags.FLAGS
TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

RunConfigType = abstract_model.RunConfigType
ParamsType = abstract_model.ParamsType
DictOrSpec = abstract_model.DictOrSpec
ModelTrainOutputType = abstract_model.ModelTrainOutputType
ExportOutputType = abstract_model.ExportOutputType


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class ClassificationModel(abstract_model.AbstractT2RModel):
  """Classification model."""

  def __init__(self, loss_function=tf.losses.log_loss, **kwargs):
    """Constructor for ClassificationModel.

    Args:
      loss_function: Python function taking in (labels, predictions) that builds
        loss tensor.
      **kwargs: Additional arguments for the TFModel parent class.
    """
    super(ClassificationModel, self).__init__(**kwargs)
    self._loss_function = loss_function
    self._label_specification = None
    self._state_specification = None

  def get_label_specification(
      self, mode):
    del mode
    return self._label_specification

  def get_feature_specification(
      self, mode):
    """Gets model inputs (including context) for inference.

    Arguments:
      mode: The mode for feature specifications

    Returns:
      feature_spec: A named tuple with fields for the state.
    """
    del mode
    return tensorspec_utils.TensorSpecStruct(state=self.state_specification)

  @property
  def state_specification(self):
    return self._state_specification

  @state_specification.setter
  def state_specification(self, value):
    self._state_specification = value

  @state_specification.setter
  def label_specification(self, value):
    self._label_specification = value

  @abc.abstractmethod
  def a_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None):
    """The F(state) function.

    We only need to define the a_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      scope: String specifying variable scope.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or tf.contrib.tpu.RunConfig) Will
        receive what is passed to Estimator in config parameter, or the default
        config (tf.estimator.RunConfig). Allows updating things in your model_fn
        based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters.

    Returns:
      outputs: A {key: Tensor} mapping. The key 'q_predicted' is required.
    """

  def loss_fn(self, labels, inference_outputs):
    """Convenience function for classification models.

    We only need to define the a_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      labels: This is the second item returned from the input_fn and parsed by
        self._extract_and_validate_inputs. A dictionary which fulfills the
        requirements of the self.get_labels_spefication.
      inference_outputs: A dict containing the output tensors of
        model_inference_fn.

    Returns:
      A scalar loss tensor.
    """
    return self._loss_function(
        labels=labels.classes, predictions=inference_outputs['a_predicted'])

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config = None,
                           params = None):
    """See base class."""
    del labels

    outputs = self.a_func(
        features, scope='a_func', mode=mode, params=params, config=config)

    if not isinstance(outputs, dict):
      raise ValueError('The output of a_func is expected to be a dict.')

    if 'a_predicted' not in outputs:
      raise ValueError('For classification models a_predicted is a required '
                       'key in outputs but is not in {}.'.format(
                           outputs.keys()))

    if self.use_summaries(params):
      tf.summary.histogram('a_t_predicted', outputs['a_predicted'])
    return outputs

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None):
    """See base class."""
    del features, mode, config, params
    loss = self.loss_fn(labels, inference_outputs)
    return loss

  def create_export_outputs_fn(self,
                               features,
                               inference_outputs,
                               mode,
                               config = None,
                               params = None):
    """See base class."""
    del features, mode, config, params
    predictions = {'prediction': inference_outputs['a_predicted']}
    return predictions

  def pack_state_to_feature_spec(self,
                                 state_params
                                ):
    """Packs the state feature spec from the state.

    Args:
      state_params: Instance of state_spec_class.

    Returns:
      feature_spec: An instance of self.feature_spec_class. This contains
        features for the state.
    """
    feature_spec = tensorspec_utils.TensorSpecStruct(state=state_params)
    return feature_spec

  def model_eval_fn(self,
                    features,
                    labels,
                    inference_outputs,
                    train_loss,
                    train_outputs,
                    mode,
                    config = None,
                    params = None):
    """See base class."""
    eval_mse = tf.metrics.mean_squared_error(
        labels=labels.classes,
        predictions=inference_outputs['a_predicted'],
        name='eval_mse')

    predictions_rounded = tf.round(inference_outputs['a_predicted'])

    eval_precision = tf.metrics.precision(
        labels=labels.classes,
        predictions=predictions_rounded,
        name='eval_precision')

    eval_accuracy = tf.metrics.accuracy(
        labels=labels.classes,
        predictions=predictions_rounded,
        name='eval_accuracy')

    eval_recall = tf.metrics.recall(
        labels=labels.classes,
        predictions=predictions_rounded,
        name='eval_recall')

    metric_fn = {
        'eval_mse': eval_mse,
        'eval_precision': eval_precision,
        'eval_accuracy': eval_accuracy,
        'eval_recall': eval_recall
    }

    return metric_fn
