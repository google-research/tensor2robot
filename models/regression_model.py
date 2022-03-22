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

# Lint as python3
"""TFModel abstract subclasses."""

import abc
from typing import Optional, Text
import warnings

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
warnings.simplefilter('always', DeprecationWarning)


@gin.configurable
@six.add_metaclass(abc.ABCMeta)
class RegressionModel(abstract_model.AbstractT2RModel):
  """Continuous-valued output using mean-squared error on target values."""

  def __init__(self, action_size=2, **kwargs):
    warnings.warn(
        'RegressionModel is deprecated. Subclass AbstractT2RModel instead',
        DeprecationWarning, stacklevel=2)

    super(RegressionModel, self).__init__(**kwargs)
    self._action_size = action_size

  @abc.abstractmethod
  def a_func(self,
             features,
             scope,
             mode,
             config = None,
             params = None,
             reuse=tf.AUTO_REUSE):
    """A(state) regression function.

    This function can return a stochastic or a deterministic tensor.

    We only need to define the a_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
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

    Returns:
      outputs: A {key: Tensor} mapping. The key 'inference_output' is required.
    """

  def loss_fn(self,
              labels,
              inference_outputs,
              mode,
              params=None):
    """Convenience function for regression models.

    We only need to define the a_func and loss_fn to have a proper model.
    For more specialization please overwrite inference_network_fn, model_*_fn.

    Args:
      labels: This is the second item returned from the input_fn and parsed by
        self._extract_and_validate_inputs. A dictionary which fulfills the
        requirements of the self.get_labels_spefication.
      inference_outputs: A dict containing the output tensors of
        model_inference_fn.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Returns:
      A scalar loss tensor.
    """
    del mode, params
    return tf.losses.mean_squared_error(
        labels=labels.target, predictions=inference_outputs['inference_output'])

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config = None,
                           params = None):
    """See base class."""
    del labels
    outputs = self.a_func(
        features=features,
        mode=mode,
        scope='a_func',
        config=config,
        params=params,
        reuse=tf.AUTO_REUSE)

    if not isinstance(outputs, dict):
      raise ValueError('The output of a_func is expected to be a dict.')

    if 'inference_output' not in outputs:
      raise ValueError('For regression models inference_output is a required '
                       'key in outputs but is not in {}.'.format(
                           list(outputs.keys())))
    if self.use_summaries(params):
      tf.summary.histogram('inference_output', outputs['inference_output'])
    return outputs

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config = None,
                     params = None):
    """See base class."""
    del features, config
    loss = self.loss_fn(labels, inference_outputs, mode=mode, params=params)
    return loss

  def create_export_outputs_fn(self,
                               features,
                               inference_outputs,
                               mode,
                               config = None,
                               params = None):
    """See base class."""
    del features, mode, config, params
    return {'inference_output': inference_outputs['inference_output']}
