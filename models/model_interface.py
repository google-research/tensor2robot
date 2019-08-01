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

"""The minimal T2RModel interface required by T2R infrastructure.

# TODO(T2R_CONTRIBUTORS): Sunset this interface in favor of AbstractT2RModel.
This is a proper interface used by T2R infrastructure such as train_eval,
input_generators, preprocessors, exporters, predictors.

For T2R model development it is highly recommended to inherit from any of the
abstract, classification, critic or regresssion model. These model abstractions
contain our best practices to write composable models by introducing additional
abstractions and re-usable functionality such as inference_network_fn and
model_train_fn etc.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
from tensor2robot.preprocessors import abstract_preprocessor
from tensor2robot.utils import tensorspec_utils
import tensorflow as tf
from typing import Any, Optional, Union, Text, Dict

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT
TensorSpecStruct = tensorspec_utils.TensorSpecStruct

RunConfigType = Optional[
    Union[tf.estimator.RunConfig, tf.contrib.tpu.RunConfig]]
ParamsType = Optional[Dict[Text, Any]]


class ModelInterface(object):
  """A minimal T2RModel interface used by T2R infrastructure."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_feature_specification_for_packing(self, mode):
    """Returns the feature_spec that create_pack_features expects."""

  @abc.abstractmethod
  def get_label_specification_for_packing(self, mode):
    """Returns the label_spec that create_pack_features expects."""

  @abc.abstractproperty
  def preprocessor(self):
    """Returns the preprocessor used to create preprocess model inputs."""

  @abc.abstractmethod
  def get_feature_specification(self, mode
                               ):
    """Required features for the model_fn.

    Note, the model_fn might use additional features for debugging/development
    purposes. The create_export_outputs_fn will however only require the
    specified required features. Only this subset of features will be used to
    generate automatic tf.Example extractors and numpy placeholders for the
    serving models.

    Args:
      mode: The mode for feature specifications
    """

  @abc.abstractmethod
  def get_label_specification(self,
                              mode):
    """Required labels for the model_fn.

    Note, the model_fn might use additional labels for debugging/development
    purposes.

    Args:
      mode: The mode for feature specifications
    """

  @abc.abstractmethod
  def get_run_config(self):
    """Get the RunConfig for Estimator model."""

  @abc.abstractmethod
  def get_tpu_run_config(self):
    """Get the TPU RunConfig for Estimator model."""

  @abc.abstractmethod
  def get_session_config(self):
    """Get the Session tf.ConfigProto for Estimator model."""

  @abc.abstractproperty
  def is_device_tpu(self):
    """Returns True if the device is TPU otherwise False."""

  @abc.abstractproperty
  def is_device_gpu(self):
    """Returns True if the device is GPU otherwise False."""

  @abc.abstractproperty
  def device_type(self):
    """Returns the device type string."""

  @abc.abstractmethod
  def model_fn(self,
               features,
               labels,
               mode,
               config = None,
               params = None):
    """Estimator model_fn.

    Args:
      features: This is the first item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      labels: This is the second item returned from the input_fn and parsed by
        tensorspec_utils.validate_and_pack. A spec_structure which fulfills the
        requirements of the self.get_feature_specification.
      mode: (ModeKeys) Specifies if this is training, evaluation or prediction.
      config: (Optional tf.estimator.RunConfig or tf.contrib.tpu.RunConfig) Will
        receive what is passed to Estimator in config parameter, or the default
        config (tf.estimator.RunConfig). Allows updating things in your model_fn
        based on  configuration such as num_ps_replicas, or model_dir.
      params: An optional dict of hyper parameters that will be passed into
        input_fn and model_fn. Keys are names of parameters, values are basic
        python types. There are reserved keys for TPUEstimator, including
        'batch_size'.

    Raises:
      ValueError: If the mode key is not supported, not in [PREDICT, TRAIN,
        EVAL].

    Returns:
      An EstimatorSpec.
    """
