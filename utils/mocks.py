# coding=utf-8
# Copyright 2020 The Tensor2Robot Authors.
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
"""Mock implementations for InputGenerator and TFModel for consistent testing."""

from typing import Any, Dict, Text, Optional

import numpy as np
from tensor2robot.export_generators import abstract_export_generator
from tensor2robot.input_generators import abstract_input_generator
from tensor2robot.models import abstract_model
from tensor2robot.utils import tensorspec_utils
from tensor2robot.utils import tfdata
import tensorflow.compat.v1 as tf

SEED = 1234
POSITIVE_SIZE = 64


class MockExportGenerator(abstract_export_generator.AbstractExportGenerator):

  def create_serving_input_receiver_numpy_fn(self, params=None):
    pass

  def create_serving_input_receiver_tf_example_fn(self,
                                                  params = None):
    pass


class MockInputGenerator(abstract_input_generator.AbstractInputGenerator):
  """A simple mock input generator, creating a linearly separable dataset.

  We generate negative and positive data samples with associated binary labels.
  """

  def __init__(self, multi_dataset=False, **kwargs):
    self._multi_dataset = multi_dataset
    super(MockInputGenerator, self).__init__(**kwargs)

  def create_numpy_data(self):
    """Create a deterministic dataset of linearly separable data.

    Returns:
      features: A numpy array with negative and positive data entries.
      labels: A numpy array with binary labels for the different input features.
    """
    np.random.seed(SEED)
    mock_positive_data = np.random.uniform(
        low=0.2, high=1.0, size=(POSITIVE_SIZE, 3))
    mock_negative_data = np.random.uniform(
        low=-1.0, high=-0.2, size=(POSITIVE_SIZE, 3))
    features = np.concatenate([mock_positive_data, mock_negative_data], axis=0)
    labels = np.concatenate([
        np.ones((mock_positive_data.shape[0], 1)),
        np.zeros((mock_positive_data.shape[0], 1))
    ],
                            axis=0)
    return features, labels

  def _create_dataset(self, mode, params=None):
    """See base class documentation."""
    batch_size = tfdata.get_batch_size(params, self._batch_size)

    features, labels = self.create_numpy_data()
    if self._multi_dataset:
      tf_features = {
          'x1': tf.constant(features, tf.float32),
          'x2': tf.constant(features, tf.float32),
      }
    else:
      tf_features = {
          'x': tf.constant(features, tf.float32),
      }

    tf_labels = {'y': tf.constant(labels, dtype=tf.float32)}
    dataset = tf.data.Dataset.from_tensor_slices((tf_features, tf_labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.repeat()
      dataset = dataset.shuffle(buffer_size=features.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if self._preprocess_fn is not None:
      dataset = dataset.map(self._preprocess_fn)
    return dataset


class MockT2RModel(abstract_model.AbstractT2RModel):
  """A simple mock network, implementing a feedforward network.

  This network contains 3 FC layers and results in a single scalar prediction.
  It can be used with the MockInputGenerator to learn how to separate
  positive from negative input features.
  """

  def __init__(self, multi_dataset=False, **kwargs):
    self._multi_dataset = multi_dataset
    super(MockT2RModel, self).__init__(**kwargs)

  def get_feature_specification(
      self, mode):
    """See base class documentation."""
    del mode
    spec_structure = tensorspec_utils.TensorSpecStruct()
    if self._multi_dataset:
      spec_structure.x1 = tensorspec_utils.ExtendedTensorSpec(
          shape=(3,),
          dtype=tf.float32,
          name='measured_position',
          dataset_key='dataset1')
      spec_structure.x2 = tensorspec_utils.ExtendedTensorSpec(
          shape=(3,),
          dtype=tf.float32,
          name='measured_position',
          dataset_key='dataset2')
    else:
      spec_structure.x = tensorspec_utils.ExtendedTensorSpec(
          shape=(3,), dtype=tf.float32, name='measured_position')
    return spec_structure

  def get_label_specification(
      self, mode):
    """See base class documentation."""
    del mode
    spec_structure = tensorspec_utils.TensorSpecStruct()
    if self._multi_dataset:
      spec_structure.y = tensorspec_utils.ExtendedTensorSpec(
          shape=(1,),
          dtype=tf.float32,
          name='valid_position',
          dataset_key='dataset1')
    else:
      spec_structure.y = tensorspec_utils.ExtendedTensorSpec(
          shape=(1,), dtype=tf.float32, name='valid_position')
    return spec_structure

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config=None,
                           params=None):
    """See base class documentation."""
    del mode, config, params
    if self._multi_dataset:
      net = features.x1 + features.x2
    else:
      net = features.x
    for pos, activations in enumerate([32, 16, 8]):
      # tf.keras does not support variable_scope and custom_getter.
      # Therefore, we cannot use this api yet for meta learning models.

      # Note, we have to add the MockTFModel name in order to support legacy
      # model loading.
      net = tf.layers.dense(
          net,
          units=activations,
          activation=tf.nn.elu,
          name='MockT2RModel.dense.{}'.format(pos))
      net = tf.layers.batch_normalization(
          net, name='MockT2RModel.batch_norm.{}'.format(pos))
    net = tf.layers.dense(net, units=1, name='MockT2RModel.dense.4')
    inference_outputs = {}
    inference_outputs['logit'] = net
    return inference_outputs

  def model_train_fn(self,
                     features,
                     labels,
                     inference_outputs,
                     mode,
                     config=None,
                     params=None):
    """See base class documentation."""
    loss = tf.keras.losses.categorical_hinge(
        y_true=labels.y, y_pred=inference_outputs['logit'])
    return tf.reduce_mean(loss)


class MockTF2T2RModel(MockT2RModel, tf.Module):
  """MockT2RModel compatible with Tf2.x and OOO Keras API."""

  def __init__(self, **kwargs):
    super(MockTF2T2RModel, self).__init__(**kwargs)
    self._model = None

  def _build_model(self):
    self._model = tf.keras.Sequential()
    if not self._multi_dataset:
      for pos, activations in enumerate([32, 16, 8]):
        self._model.add(
            tf.keras.layers.Dense(
                units=activations,
                activation=tf.keras.activations.elu,
                name='MockTF2T2RModel.dense.{}'.format(pos)))
        self._model.add(
            tf.keras.layers.BatchNormalization(
                name='MockTF2T2RModel.batch_norm.{}'.format(pos)))
    self._model.add(
        tf.keras.layers.Dense(units=1, name='MockTF2T2RModel.dense.4'))

  def inference_network_fn(self,
                           features,
                           labels,
                           mode,
                           config=None,
                           params=None):
    """See base class documentation."""
    del mode, config, params
    if not self._model:
      self._build_model()

    if self._multi_dataset:
      if tf.executing_eagerly():
        x1 = tf.convert_to_tensor(features.x1)
        x2 = tf.convert_to_tensor(features.x2)
      else:
        x1 = features.x1
        x2 = features.x2
      net = x1 + x2
    else:
      net = features.x

    net = self._model(net)
    return dict(logits=net)
