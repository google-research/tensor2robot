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

# Lint as: python3
"""Tests for tensor2robot.predictors.saved_model_v2_predictor."""

import os
import numpy as np

from tensor2robot.predictors import saved_model_v2_predictor
from tensor2robot.proto import t2r_pb2
from tensor2robot.utils import mocks
from tensor2robot.utils import tensorspec_utils

from tensorflow import estimator as tf_estimator
from tensorflow.compat.v1 import estimator as tf_compat_v1_estimator
import tensorflow.compat.v2 as tf

_BATCH_SIZE = 2


def setUpModule():
  tf.enable_v2_behavior()


def _generate_assets(model, export_dir):
  in_feature_spec = model.get_feature_specification_for_packing(
      mode=tf_estimator.ModeKeys.PREDICT)
  in_label_spec = model.get_label_specification_for_packing(
      mode=tf_compat_v1_estimator.ModeKeys.PREDICT)

  in_feature_spec = tensorspec_utils.filter_required_flat_tensor_spec(
      in_feature_spec)
  in_label_spec = tensorspec_utils.filter_required_flat_tensor_spec(
      in_label_spec)

  t2r_assets = t2r_pb2.T2RAssets()
  t2r_assets.feature_spec.CopyFrom(in_feature_spec.to_proto())
  t2r_assets.label_spec.CopyFrom(in_label_spec.to_proto())
  t2r_assets_dir = os.path.join(export_dir,
                                tensorspec_utils.EXTRA_ASSETS_DIRECTORY)

  tf.io.gfile.makedirs(t2r_assets_dir)
  t2r_assets_filename = os.path.join(t2r_assets_dir,
                                     tensorspec_utils.T2R_ASSETS_FILENAME)
  tensorspec_utils.write_t2r_assets_to_file(t2r_assets, t2r_assets_filename)


class SavedModelV2PredictorTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(SavedModelV2PredictorTest, self).__init__(*args, **kwargs)
    self._saved_model_path = None

  def _save_model(self, model, sample_features):
    if self._saved_model_path:
      return self._saved_model_path

    # Save inference_network_fn as the predict method for the saved_model.
    @tf.function(autograph=False)
    def predict(features):
      return model.inference_network_fn(features, None,
                                        tf_compat_v1_estimator.ModeKeys.PREDICT)

    # Call the model for the tf.function tracing side effects.
    predict(sample_features)
    model.predict = predict

    self._saved_model_path = self.create_tempdir().full_path
    tf.saved_model.save(model, self._saved_model_path)
    _generate_assets(model, self._saved_model_path)
    return self._saved_model_path

  def _test_predictor(self, predictor_cls, multi_dataset):
    mock_model = mocks.MockTF2T2RModel(multi_dataset=multi_dataset)

    # Generate a sample to evaluate
    feature_spec = mock_model.preprocessor.get_in_feature_specification(
        tf_compat_v1_estimator.ModeKeys.PREDICT)
    sample_features = tensorspec_utils.make_random_numpy(
        feature_spec, batch_size=_BATCH_SIZE)

    # Generate a saved model and load it.
    path = self._save_model(mock_model, sample_features)
    saved_model_predictor = predictor_cls(path)

    # Not restored yet.
    with self.assertRaises(ValueError):
      saved_model_predictor.predict(sample_features)

    saved_model_predictor.restore()

    # Validate evaluations are the same afterwards.
    original_model_out = mock_model.inference_network_fn(
        sample_features, None, tf_compat_v1_estimator.ModeKeys.PREDICT)

    predictor_out = saved_model_predictor.predict(sample_features)

    np.testing.assert_almost_equal(original_model_out['logits'],
                                   predictor_out['logits'])

  def testTF1PredictorSingleDataset(self):
    self._test_predictor(saved_model_v2_predictor.SavedModelTF1Predictor, False)

  def testTF1PredictorMultiDataset(self):
    self._test_predictor(saved_model_v2_predictor.SavedModelTF1Predictor, True)

  def testTF2PredictorSingleDataset(self):
    self._test_predictor(saved_model_v2_predictor.SavedModelTF2Predictor, False)

  def testTF2PredictorMultiDataset(self):
    self._test_predictor(saved_model_v2_predictor.SavedModelTF2Predictor, True)


if __name__ == '__main__':
  tf.test.main()
