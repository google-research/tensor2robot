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

"""Tests that we can run a few training steps (mock data) on T2R model."""

from absl import flags
from absl.testing import parameterized
from tensor2robot.research.qtopt import t2r_models
from tensor2robot.utils.t2r_test_fixture import T2RModelFixture
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

FLAGS = flags.FLAGS
TRAIN = tf_estimator.ModeKeys.TRAIN
MODEL_NAME = 'Grasping44E2EOpenCloseTerminateGripperStatusHeightToBottom'


class GraspPredictT2RTest(parameterized.TestCase):

  def setUp(self):
    super(GraspPredictT2RTest, self).setUp()
    self._fixture = T2RModelFixture(
        test_case=self,
        use_tpu=False,
    )

  @parameterized.parameters(
      (MODEL_NAME,))
  def test_random_train(self, model_name):
    self._fixture.random_train(
        module_name=t2r_models, model_name=model_name)

  @parameterized.parameters(
      (MODEL_NAME,))
  def test_inference(self, model_name):
    result = self._fixture.random_predict(
        t2r_models, model_name, action_batch_size=64)
    self.assertIsNotNone(result)
    self.assertDictContainsSubset({'global_step': 0}, result)


if __name__ == '__main__':
  tf.test.main()
