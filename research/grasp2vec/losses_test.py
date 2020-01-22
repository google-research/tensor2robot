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

"""Tests for tensor2robot.research.grasp2vec.losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensor2robot.research.grasp2vec import losses
import tensorflow.compat.v1 as tf  # tf


EMBEDDING = 512
BATCH_SIZE = 32
fake_data = {
    'pregrasp_embedding': np.random.random((BATCH_SIZE, EMBEDDING)),
    'postgrasp_embedding': np.random.random((BATCH_SIZE, EMBEDDING)),
    'goal_embedding': np.random.random((BATCH_SIZE, EMBEDDING)),
}


def cosine_distance(x, y):
  dots = np.sum(x*y, axis=1)
  norm_x = np.linalg.norm(x, axis=1)
  norm_y = np.linalg.norm(y, axis=1)
  return 1-dots/(norm_x*norm_y)


def l2_distance(x, y):
  dist = x-y
  return np.linalg.norm(dist, axis=1)**2


class LossesTest(parameterized.TestCase):

  def test_cosine_arithmetic_loss_zeros_mask(self):
    all_zeros_mask = np.zeros((BATCH_SIZE))
    loss = losses.CosineArithmeticLoss(fake_data['pregrasp_embedding'],
                                       fake_data['goal_embedding'],
                                       fake_data['postgrasp_embedding'],
                                       all_zeros_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    self.assertEqual(output, 0)

  def test_cosine_arithmetic_loss_ones_mask(self):
    all_ones_mask = np.ones((BATCH_SIZE))
    loss = losses.CosineArithmeticLoss(fake_data['pregrasp_embedding'],
                                       fake_data['goal_embedding'],
                                       fake_data['postgrasp_embedding'],
                                       all_ones_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    true_answer = cosine_distance(
        fake_data['pregrasp_embedding']-fake_data['postgrasp_embedding'],
        fake_data['goal_embedding']
    )

    self.assertAlmostEqual(output, np.mean(true_answer), places=3)

  def test_cosine_arithmetic_loss_mixed_mask(self):
    mixed_mask = np.zeros((BATCH_SIZE))
    mixed_mask[0] = 1
    loss = losses.CosineArithmeticLoss(fake_data['pregrasp_embedding'],
                                       fake_data['goal_embedding'],
                                       fake_data['postgrasp_embedding'],
                                       mixed_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    true_answer = cosine_distance(
        (fake_data['pregrasp_embedding'][:1]-
         fake_data['postgrasp_embedding'][:1]),
        fake_data['goal_embedding'][:1])
    self.assertAlmostEqual(output, true_answer[0], places=3)

  def test_l2_arithmetic_loss_zeros_mask(self):
    all_zeros_mask = np.zeros((BATCH_SIZE))
    loss = losses.L2ArithmeticLoss(fake_data['pregrasp_embedding'],
                                   fake_data['goal_embedding'],
                                   fake_data['postgrasp_embedding'],
                                   all_zeros_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    self.assertEqual(output, 0)

  def test_l2_arithmetic_loss_ones_mask(self):
    all_ones_mask = np.ones((BATCH_SIZE))
    loss = losses.L2ArithmeticLoss(fake_data['pregrasp_embedding'],
                                   fake_data['goal_embedding'],
                                   fake_data['postgrasp_embedding'],
                                   all_ones_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    true_answer = l2_distance(
        fake_data['pregrasp_embedding'],
        fake_data['postgrasp_embedding'] + fake_data['goal_embedding']
    )

    self.assertAlmostEqual(output, np.mean(true_answer), places=3)

  def test_l2_arithmetic_loss_mixed_mask(self):
    mixed_mask = np.zeros((BATCH_SIZE))
    mixed_mask[0] = 1
    loss = losses.L2ArithmeticLoss(fake_data['pregrasp_embedding'],
                                   fake_data['goal_embedding'],
                                   fake_data['postgrasp_embedding'],
                                   mixed_mask)
    with tf.Session() as sess:
      output = sess.run(loss)
    true_answer = l2_distance(
        fake_data['pregrasp_embedding'][:1],
        (fake_data['postgrasp_embedding'][:1]
         + fake_data['goal_embedding'][:1]))
    self.assertAlmostEqual(output, true_answer[0], places=3)

  @parameterized.named_parameters(
      ('CorrectKeypoints',
       np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
                dtype=np.float32),
       np.array([1, 3, 0, 2]),
       1.0),
      ('IncorrectKeypoints',
       np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
                dtype=np.float32),
       np.array([2, 0, 3, 1]),
       0.0),
      ('HalfcorrectKeypoints',
       np.array([[-0.6, -0.4], [-0.4, 0.1], [0.3, -0.6], [0.7, 0.9]],
                dtype=np.float32),
       np.array([1, 0, 0, 1]),
       0.5))
  def test_keypoint_accuracy(self, keypoints, labels, expected_accuracy):
    acc_op, _ = losses.KeypointAccuracy(keypoints, labels)
    with tf.Session() as sess:
      acc = sess.run(acc_op)

    self.assertEqual(acc, expected_accuracy)

  def test_npairs_loss_multilabel(self):
    sample_a = np.random.rand(16)
    sample_c = np.random.rand(16)
    sample_b = np.random.rand(16)/10
    pregrasp = np.array([sample_a+sample_b, sample_a, sample_c]).astype(
        np.float32)
    postgrasp = np.zeros(pregrasp.shape, np.float32)
    goal = pregrasp.copy()
    grasp_success = np.ones((3), np.int32)
    multilabel_loss_op = losses.NPairsLossMultilabel(pregrasp, goal, postgrasp,
                                                     grasp_success, {})
    singlelabel_loss_op = losses.NPairsLoss(pregrasp, goal, postgrasp, {})
    with tf.Session() as sess:
      multilabel_loss, singlelabel_loss = sess.run(
          [multilabel_loss_op, singlelabel_loss_op])
    self.assertAlmostEqual(multilabel_loss, singlelabel_loss, places=5)
    grasp_success = np.array([0, 0, 1])
    multilabel_loss_op = losses.NPairsLossMultilabel(pregrasp, goal, postgrasp,
                                                     grasp_success, {})
    singlelabel_loss_op = losses.NPairsLoss(pregrasp, goal, postgrasp, {})
    with tf.Session() as sess:
      multilabel_loss2, singlelabel_loss2 = sess.run(
          [multilabel_loss_op, singlelabel_loss_op])
    self.assertGreater(multilabel_loss2, singlelabel_loss2)
    self.assertGreater(multilabel_loss2, multilabel_loss)

if __name__ == '__main__':
  tf.test.main()
