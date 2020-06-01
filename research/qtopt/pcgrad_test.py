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

# Lint as: python3
"""Tests for tensor2robot.research.qtopt.pcgrad."""

import numpy as np
from tensor2robot.research.qtopt import pcgrad
import tensorflow.compat.v1 as tf


class PcgradTest(tf.test.TestCase):

  def testPCgradBasic(self):
    tf.disable_eager_execution()
    for dtype in [tf.dtypes.float32, tf.dtypes.float64]:
      with self.session(graph=tf.Graph()):
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        const0_np = np.array([1., 0.], dtype=dtype.as_numpy_dtype)
        const1_np = np.array([-1., -1.], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np, dtype=dtype)
        const0 = tf.constant(const0_np)
        const1 = tf.constant(const1_np)
        loss0 = tf.tensordot(var0, const0, 1)
        loss1 = tf.tensordot(var0, const1, 1)

        learning_rate = lambda: 0.001
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        losses = loss0 + loss1
        opt_grads = opt.compute_gradients(losses, var_list=[var0])

        pcgrad_opt = pcgrad.PCGrad(
            tf.train.GradientDescentOptimizer(learning_rate))
        pcgrad_col_opt = pcgrad.PCGrad(
            tf.train.GradientDescentOptimizer(learning_rate),
            use_collection_losses=True)
        losses = [loss0, loss1]
        pcgrad_grads = pcgrad_opt.compute_gradients(losses, var_list=[var0])
        tf.add_to_collection(pcgrad.PCGRAD_LOSSES_COLLECTION, loss0)
        tf.add_to_collection(pcgrad.PCGRAD_LOSSES_COLLECTION, loss1)
        pcgrad_grads_collection = pcgrad_col_opt.compute_gradients(
            None, var_list=[var0])

        with tf.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEmpty(opt.variables())

        self.evaluate(tf.global_variables_initializer())
        grad_vec, pcgrad_vec, pcgrad_col_vec = self.evaluate(
            [opt_grads, pcgrad_grads, pcgrad_grads_collection])
        # Make sure that both methods take grads of the same vars.
        self.assertAllCloseAccordingToType(pcgrad_vec, pcgrad_col_vec)
        self.assertAllCloseAccordingToType(grad_vec[0][1], pcgrad_vec[0][1])
        self.assertAllCloseAccordingToType(grad_vec[0][0], [0.0, -1.0])
        self.assertAllCloseAccordingToType(pcgrad_vec[0][0], [0.5, -1.5])

        self.evaluate(opt.apply_gradients(pcgrad_grads))
        self.assertAllCloseAccordingToType(
            self.evaluate(var0), [0.9995, 2.0015])


if __name__ == "__main__":
  tf.test.main()
