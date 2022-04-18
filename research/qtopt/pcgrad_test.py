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

"""Tests for tensor2robot.research.qtopt.pcgrad."""

from absl.testing import parameterized
import numpy as np
from tensor2robot.research.qtopt import pcgrad
import tensorflow.compat.v1 as tf


class PcgradTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (None, None, [0, 1]),
      (None, ['*var*'], [0, 1]),
      (['second*'], None, [0]),
      (None, ['first*'], [0]),
      (None, ['*0'], [0]),
      (['first*'], None, [1]),
      (['*var*'], None, []),
  )
  def testPCgradBasic(self,
                      denylist,
                      allowlist,
                      pcgrad_var_idx):
    tf.disable_eager_execution()
    for dtype in [tf.dtypes.float32, tf.dtypes.float64]:
      with self.session(graph=tf.Graph()):
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        const0_np = np.array([1., 0.], dtype=dtype.as_numpy_dtype)
        const1_np = np.array([-1., -1.], dtype=dtype.as_numpy_dtype)
        const2_np = np.array([-1., 1.], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np, dtype=dtype, name='first_var/var0')
        var1 = tf.Variable(var1_np, dtype=dtype, name='second_var/var1')
        const0 = tf.constant(const0_np)
        const1 = tf.constant(const1_np)
        const2 = tf.constant(const2_np)
        loss0 = tf.tensordot(var0, const0, 1) + tf.tensordot(var1, const2, 1)
        loss1 = tf.tensordot(var0, const1, 1) + tf.tensordot(var1, const0, 1)

        learning_rate = lambda: 0.001
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        losses = loss0 + loss1
        opt_grads = opt.compute_gradients(losses, var_list=[var0, var1])

        pcgrad_opt = pcgrad.PCGrad(
            tf.train.GradientDescentOptimizer(learning_rate),
            denylist=denylist,
            allowlist=allowlist)
        pcgrad_col_opt = pcgrad.PCGrad(
            tf.train.GradientDescentOptimizer(learning_rate),
            use_collection_losses=True,
            denylist=denylist,
            allowlist=allowlist)
        losses = [loss0, loss1]
        pcgrad_grads = pcgrad_opt.compute_gradients(
            losses, var_list=[var0, var1])
        tf.add_to_collection(pcgrad.PCGRAD_LOSSES_COLLECTION, loss0)
        tf.add_to_collection(pcgrad.PCGRAD_LOSSES_COLLECTION, loss1)
        pcgrad_grads_collection = pcgrad_col_opt.compute_gradients(
            None, var_list=[var0, var1])

        with tf.Graph().as_default():
          # Shouldn't return non-slot variables from other graphs.
          self.assertEmpty(opt.variables())

        self.evaluate(tf.global_variables_initializer())
        grad_vec, pcgrad_vec, pcgrad_col_vec = self.evaluate(
            [opt_grads, pcgrad_grads, pcgrad_grads_collection])
        # Make sure that both methods take grads of the same vars.
        self.assertAllCloseAccordingToType(pcgrad_vec, pcgrad_col_vec)

        results = [{
            'var': var0,
            'pcgrad_vec': [0.5, -1.5],
            'result': [0.9995, 2.0015]
        }, {
            'var': var1,
            'pcgrad_vec': [0.5, 1.5],
            'result': [2.9995, 3.9985]
        }]
        grad_var_idx = {0, 1}.difference(pcgrad_var_idx)

        self.assertAllCloseAccordingToType(
            grad_vec[0][0], [0.0, -1.0], atol=1e-5)
        self.assertAllCloseAccordingToType(
            grad_vec[1][0], [0.0, 1.0], atol=1e-5)
        pcgrad_vec_idx = 0
        for var_idx in pcgrad_var_idx:
          self.assertAllCloseAccordingToType(
              pcgrad_vec[pcgrad_vec_idx][0],
              results[var_idx]['pcgrad_vec'],
              atol=1e-5)
          pcgrad_vec_idx += 1

        for var_idx in grad_var_idx:
          self.assertAllCloseAccordingToType(
              pcgrad_vec[pcgrad_vec_idx][0], grad_vec[var_idx][0], atol=1e-5)
          pcgrad_vec_idx += 1

        self.evaluate(opt.apply_gradients(pcgrad_grads))
        self.assertAllCloseAccordingToType(
            self.evaluate([results[idx]['var'] for idx in pcgrad_var_idx]),
            [results[idx]['result'] for idx in pcgrad_var_idx])


if __name__ == '__main__':
  tf.test.main()
