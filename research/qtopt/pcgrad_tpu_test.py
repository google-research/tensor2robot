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

from tensor2robot.research.qtopt import pcgrad
import tensorflow.compat.v1 as tf


class PcgradTest(tf.test.TestCase):

  def testPCgradNetworkTPU(self):
    tf.reset_default_graph()
    tf.disable_eager_execution()
    learning_rate = lambda: 0.001
    def pcgrad_computation():
      x = tf.constant(1., shape=[64, 472, 472, 3])
      layers = [
          tf.keras.layers.Conv2D(filters=64, kernel_size=3),
          tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
          tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
      ]
      y = x
      for layer in layers:
        y = layer(y)
      n_tasks = 10
      task_loss_0 = tf.reduce_sum(y)
      task_losses = [task_loss_0 * (1. + (n / 10.)) for n in range(n_tasks)]

      pcgrad_opt = pcgrad.PCGrad(
          tf.train.GradientDescentOptimizer(learning_rate))
      pcgrad_grads_and_vars = pcgrad_opt.compute_gradients(
          task_losses, var_list=tf.trainable_variables())
      return pcgrad_opt.apply_gradients(pcgrad_grads_and_vars)

    tpu_computation = tf.compat.v1.tpu.batch_parallel(pcgrad_computation,
                                                      num_shards=2)
    self.evaluate(tf.compat.v1.tpu.initialize_system())
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.evaluate(tpu_computation)
    self.evaluate(tf.compat.v1.tpu.shutdown_system())

if __name__ == "__main__":
  tf.test.main()
