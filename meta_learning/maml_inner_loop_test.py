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

# Lint as: python2, python3
"""Tests for learning.estimator_models.meta_learning.maml_inner_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from six.moves import range
from tensor2robot.meta_learning import maml_inner_loop
import tensorflow.compat.v1 as tf
from tensorflow.contrib import graph_editor as contrib_graph_editor

LEARNING_RATE = 0.001
TARGET = 'target'
TARGET_VALUE = 0.0
NUM_STEPS = 3
COEFF_A = 'coeff_a'
COEFF_A_VALUE = 0.25
X_INIT = 2.0


def create_inputs():
  features = {
      COEFF_A: tf.placeholder(name=COEFF_A, shape=(1,), dtype=tf.float32),
  }
  labels = {
      TARGET: tf.constant([TARGET_VALUE], dtype=tf.float32)
  }
  return features, labels


def inference_network_fn(features, labels=None, mode=None, params=None):
  del labels, mode, params
  x = tf.get_variable(
      'x',
      shape=(1,),
      dtype=tf.float32,
      initializer=tf.constant_initializer([X_INIT], dtype=tf.float32))
  return {'prediction': x * features[COEFF_A]}


def model_train_fn(features,
                   labels,
                   inference_outputs,
                   mode=None,
                   config=None,
                   params=None):
  del features, mode, config, params
  return tf.losses.mean_squared_error(
      labels=labels[TARGET], predictions=inference_outputs['prediction'])


def learned_model_train_fn(features,
                           labels,
                           inference_outputs,
                           mode=None,
                           config=None,
                           params=None):
  """A model_train_fn where the loss function itself is learned."""
  del features, labels, mode, config, params
  with tf.variable_scope('learned_loss', reuse=tf.AUTO_REUSE):
    learned_label = tf.get_variable(
        'learned_label',
        shape=(1,),
        dtype=tf.float32,
        initializer=tf.constant_initializer([1.0], dtype=tf.float32))
  return tf.losses.mean_squared_error(
      labels=learned_label, predictions=inference_outputs['prediction'])


def dummy_inner_loop(inputs_list, maml_inner_loop_instance):
  # This is a dummy optimization. We simply do several gradient steps
  # in the inner loop and try to minimize coeff_a * x**2.
  # The outer loop tries to optimize the same thing. Note, since
  # we have the inner loop and we do gradient descent, we might overshoot
  # and thus the loss might increase despite having a good starting x.
  inner_losses = []
  with tf.variable_scope(
      'inner_loop',
      custom_getter=maml_inner_loop_instance._create_variable_getter_fn()):
    for train_features, train_labels in inputs_list[:-1]:
      outputs = inference_network_fn(
          features=train_features, labels=train_labels)
      train_fn_result = model_train_fn(
          features=train_features,
          labels=train_labels,
          inference_outputs=outputs)
      if isinstance(train_fn_result, tf.Tensor):
        train_loss = train_fn_result
      elif isinstance(train_fn_result, tuple):
        train_loss = train_fn_result[0]
      else:
        raise ValueError('The model_train_fn should return a '
                         'tuple(loss, train_outputs) or loss.')
      maml_inner_loop_instance._compute_and_apply_gradients(train_loss)
      inner_losses.append(train_loss)

    val_features, val_labels = inputs_list[-1]
    outputs = inference_network_fn(
        features=val_features, labels=val_labels)
    train_fn_result = model_train_fn(
        features=val_features,
        labels=val_labels,
        inference_outputs=outputs)
    if isinstance(train_fn_result, tf.Tensor):
      train_loss = train_fn_result
    elif isinstance(train_fn_result, tuple):
      train_loss = train_fn_result[0]
    else:
      raise ValueError('The model_train_fn should return a '
                       'tuple(loss, train_outputs) or loss.')
    return train_loss, inner_losses


class MAMLInnerLoopGradientDescentTest(
    parameterized.TestCase, tf.test.TestCase):

  def test_default_initialization(self):
    maml_inner_loop_instance = maml_inner_loop.MAMLInnerLoopGradientDescent(
        learning_rate=LEARNING_RATE)

    with self.session() as sess:
      features, labels = create_inputs()

      # We have not populated our custom getter scope yet.
      with self.assertRaises(ValueError):
        maml_inner_loop_instance._compute_and_apply_gradients(None)

      # Our custom getter variable cache has not been populated.
      self.assertEmpty(maml_inner_loop_instance._custom_getter_variable_cache)
      with tf.variable_scope(
          'init_variables',
          custom_getter=maml_inner_loop_instance._create_variable_getter_fn()):
        outputs = inference_network_fn(features=features)
        loss = model_train_fn(
            features=features,
            labels=labels,
            inference_outputs=outputs)
        # Now we have variables cached in our custom getter cache.
        self.assertNotEmpty(
            maml_inner_loop_instance._custom_getter_variable_cache)

        # Initially we have nothing in the variable cache.
        self.assertEmpty(maml_inner_loop_instance._variable_cache)
        maml_inner_loop_instance._compute_and_apply_gradients(loss)

        # compute_and_apply_gradients has populated the cache.
        self.assertLen(maml_inner_loop_instance._variable_cache, 1)

        sess.run(tf.global_variables_initializer())
        variables = sess.run(maml_inner_loop_instance._variable_cache[0])
        self.assertEqual(variables['init_variables/x'], X_INIT)

  @parameterized.parameters((False,), (True,))
  def test_inner_loop_internals(self, learn_inner_lr):
    tensors = []
    # We iterate over the two options to make sure that the second order graph
    # is indeed larger than the "first order" graph.
    for use_second_order in [False, True]:
      graph = tf.Graph()
      with tf.Session(graph=graph) as sess:
        maml_inner_loop_instance = maml_inner_loop.MAMLInnerLoopGradientDescent(
            learning_rate=LEARNING_RATE,
            use_second_order=use_second_order,
            learn_inner_lr=learn_inner_lr)

        inputs = create_inputs()
        features, _ = inputs
        outer_loss, inner_losses = dummy_inner_loop(
            [inputs, inputs, inputs], maml_inner_loop_instance)

        sess.run(tf.global_variables_initializer())

        # Here we check that we actually improved with our gradient descent
        # steps.
        np_inner_losses, np_outer_loss = sess.run(
            [inner_losses, outer_loss],
            feed_dict={features[COEFF_A]: [COEFF_A_VALUE]})

        # Verify that we make progress in the inner loss with every step.
        # We know this is true for the first sequence. The learning rate is
        # small enough such that we do not overshoot.
        previous_loss_value = np_inner_losses[0]
        for loss_value in np_inner_losses[1:]:
          self.assertLess(loss_value, previous_loss_value)
          previous_loss_value = loss_value

        # The last inner loss has one gradient step less, which is why it's
        # value should be higher.
        self.assertLess(np_outer_loss, np_inner_losses[-1])

        # Now we optimize the outer loop.
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(outer_loss)

        # Again we know that the x sequence is converging, the loss might
        # not be go down monotonically due to the inner loop though.
        x_previous = sess.run(
            list(maml_inner_loop_instance._variable_cache[0].values())[0])
        for _ in range(10):
          sess.run([train_op], feed_dict={features[COEFF_A]: [COEFF_A_VALUE]})
          x_new = sess.run(
              list(maml_inner_loop_instance._variable_cache[0].values())[0])
          self.assertLess(x_new, x_previous)
          x_previous = x_new

        tensors.append(contrib_graph_editor.get_tensors(tf.get_default_graph()))

    # When we use second order, we have a larger graph due to the additional
    # required computation nodes.
    self.assertLess(len(tensors[0]), len(tensors[1]))

  @parameterized.parameters((False,), (True,))
  def test_inner_loop(self, learn_inner_lr):
    tensors = []
    # We iterate over the two options to make sure that the second order graph
    # is indeed larger than the "first order" graph.
    for use_second_order in [False, True]:
      graph = tf.Graph()
      with tf.Session(graph=graph) as sess:
        maml_inner_loop_instance = maml_inner_loop.MAMLInnerLoopGradientDescent(
            learning_rate=LEARNING_RATE,
            use_second_order=use_second_order,
            learn_inner_lr=learn_inner_lr)
        inputs = create_inputs()
        features, labels = inputs
        outputs, _, _ = maml_inner_loop_instance.inner_loop(
            [inputs, inputs, inputs], inference_network_fn, model_train_fn)

        # outputs 0 is unconditioned 1 conditioned.
        outputs = outputs[1]

        outer_loss = model_train_fn(
            features=features, labels=labels, inference_outputs=outputs)
        # Now we optimize the outer loop.
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(outer_loss)

        # Initialize the variables manually.
        sess.run(tf.global_variables_initializer())

        # We make sure we can run the inner loop.
        sess.run(outputs, feed_dict={features[COEFF_A]: [COEFF_A_VALUE]})

        # We make sure we can run the inner loop.
        sess.run(train_op, feed_dict={features[COEFF_A]: [COEFF_A_VALUE]})

        # We know that the x sequence is converging, the loss might
        # not be go down monotonically due to the inner loop though.
        x_previous = sess.run(
            list(maml_inner_loop_instance._variable_cache[0].values())[0])
        for _ in range(10):
          sess.run([train_op], feed_dict={features[COEFF_A]: [COEFF_A_VALUE]})
          x_new = sess.run(
              list(maml_inner_loop_instance._variable_cache[0].values())[0])
          self.assertLess(x_new, x_previous)
          x_previous = x_new

        tensors.append(contrib_graph_editor.get_tensors(tf.get_default_graph()))

    # When we use second order, we have a larger graph due to the additional
    # required computation nodes.
    self.assertLess(len(tensors[0]), len(tensors[1]))

  @parameterized.parameters((False,), (True,))
  def test_inner_loop_reuse(self, learn_inner_lr):
    # Inner loop should create as many trainable vars in 'inner_loop' scope as a
    # direct call to inference_network_fn would. Learned learning rates and
    # learned loss variables should be created *outside* the 'inner_loop' scope
    # since they do not adapt.
    graph = tf.Graph()
    with tf.Session(graph=graph):
      inputs = create_inputs()
      features, _ = inputs
      # Record how many trainable vars a call to inference_network_fn creates.
      with tf.variable_scope('test_scope'):
        inference_network_fn(features)
      expected_num_train_vars = len(tf.trainable_variables(scope='test_scope'))
      maml_inner_loop_instance = maml_inner_loop.MAMLInnerLoopGradientDescent(
          learning_rate=LEARNING_RATE, learn_inner_lr=learn_inner_lr)
      maml_inner_loop_instance.inner_loop(
          [inputs, inputs, inputs],
          inference_network_fn,
          learned_model_train_fn)
      num_train_vars = len(tf.trainable_variables(scope='inner_loop'))
      self.assertEqual(expected_num_train_vars, num_train_vars)


if __name__ == '__main__':
  tf.test.main()
