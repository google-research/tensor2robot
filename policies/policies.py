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

"""Policies that use predictors."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc

import gin
import numpy as np
from tensor2robot.models import critic_model
from tensor2robot.models import regression_model
from tensor2robot.predictors import abstract_predictor
from tensor2robot.utils import cross_entropy
from typing import Any, Optional, Text

_BUSY_WAITING_SLEEP_TIME_IN_SECS = 10


@gin.configurable
class Policy(object):
  """Base Policy class."""

  __metaclass__ = abc.ABCMeta

  def __init__(
      self, predictor = None):
    self._predictor = predictor

  @abc.abstractmethod
  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    """Selects an action given the current state.

    The implementation of this method should not modify the state or context.

    Args:
      state: An arbitrary object representing the observed state.
      context: An arbitrary object representing additional contextual
        information, or None if no context is applicable.
      timestep: An integer representing the timestep (0-indexed) in the current
        episode.

    Returns:
      action: An arbitrary object representing the action to be taken.
    """

  def reset(self):
    """Reset the policy."""

  def init_randomly(self):
    """Initializes policy parameters from with random values."""
    if self._predictor is not None:
      self._predictor.init_randomly()

  def restore(self):
    """Restore policy parameters from a checkpoint."""
    if self._predictor is not None:
      self._predictor.restore()

  @property
  def model_path(self):
    if self._predictor is not None:
      return self._predictor.model_path
    return 'No model path defined.'

  @property
  def global_step(self):
    """The global step the model was saved with."""
    if self._predictor is not None:
      return self._predictor.model_version
    return 0

  def sample_action(self, obs, explore_prob):
    """Selects action for stepping from `dql_grasping:run_env` framework.

    Internal code uses `SelectAction` to choose actions. This method
    adds compatibility to the Policy abstraction used by dql_grasping.

    Args:
      obs: Observations from the environment.
      explore_prob: Probability of sampling from the non-greedy policy. This is
        ignored.

    Returns:
      Action: Action computed by the policy.
      debug: Optional debug information to be consumed by `run_env` and
        `episode_to_transitions` functions.
    """
    del explore_prob
    action = self.SelectAction(obs, None, None)
    debug = None
    return action, debug


@gin.configurable
class CEMPolicy(Policy):
  """CEM policy for continuous-action critic models."""

  # TODO(T2R_CONTRIBUTORS): Replace t2r_model with pack feature function.
  def __init__(self,
               t2r_model,
               action_size=2,
               cem_iters=3,
               cem_samples=64,
               num_elites=10,
               pack_fn=None,
               **parent_kwargs):
    super(CEMPolicy, self).__init__(**parent_kwargs)
    self._cem_iters = cem_iters
    self._cem_samples = cem_samples
    self._action_size = action_size
    self._num_elites = num_elites
    self.sample_fn = self._default_sample_fn
    if pack_fn is None:
      pack_fn = self._default_pack_fn
    self.pack_fn = pack_fn
    self._t2r_model = t2r_model

  def _default_sample_fn(self, mean, stddev):
    return mean + stddev * np.random.standard_normal((self._cem_samples,
                                                      self._action_size))

  def get_cem_action(self, objective_fn):
    """Returns CEM approximate argmax on an objective_fn.

    Args:
      objective_fn: A python callable that takes in a batch of inputs and
        evaluates the function we want to maximize for each input.

    Returns:
      best_continuous_action: The sample value that maximizes `objective_fn`.
      debug: A dictionary of debug information from CEM optimization.
    """

    def update_fn(params, elite_samples):
      del params
      return {
          'mean': np.mean(elite_samples, axis=0),
          'stddev': np.std(elite_samples, axis=0, ddof=1),
      }

    mu = np.zeros(self._action_size)
    initial_params = {'mean': mu, 'stddev': np.ones(self._action_size)}
    samples, values, final_params = cross_entropy.CrossEntropyMethod(
        self.sample_fn,
        objective_fn,
        update_fn,
        initial_params,
        num_elites=self._num_elites,
        num_iterations=self._cem_iters)

    idx = np.argmax(values)
    best_continuous_action, best_continuous_value = samples[idx], values[idx]
    debug = {'q_predicted': best_continuous_value, 'final_params': final_params,
             'best_idx': idx}
    return best_continuous_action, debug

  def _default_pack_fn(self, t2r_model, state, context, timestep, samples):
    return t2r_model.pack_features(state, context, timestep, samples)

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name

    def objective_fn(samples):
      """The q value as obtained from the internal tf model."""
      # TODO(T2R_CONTRIBUTORS): remove this disable when the bug is fixed
      # pytype: disable=attribute-error
      np_inputs = self.pack_fn(self._t2r_model, state, context, timestep,
                               samples)
      # b/77299790 -> self._predictions.q OutputSpecification.
      q_values = self._predictor.predict(np_inputs)['q_predicted']
      # pytype: enable=attribute-error
      return q_values
    action, _ = self.get_cem_action(objective_fn)
    return action


@gin.configurable
class LSTMCEMPolicy(CEMPolicy):
  """Like CEMPolicy, but caches hidden state of critic."""

  def __init__(self, hidden_state_size, **kwargs):
    self._hidden_state_size = hidden_state_size
    super(LSTMCEMPolicy, self).__init__(**kwargs)

  def reset(self):
    # Instead of storing episode, cache hidden state returned on previous iter.
    self._hidden_state = np.zeros((self._hidden_state_size,), dtype=np.float32)

  def SelectAction(self, state, context, timestep):

    def objective_fn(samples):
      """The q value as obtained from the internal tf model."""
      # TODO(T2R_CONTRIBUTORS): remove this disable when the bug is fixed
      # pytype: disable=attribute-error
      # TODO(T2R_CONTRIBUTORS): Replace t2r_model with pack feature function.
      np_inputs = self.pack_fn(self._t2r_model, state, self._hidden_state,
                               timestep, samples)
      # This is a bit of a hack, but we cache all hidden states per CEM iter
      # query the selected hidden state by idx after CEM completed.
      predictions = self._predictor.predict(np_inputs)
      q_values = predictions['q_predicted']
      self._hidden_state_batch = predictions['lstm_hidden_state']
      # pytype: enable=attribute-error
      return q_values
    action, debug = self.get_cem_action(objective_fn)
    # Hidden state for the next round.
    self._hidden_state = self._hidden_state_batch[debug['best_idx']]
    return action


@gin.configurable
class RegressionPolicy(Policy):
  """CEM policy for continuous-action regression models."""

  # TODO(T2R_CONTRIBUTORS): Replace t2r_model with pack feature function.
  def __init__(self, t2r_model,
               **parent_kwargs):
    super(RegressionPolicy, self).__init__(**parent_kwargs)
    self._t2r_model = t2r_model

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    # pytype: disable=attribute-error
    np_inputs = self._t2r_model.pack_features(state, context, timestep)
    action = self._predictor.predict(np_inputs)['inference_output']
    # pytype: enable=attribute-error
    return action[0]


@gin.configurable
class SequentialRegressionPolicy(RegressionPolicy):
  """Store past packed frames into context."""

  def reset(self):
    # Sequence Context stores last input.
    self._sequence_context = None

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    # pylint: disable=attribute-error
    np_inputs = self._pack_features_fn(
        state=state, context=self._sequence_context, timestep=timestep)
    self._sequence_context = np_inputs
    action = self._predict_fn(np_inputs)['inference_output']
    # pylint: enable=attribute-error
    return action[0]


@gin.configurable
class OUExploreRegressionPolicy(Policy):
  """Adds action noise generated by an Ornstein-Uhlenbeck process."""

  # TODO(T2R_CONTRIBUTORS): Replace t2r_model with pack feature function.
  def __init__(self,
               t2r_model,
               action_size=2,
               theta=.2,
               sigma=.15,
               use_noise=True,
               **parent_kwargs):
    super(OUExploreRegressionPolicy, self).__init__(**parent_kwargs)
    self._t2r_model = t2r_model
    self.theta, self.sigma, self.mu = theta, sigma, 0
    self._action_size = action_size
    self._x_t = np.zeros(action_size)
    self._use_noise = use_noise

  def ou_step(self):
    theta, mu, sigma = self.theta, self.mu, self.sigma
    dx_t = theta * (mu - self._x_t) + sigma * np.random.randn(*self._x_t.shape)
    self._x_t = self._x_t + dx_t
    return self._x_t

  def reset(self):
    self._x_t = np.zeros(self._action_size)

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    # pytype: disable=attribute-error
    np_inputs = self._t2r_model.pack_features(state, context, timestep)
    action = self._predictor.predict(np_inputs)['inference_output']
    # pytype: enable=attribute-error
    noise = self.ou_step() if self._use_noise else 0
    return action[0] + noise


@gin.configurable
class ScheduledExplorationRegressionPolicy(Policy):
  """Adds gaussian action noise according to a linear stddev schedule."""

  # TODO(T2R_CONTRIBUTORS): Replace t2r_model with pack feature function.
  def __init__(self,
               t2r_model,
               action_size=2,
               stddev_0=0.2,
               slope=0,
               **parent_kwargs):
    super(ScheduledExplorationRegressionPolicy, self).__init__(**parent_kwargs)
    self._t2r_model = t2r_model
    self._action_size = action_size
    self._stddev_0 = stddev_0
    self._slope = slope

  def get_noise(self):
    stddev = max(self._stddev_0 + self.global_step * self._slope, 0)
    return stddev * np.random.randn(self._action_size)

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    # pytype: disable=attribute-error
    np_inputs = self._t2r_model.pack_features(state, context, timestep)
    action = self._predictor.predict(np_inputs)['inference_output']
    # pytype: enable=attribute-error
    return action[0] + self.get_noise()


@gin.configurable
class PerEpisodeSwitchPolicy(Policy):
  """Randomly uses exploration policy or a greedy policy per-episode.

  A typical use case would be a scripted policy used to get some reasonable
  amount of random successes, and a greedy policy that is learned.

  Each of the exploration and greedy policies can still perform their own
  exploration actions after being selected by the PerEpisodeSwitchPolicy.
  """

  def __init__(self, explore_policy_class, greedy_policy_class, explore_prob,
               **parent_kwargs):
    super(PerEpisodeSwitchPolicy, self).__init__(**parent_kwargs)
    self._explore_policy = explore_policy_class()
    self._greedy_policy = greedy_policy_class()
    self._explore_prob = explore_prob
    self._active_policy = None

  def reset(self):
    self._explore_policy.reset()
    self._greedy_policy.reset()
    if np.random.random() < self._explore_prob:
      self._active_policy = self._explore_policy
    else:
      self._active_policy = self._greedy_policy

  def init_randomly(self):
    self._explore_policy.init_randomly()
    self._greedy_policy.init_randomly()

  def restore(self):
    self._explore_policy.restore()
    self._greedy_policy.restore()

  @property
  def global_step(self):
    """Returns the global step of the greedy policy."""
    return self._greedy_policy.global_step

  def SelectAction(self, state, context, timestep):  # pylint: disable=invalid-name
    return self._active_policy.SelectAction(state, context, timestep)
