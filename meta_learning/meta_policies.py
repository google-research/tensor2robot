# coding=utf-8
# Copyright 2021 The Tensor2Robot Authors.
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

"""Policy abstraction for TFModels."""

import abc
import gin
import numpy as np
import six
from tensor2robot.meta_learning import meta_tf_models
from tensor2robot.policies import policies


@six.add_metaclass(abc.ABCMeta)
class MetaLearningPolicy(policies.Policy):
  """Abstract class for Tensorflow-based Meta-learning policies.
  """

  def reset_task(self):
    pass

  @abc.abstractmethod
  def adapt(self, episode_data):
    raise NotImplementedError()


@gin.configurable
class MAMLCEMPolicy(MetaLearningPolicy, policies.CEMPolicy):
  """CEM Policy that uses MAML/gradient descent for fast adaptation."""

  # TODO(T2R_CONTRIBUTORS) Replace t2r_model with pack feature function.
  def __init__(self,
               t2r_model,
               action_size = 2,
               cem_iters = 3,
               cem_samples = 64,
               num_elites = 10,
               **parent_kwargs):
    self._cem_iters = cem_iters
    self._cem_samples = cem_samples
    self._action_size = action_size
    self._num_elites = num_elites
    super(MAMLCEMPolicy,
          self).__init__(t2r_model, action_size, cem_iters, cem_samples,
                         num_elites, **parent_kwargs)

  def reset_task(self):
    self._prev_episode_data = None

  def adapt(self, episode_data):
    self._prev_episode_data = episode_data

  def SelectAction(self, state, context, timestep):

    # TODO(T2R_CONTRIBUTORS)
    if self._prev_episode_data:
      prediction_key = 'val_output'
    else:
      prediction_key = 'train_output'

    def objective_fn(samples):
      """The CEM objective function.

      Args:
        samples: The samples we evaluate the network on.

      Returns:
        q_values: The predicted Q values.
      """
      # TODO(chelseaf): This is the inefficient way to do it. Would be faster
      # to tile after the conv layers.
      cem_state = np.tile(np.expand_dims(state, 0), [samples.shape[0], 1, 1, 1])
      np_inputs = self._t2r_model.pack_features(cem_state,
                                                self._prev_episode_data,
                                                timestep, samples)
      q_values = self._predictor.predict(np_inputs)[prediction_key]
      if not self._prev_episode_data:
        q_values *= 0
      return q_values[0]

    action, _ = self.get_cem_action(objective_fn)
    return action


@gin.configurable
class MAMLRegressionPolicy(MetaLearningPolicy, policies.RegressionPolicy):
  """Actor network policy that uses MAML/gradient descent for fast adaptation.

     This is basically the same as the RL2 policy (only changed t2r_model)
  """

  def reset_task(self):
    self._prev_episode_data = None

  def adapt(self, episode_data):
    self._prev_episode_data = episode_data

  def sample_action(self, obs, explore_prob):
    del explore_prob
    action = self.SelectAction(obs, None, None)
    # Replay writers require the is_demo flag when forming MetaExamples.
    debug = {'is_demo': False}
    return action, debug

  def SelectAction(self, state, context, timestep):
    np_features = self._t2r_model.pack_features(state, self._prev_episode_data,
                                                timestep)
    # This key must be 'inference_output' b.c. MAMLModel performs a check for
    # this key.
    action = self._predictor.predict(np_features)['inference_output']

    # TODO(allanz): Rank 4 actions are due to VRGripperRegressionModel having
    # an additional time dimension. Remove this once we have a better way to
    # handle multiple timesteps.
    if len(action.shape) == 4:
      return action[0, 0, 0]
    elif len(action.shape) == 3:
      return action[0, 0]
    else:
      raise ValueError('Invalid action rank.')


@gin.configurable
class FixedLengthSequentialRegressionPolicy(
    MetaLearningPolicy, policies.RegressionPolicy):
  """Fixed Episode Length sequential policy. a_t is t'th output of model."""

  def reset_task(self):
    # prev_episode_data is the conditioning episode, e.g. a demo.
    self._prev_episode_data = None

  def adapt(self, episode_data):
    self._prev_episode_data = episode_data

  def reset(self):
    # current episode data is the temporal context for the current episode.
    self._current_episode_data = None
    self._t = 0

  def SelectAction(self, state, context, timestep):
    np_features = self._t2r_model.pack_features(state,
                                                self._prev_episode_data,
                                                self._current_episode_data,
                                                self._t)
    # Action is [batch, inference_episode, T, action_dim].
    action = self._predictor.predict(np_features)['inference_output']
    self._current_episode_data = np_features
    assert len(action.shape) == 4
    a = action[0, 0, self._t]
    self._t += 1
    return a


@gin.configurable
class ScheduledExplorationMAMLRegressionPolicy(
    MetaLearningPolicy, policies.ScheduledExplorationRegressionPolicy):
  """Like MAMLRegressionPolicy, but with scheduled action noise for exploration.
  """

  def reset_task(self):
    self._prev_episode_data = None

  def adapt(self, episode_data):
    self._prev_episode_data = episode_data

  def sample_action(self, obs, explore_prob):
    del explore_prob
    action = self.SelectAction(obs, None, None)
    # Replay writers require the is_demo flag when forming MetaExamples.
    debug = {'is_demo': False}
    return action, debug

  def SelectAction(self, state, context, timestep):
    del context
    np_features = self._t2r_model.pack_features(state, self._prev_episode_data,
                                                timestep)
    action = self._predictor.predict(np_features)['inference_output']

    # TODO(allanz): Rank 4 actions are due to VRGripperRegressionModel having
    # an additional time dimension. Remove this once we have a better way to
    # handle multiple timesteps.
    if len(action.shape) == 4:
      action = action[0, 0, 0]
    elif len(action.shape) == 3:
      action = action[0, 0]
    else:
      raise ValueError('Invalid action rank.')

    return action + self.get_noise()
