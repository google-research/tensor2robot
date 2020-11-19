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
"""Library function for stepping/evaluating a policy in a Gym environment.

Also supports TF-Agents environments.
"""

import collections
import datetime
import os
import gin
import numpy as np
import PIL.Image as Image
import six
from six.moves import range
import tensorflow.compat.v1 as tf
from tf_agents.trajectories import time_step as ts


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def _gym_env_reset(env):
  obs = env.reset()
  return obs


def _gym_env_step(env, action):
  """Step through Gym env and return (o, r, done, debug) tuple."""
  new_obs, rew, done, env_debug = env.step(action)
  return new_obs, rew, done, env_debug


def _tfagents_env_reset(env):
  obs = env.reset().observation
  return obs


def _tfagents_env_step(env, action):
  """Step through TF-Agents env and return (o, r, done, debug) tuple."""
  timestep = env.step(action)
  new_obs = timestep.observation
  rew = timestep.reward
  done = timestep.step_type == ts.StepType.LAST
  env_debug = ()
  return new_obs, rew, done, env_debug


@gin.configurable(denylist=['global_step', 'tag'])
def run_env(env,
            policy=None,
            explore_schedule=None,
            episode_to_transitions_fn=None,
            replay_writer=None,
            root_dir=None,
            task=0,
            global_step=0,
            num_episodes=100,
            tag='collect'):
  """Runs agent+Gym env loop num_episodes times, logging performance."""
  return _run_env(
      env,
      reset_fn=_gym_env_reset,
      step_fn=_gym_env_step,
      policy=policy,
      explore_schedule=explore_schedule,
      episode_to_transitions_fn=episode_to_transitions_fn,
      replay_writer=replay_writer,
      root_dir=root_dir,
      task=task,
      global_step=global_step,
      num_episodes=num_episodes,
      tag=tag,
      unpack_action=False)


@gin.configurable(denylist=['global_step', 'tag'])
def run_tfagents_env(env,
                     policy=None,
                     explore_schedule=None,
                     episode_to_transitions_fn=None,
                     replay_writer=None,
                     root_dir=None,
                     task=0,
                     global_step=0,
                     num_episodes=100,
                     tag='collect'):
  """Runs agent+TF-Agents env loop num_episodes times, logging performance."""
  return _run_env(
      env,
      reset_fn=_tfagents_env_reset,
      step_fn=_tfagents_env_step,
      policy=policy,
      explore_schedule=explore_schedule,
      episode_to_transitions_fn=episode_to_transitions_fn,
      replay_writer=replay_writer,
      root_dir=root_dir,
      task=task,
      global_step=global_step,
      num_episodes=num_episodes,
      tag=tag,
      unpack_action=True)


def _run_env(env,
             reset_fn,
             step_fn,
             policy,
             explore_schedule,
             episode_to_transitions_fn,
             replay_writer,
             root_dir,
             task,
             global_step,
             num_episodes,
             tag,
             unpack_action):
  """Runs agent+env loop num_episodes times and log performance + collect data.

  Interpolates between an exploration policy and greedy policy according to a
  explore_schedule. Run this function separately for collect/eval.

  Args:
    env: Environment.
    reset_fn: A function that resets the environment, returning starting obs.
    step_fn: A function that steps the environment, returning next_obs, reward,
      done, and debug.
    policy: Policy to collect/evaluate.
    explore_schedule: Exploration schedule that defines a `value(t)` function
      to compute the probability of exploration as a function of global step t.
    episode_to_transitions_fn: Function that converts episode data to transition
      protobufs (e.g. TFExamples).
    replay_writer: Instance of a replay writer that writes a list of transition
      protos to disk (optional).
    root_dir: Root directory of the experiment summaries and data collect. If
      replay_writer is specified, data is written to the `policy_*` subdirs.
      Setting root_dir=None results in neither summaries or transitions being
      saved to disk.
    task: Task number for replica trials for a given experiment. Debug summaries
      are only written when task == 0.
    global_step: Training step corresponding to policy checkpoint.
    num_episodes: Number of episodes to run.
    tag: String prefix for evaluation summaries and collect data.
    unpack_action: Whether the action output is wrapped in batch_size 1 or is
      returned as is. If True, unpack the outer dimension.
  """
  episode_rewards = []
  episode_q_values = collections.defaultdict(list)

  if root_dir and replay_writer:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    record_prefix = os.path.join(root_dir, 'policy_%s' % tag,
                                 'gs%d_t%d_%s' % (global_step, task, timestamp))
  if root_dir and task == 0:
    summary_dir = os.path.join(root_dir, 'live_eval_%d' % task)
    summary_writer = tf.summary.FileWriter(summary_dir)

  if replay_writer:
    replay_writer.open(record_prefix)

  for ep in range(num_episodes):
    done, env_step, episode_reward, episode_data = (False, 0, 0.0, [])
    policy.reset()
    obs = reset_fn(env)
    if explore_schedule:
      explore_prob = explore_schedule.value(global_step)
    else:
      explore_prob = 0
    while not done:
      action, policy_debug = policy.sample_action(obs, explore_prob)
      if unpack_action:
        action = action[0]
      if policy_debug and 'q' in policy_debug:
        episode_q_values[env_step].append(policy_debug['q'])
      new_obs, rew, done, env_debug = step_fn(env, action)
      env_step += 1
      episode_reward += rew

      episode_data.append((obs, action, rew, new_obs, done, env_debug))
      obs = new_obs
      if done:
        tf.logging.info('Episode %d reward: %f' % (ep, episode_reward))
        episode_rewards.append(episode_reward)
        if replay_writer:
          transitions = episode_to_transitions_fn(episode_data)
          replay_writer.write(transitions)
    if episode_rewards and len(episode_rewards) % 10 == 0:
      tf.logging.info('Average %d collect episodes reward: %f' %
                      (len(episode_rewards), np.mean(episode_rewards)))

  tf.logging.info('Closing environment.')
  env.close()

  if replay_writer:
    replay_writer.close()

  if root_dir and task == 0:
    summary_values = [
        tf.Summary.Value(
            tag='%s/episode_reward' % tag,
            simple_value=np.mean(episode_rewards))
    ]
    for step, q_values in episode_q_values.items():
      summary_values.append(
          tf.Summary.Value(
              tag='%s/Q/%d' % (tag, step), simple_value=np.mean(q_values)))
    summary = tf.Summary(value=summary_values)
    summary_writer.add_summary(summary, global_step)
