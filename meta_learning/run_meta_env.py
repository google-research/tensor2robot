# coding=utf-8
# Copyright 2023 The Tensor2Robot Authors.
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

"""Stepping & evaluating meta-learning policies in a Gym environment.
"""

import collections
import copy
import datetime
import os

import gin
import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf


@gin.configurable
def run_meta_env(env,
                 policy=None,
                 demo_policy_cls=None,
                 explore_schedule=None,
                 episode_to_transitions_fn=None,
                 replay_writer=None,
                 root_dir=None,
                 task=0,
                 global_step=0,
                 num_episodes=None,
                 num_tasks=10,
                 num_adaptations_per_task=2,
                 num_episodes_per_adaptation=1,
                 num_demos=1,
                 break_after_one_task=False,
                 tag='collect',
                 write_tf_summary=False):
  """Runs agent+env loop and log performance + collect data.

  Interpolates between an exploration policy and greedy policy according to a
  explore_schedule. Run this function separately for collect/eval.

  Args:
    env: Gym environment.
    policy: Meta-Learning Policy to collect/evaluate.
    demo_policy_cls: Class for the policy that will replay demonstration files
      in the environment.
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
    task: Task number for replica trials for a given experiment. Note that this
      is usually the FLAGS.task passed by distributed job replicas, and has
      nothing to do with the meta-learning context.
    global_step: Training step corresponding to policy checkpoint.
    num_episodes: This argument is ignored. num_tasks should be provided
      instead.
    num_tasks: Number of tasks to run.
    num_adaptations_per_task: Number of adaptation steps to run within each
      task.
    num_episodes_per_adaptation: Number of episodes to run within each
      adaptation step.
    num_demos: Number of demosntration episodes to condition on.
    break_after_one_task: Returns after one task. Useful for testing purposes.
    tag: String prefix for evaluation summaries and collect data.
    write_tf_summary: If True, write episode stats to tf event files.
  """
  del num_episodes

  # Episode rewards for each step, for each episode. This structure is a dict of
  # dict of lists.
  task_step_rewards = collections.defaultdict(
      lambda: collections.defaultdict(list))
  # Same structure, but aggregated across task families.
  task_family_step_rewards = collections.defaultdict(
      lambda: collections.defaultdict(list))

  # track episode rewards within each trial to see whether they increase.
  # throughout each trial via fast adaptation.
  episode_q_values = collections.defaultdict(list)

  summary_writer = None
  if root_dir and write_tf_summary:
    summary_dir = os.path.join(root_dir, 'live_eval_%d' % task)
    summary_writer = tf.summary.FileWriter(summary_dir)

  for task_idx in range(num_tasks):
    if hasattr(policy, 'reset_task'):
      policy.reset_task()
    env.reset_task()
    task_family = None
    if hasattr(env, 'get_task_family'):
      task_family = env.get_task_family()
    # Save each task in a different recordio.
    if root_dir and replay_writer:
      timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
      record_name = os.path.join(
          root_dir, 'gs%d_t%d_%s_%d' %
          (global_step, task, timestamp, task_idx))
    if replay_writer:
      replay_writer.open(record_name)

    # A list of episode data (each episode is a list of transition tuples) to
    # condition on. Transitions from the demo vs. retrial are adapted together
    # in a single gradient step.
    condition_data = []
    if hasattr(env, 'get_demonstration') and hasattr(policy, 'adapt'):
      for _ in range(num_demos):
        obs = env.reset()
        demo_policy = demo_policy_cls(env)
        episode_data = []
        while True:
          action, debug = demo_policy.sample_action(obs, 0)
          if action is None:
            break
          next_obs, rew, done, debug = env.step(action)
          debug['is_demo'] = True
          episode_data.append((obs, action, rew, next_obs, done, debug))
          obs = next_obs
        condition_data.append(episode_data)

        if replay_writer and episode_to_transitions_fn:
          # Note: is_demo arg assumes VRGripper's ep to transitions fn.
          replay_writer.write(
              episode_to_transitions_fn(episode_data, is_demo=True))

          # TODO(allanz): This only works if env.disable_trial_randomization.
          # Record the demo actions for the inference scene (the scene where the
          # policy will do its trials). We need this to for the second demo of
          # our (demo, trial, demo) triplet.
          obs = env.reset()
          demo_policy = demo_policy_cls(env)
          episode_data = []
          while True:
            action, debug = demo_policy.sample_action(obs, 0)
            if action is None:
              break
            next_obs, rew, done, debug = env.step(action)
            debug['is_demo'] = True
            episode_data.append((obs, action, rew, next_obs, done, debug))
            obs = next_obs
          replay_writer.write(
              episode_to_transitions_fn(episode_data, is_demo=True))

      policy.adapt(copy.copy(condition_data))
    # Unlike VRGripperEnv, HVGripperEnv directly retrieves demonstration data
    # from a record-backed dataset and does not need a demo policy.
    elif hasattr(env, 'task_data') and hasattr(policy, 'adapt'):
      env_task_data = env.task_data
      for episode_name, episode_data in env_task_data.items():
        if six.ensure_str(episode_name).startswith('condition_ep'):
          condition_data.append(episode_data)
      policy.adapt(copy.copy(condition_data))

    for step_num in range(num_adaptations_per_task):
      # Fast adaptation of the policy with prior data.
      if step_num != 0 and hasattr(policy, 'adapt'):
        policy.adapt(copy.copy(condition_data))
      for ep in range(num_episodes_per_adaptation):
        done, env_step, episode_reward, episode_data = (False, 0, 0.0, [])
        policy.reset()
        obs = env.reset()
        if explore_schedule:
          explore_prob = explore_schedule.value(global_step)
        else:
          explore_prob = 0
        # Run the episode.
        while not done:
          debug = {}
          action, policy_debug = policy.sample_action(obs, explore_prob)
          if policy_debug is not None:
            debug.update(policy_debug)
          if policy_debug and 'q_predicted' in policy_debug:
            episode_q_values[env_step].append(policy_debug['q_predicted'])
          new_obs, rew, done, env_debug = env.step(action)
          debug.update(env_debug)
          env_step += 1
          episode_reward += rew

          episode_data.append((obs, action, rew, new_obs, done, debug))
          obs = new_obs
          if done:
            tf.logging.info(
                'Step %d episode %d reward: %f',
                step_num, ep, episode_reward)
            # Record reward.
            task_step_rewards[task_idx][step_num].append(episode_reward)
            if task_family is not None:
              task_family_step_rewards[task_family][step_num].append(
                  episode_reward)
            if replay_writer:
              transitions = episode_to_transitions_fn(episode_data)
              replay_writer.write(transitions)
        condition_data.append(episode_data)
    avg_ep_rew_in_task = np.mean(
        task_step_rewards[task_idx][num_adaptations_per_task-1])
    tf.logging.info(
        'Task %d avg reward: %f',
        task_idx, avg_ep_rew_in_task)
    # if task_family is not None:
    #   task_rewards_by_family[task_family].append(avg_ep_rew_in_task)
    tf.logging.info('Average Task reward: %f', avg_ep_rew_in_task)

    if replay_writer:
      replay_writer.close()
    if break_after_one_task:
      break

  if summary_writer:
    # Track average task rewards of episodes at each adapt step.
    summary_values = []
    for step_num in range(num_adaptations_per_task):
      step_rewards = [
          np.mean(task_step_rewards[t][step_num]) for t in range(num_tasks)]
      # Reward averaged across tasks for this step.
      summary_values.append(tf.Summary.Value(
          tag='%s/step_%d_reward' % (tag, step_num),
          simple_value=np.mean(step_rewards)
      ))
      # Average task improvement delta.
      if step_num > 0:
        delta = np.mean(
            [np.array(task_step_rewards[t][step_num]) -
             np.array(task_step_rewards[t][step_num-1])
             for t in range(num_tasks)])
        summary_values.append(tf.Summary.Value(
            tag='%s/step_%d_improvement' % (tag, step_num),
            simple_value=delta))

    for step, q_values in episode_q_values.items():
      summary_values.append(tf.Summary.Value(tag='%s/Q/%d' % (tag, step),
                                             simple_value=np.mean(q_values)))
    for task_family, step_rewards in task_family_step_rewards.items():
      # Reward on the last adaptation step.
      rewards = step_rewards[num_adaptations_per_task-1]
      summary_values.append(tf.Summary.Value(
          tag='%s/task_family_%d_reward' % (tag, task_family),
          simple_value=np.mean(rewards)))
    summary = tf.Summary(value=summary_values)
    summary_writer.add_summary(summary, global_step)
