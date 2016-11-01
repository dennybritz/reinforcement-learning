import gym
import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops


class Worker(object):
    def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor, summary_writer=None):
        self.name = name
        self.discount_factor = discount_factor
        self.global_step = tf.contrib.framework.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.sp = StateProcessor()
        self.summary_writer = summary_writer
        self.env = env

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = PolicyEstimator(policy_net.num_outputs)
            self.value_net = ValueEstimator(reuse=True)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.state = None

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # Create a new env for this thead
            self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
            try:
                while not coord.should_stop():
                    # Copy Parameters from the global nets
                    sess.run(self.copy_params_op)

                    # Run on iteration to collect gradients
                    transitions = self.run_n_steps(t_max, sess)

                    _, _, policy_net_summaries, value_net_summaries = self.apply_gradients(transitions, sess)

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, sess):
        feed_dict = { self.policy_net.states: [state] }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["probs"][0]

    def _value_net_predict(self, state, sess):
        feed_dict = { self.value_net.states: [state] }
        preds = sess.run(self.value_net.predictions, feed_dict)
        return preds["logits"][0]

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in range(n):
            # Take a step
            action_probs = self._policy_net_predict(self.state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self.env.step(action)
            next_state = atari_helpers.atari_make_next_state(self.state, self.sp.process(next_state))

            # Store transition
            transitions.append(Transition(
              state=self.state, action=action, reward=reward, next_state=next_state, done=done))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                self.state = atari_helpers.atari_make_initial_state(self.sp.process(self.env.reset()))
                break
            else:
                self.state = next_state
        return transitions

    def apply_gradients(self, transitions, sess):
        reward = 0.0
        if not transitions[-1].done:
            # Bootstrap from the last state
            reward = self._value_net_predict(self.state, sess)

        # Accumulate examples
        states = []
        policy_targets = []
        value_targets = []
        actions = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            policy_target = (reward - self._value_net_predict(transition.state, sess))

            # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        # Apply policy net update
        feed_dict = {
            self.global_policy_net.states: np.array(states),
            self.global_policy_net.targets: policy_targets,
            self.global_policy_net.actions: actions,
            self.global_value_net.states: np.array(states),
            self.global_value_net.targets: value_targets,
        }

        global_step, policy_net_loss, policy_net_summaries, value_net_loss, value_net_summaries = sess.run(
            [self.global_step, self.global_policy_net.train_op, self.global_policy_net.summaries, self.global_value_net.train_op, self.global_value_net.summaries],
            feed_dict)

        if self.summary_writer is not None:
            self.summary_writer.add_summary(policy_net_summaries, global_step)
            self.summary_writer.add_summary(value_net_summaries, global_step)
            self.summary_writer.flush()

        return policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries
