import gym
import sys
import os
import itertools
import collections
import unittest
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
from worker import Worker
from estimators import ValueEstimator, PolicyEstimator

def make_env():
  return gym.envs.make("Breakout-v0")

VALID_ACTIONS = [0, 1, 2, 3]

class WorkerTest(tf.test.TestCase):
  def setUp(self):
    super(WorkerTest, self).setUp()

    self.env = make_env()
    self.discount_factor = 0.99
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.global_counter = itertools.count()
    self.sp = StateProcessor()

    with tf.variable_scope("global") as vs:
      self.global_policy_net = PolicyEstimator(len(VALID_ACTIONS))
      self.global_value_net = ValueEstimator(reuse=True)

  def testPolicyNetPredict(self):
    w = Worker(
      name="test",
      env=make_env(),
      policy_net=self.global_policy_net,
      value_net=self.global_value_net,
      global_counter=self.global_counter,
      discount_factor=self.discount_factor)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      state = self.sp.process(self.env.reset())
      processed_state = atari_helpers.atari_make_initial_state(state)
      action_values = w._policy_net_predict(processed_state, sess)
      self.assertEqual(action_values.shape, (4,))


  def testValueNetPredict(self):
    w = Worker(
      name="test",
      env=make_env(),
      policy_net=self.global_policy_net,
      value_net=self.global_value_net,
      global_counter=self.global_counter,
      discount_factor=self.discount_factor)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      state = self.sp.process(self.env.reset())
      processed_state = atari_helpers.atari_make_initial_state(state)
      state_value = w._value_net_predict(processed_state, sess)
      self.assertEqual(state_value.shape, ())

  def testRunNStepsAndUpdate(self):
    w = Worker(
      name="test",
      env=make_env(),
      policy_net=self.global_policy_net,
      value_net=self.global_value_net,
      global_counter=self.global_counter,
      discount_factor=self.discount_factor)

    with self.test_session() as sess:
      sess.run(tf.initialize_all_variables())
      state = self.sp.process(self.env.reset())
      processed_state = atari_helpers.atari_make_initial_state(state)
      w.state = processed_state
      transitions, local_t, global_t = w.run_n_steps(10, sess)
      policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries = w.update(transitions, sess)

    self.assertEqual(len(transitions), 10)
    self.assertIsNotNone(policy_net_loss)
    self.assertIsNotNone(value_net_loss)
    self.assertIsNotNone(policy_net_summaries)
    self.assertIsNotNone(value_net_summaries)


if __name__ == '__main__':
  unittest.main()